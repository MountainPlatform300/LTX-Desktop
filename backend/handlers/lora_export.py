"""Build and parse portable LoRA dataset bundles (export + import).

A bundle is a self-contained folder (optionally zipped) that serves two
goals at once:

1. **Train anywhere** — `dataset.json` + `clips/` use *relative* media
   paths, which is exactly what the official LTX-2 trainer's
   `process_dataset.py` consumes. Drop the folder on a GPU box and run
   the two commands in the bundled `README.md`.
2. **Move between machines** — `ltxdesktop.json` is a sidecar manifest
   the trainer ignores but our importer reads, so another LTX Desktop
   can re-create the dataset losslessly (captions, trigger word,
   IC-LoRA pairing, triage, origin).

Layout::

    <name>/
      dataset.json        # standard: [{caption, media_path}]
                          # IC-LoRA:  [{caption, video, reference_video}] (pairs)
      clips/              # included media (IC-LoRA: normalized + paired)
      ltxdesktop.json     # portable manifest for re-import
      train_config.yaml   # ready-to-run trainer config (placeholder paths)
      README.md           # exact terminal commands

This module is pure plumbing: it copies files and (de)serializes JSON
given fully-resolved paths, so the handler stays a thin orchestrator and
the logic is trivially unit-testable.
"""

from __future__ import annotations

import json
import re
import shutil
import stat
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from handlers.lora_config_builder import build_training_yaml
from state.lora_training_state import (
    ClipOrigin,
    ClipTriage,
    LoraClip,
    LoraDataset,
    LoraDatasetType,
    TrainingConfig,
)

if TYPE_CHECKING:
    from handlers.lora_dataset_prep import EmittedPair, PrepOptions, PrepReport
    from services.clip_processor.clip_processor import ClipProcessor

MANIFEST_NAME = "ltxdesktop.json"
MANIFEST_KIND = "ltx-desktop-lora-dataset"
MANIFEST_SCHEMA_VERSION = 1

_VALID_ORIGINS: frozenset[str] = frozenset({"imported", "gen_space", "ai_derived"})
_VALID_TRIAGE: frozenset[str] = frozenset({"keep", "reject"})


class BundleError(ValueError):
    """Raised when an import source is not a valid LTX Desktop bundle."""


@dataclass(frozen=True, slots=True)
class BundleComponents:
    """Which optional artifacts to write alongside the core dataset.

    The dataset itself (``clips/`` + ``dataset.json``) is always written — it's
    the point of the export. These toggle the supplementary files so the user
    can ship exactly what they need (see the export modal's checkboxes).
    """

    train_config: bool = True
    readme: bool = True
    manifest: bool = True
    model_card: bool = True


def safe_stem(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return (stem or "clip")[:48]


def safe_dirname(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned or "lora-dataset"


def refs_of(clip: LoraClip) -> list[str]:
    """All conditioning references for a clip (primary first), deduped."""
    out: list[str] = []
    for ref in ([clip.reference_path] if clip.reference_path else []) + clip.reference_paths:
        if ref and ref not in out:
            out.append(ref)
    return out


def plan_clip_layout(
    dataset: LoraDataset, clips: list[LoraClip]
) -> tuple[dict[str, str], list[LoraClip]]:
    """Assign each shipped clip a human-readable relative path under ``clips/``.

    The naming is the single source of truth shared by *both* the portable
    export bundle and the remote-upload staging, so the two can never drift.
    Names are cosmetic (the trainer follows the paths in ``dataset.json``) but
    a role/example prefix makes ``clips/`` self-organizing: for IC-LoRA every
    example's input(s) and output sort together and read at a glance, e.g.

        0001_input_walk.mp4, 0001_output_walk_edit.mp4
        0002_input-1_a.mp4, 0002_input-2_b.mp4, 0002_output_c.mp4

    Standard LoRA has no references, so a plain numbered sequence is clearest.

    ``clips`` is the (already-filtered) set to ship; an IC-LoRA example's
    inputs are pulled in from the full ``dataset.clips`` even when a referenced
    clip was filtered out, so ``reference_path``s always resolve. Returns
    ``(rel_by_local, order)`` — the local-path -> ``clips/<name>`` map and the
    deterministic copy/manifest order. Raises ``BundleError`` for a missing file.
    """
    by_local: dict[str, LoraClip] = {c.local_path: c for c in dataset.clips}
    is_ic = dataset.type == "ic_lora"
    rel_by_local: dict[str, str] = {}
    order: list[LoraClip] = []

    def _name(c: LoraClip, base: str) -> None:
        if c.local_path in rel_by_local:
            return  # shared input already placed under an earlier example
        src = Path(c.local_path)
        if not src.is_file():
            raise BundleError(f"Clip file is missing on disk: {c.local_path}")
        ext = src.suffix or ".mp4"
        rel_by_local[c.local_path] = f"clips/{base}_{safe_stem(src.stem)}{ext}"
        order.append(c)

    if is_ic:
        example = 0
        # Pass 1: grouped examples — each included clip that has resolvable
        # input(s) is an output; pull in its inputs (even if filtered out).
        for clip in clips:
            inputs = [by_local[r] for r in refs_of(clip) if r in by_local]
            if not inputs:
                continue
            example += 1
            _name(clip, f"{example:04d}_output")
            multi = len(inputs) > 1
            for i, ref_clip in enumerate(inputs, start=1):
                _name(ref_clip, f"{example:04d}_input-{i}" if multi else f"{example:04d}_input")
        # Pass 2: standalone clips (no inputs, not already placed as an input).
        for clip in clips:
            if clip.local_path in rel_by_local:
                continue
            example += 1
            _name(clip, f"{example:04d}_clip")
    else:
        for idx, clip in enumerate(clips, start=1):
            _name(clip, f"{idx:04d}")

    return rel_by_local, order


def build_dataset_rows(
    dataset: LoraDataset,
    clips: list[LoraClip],
    rel_by_local: dict[str, str],
    *,
    render_media: Callable[[str], str],
) -> list[dict[str, str]]:
    """Trainer-ready ``dataset.json`` rows — one per *included* clip.

    ``render_media`` maps a relative ``clips/<name>`` path to the value written
    into ``media_path`` / ``reference_path``: identity for a portable bundle
    (relative paths) or an absolute remote path for upload staging (the remote
    ``process_dataset.py`` runs with the trainer repo as its cwd, so relative
    media paths wouldn't resolve there). For IC-LoRA the primary resolvable
    reference is emitted as ``reference_path`` so the trainer can build
    ``reference_latents/``.
    """
    is_ic = dataset.type == "ic_lora"
    rows: list[dict[str, str]] = []
    for clip in clips:
        media = rel_by_local.get(clip.local_path)
        if media is None:
            continue
        row: dict[str, str] = {"caption": clip.caption, "media_path": render_media(media)}
        if is_ic:
            refs = refs_of(clip)
            primary = next((r for r in refs if r in rel_by_local), None)
            if primary is not None:
                row["reference_path"] = render_media(rel_by_local[primary])
        rows.append(row)
    return rows


def build_bundle(
    *,
    dataset: LoraDataset,
    clips: list[LoraClip],
    staging_dir: Path,
    config: TrainingConfig | None = None,
    processor: "ClipProcessor | None" = None,
    options: "PrepOptions | None" = None,
    components: BundleComponents | None = None,
) -> "PrepReport":
    """Write a complete bundle into ``staging_dir``. Returns a `PrepReport`.

    ``clips`` is the (already-filtered) set to ship. Two shapes:

    - **IC-LoRA** routes through `lora_dataset_prep`: it ships ONE
      ``{caption, video, reference_video}`` record per *normalized, validated*
      pair (a `processor` is required); unusable pairs are dropped and reported.
    - **Standard** keeps the legacy numbered ``media_path`` layout.

    ``config`` seeds the bundled ``train_config.yaml`` (a chosen training
    profile); ``None`` uses the trainer defaults. ``components`` selects which
    supplementary files to write (config / readme / manifest / model card); the
    dataset itself is always written.
    """
    # Imported lazily to avoid a module import cycle (prep imports helpers here).
    from handlers import lora_dataset_prep as prep
    from handlers import lora_model_card

    parts = components or BundleComponents()

    clips_dir = staging_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    is_ic = dataset.type == "ic_lora"
    if is_ic:
        if processor is None:
            raise BundleError("IC-LoRA export requires a clip processor")
        report = prep.prepare_ic_lora_bundle(
            dataset=dataset,
            clips=clips,
            staging_dir=staging_dir,
            processor=processor,
            options=options or prep.PrepOptions(trigger_word=dataset.trigger_word),
            render_media=lambda rel: rel,
        )
        manifest_clips = _ic_manifest_clips(report.pairs)
    else:
        rel_by_local, order = plan_clip_layout(dataset, clips)
        for clip in order:
            shutil.copy2(Path(clip.local_path), staging_dir / rel_by_local[clip.local_path])
        rows = build_dataset_rows(dataset, clips, rel_by_local, render_media=lambda rel: rel)
        (staging_dir / "dataset.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )
        manifest_clips: list[dict[str, Any]] = []
        for clip in order:
            refs = [rel_by_local[r] for r in refs_of(clip) if r in rel_by_local]
            manifest_clips.append(
                {
                    "file": rel_by_local[clip.local_path],
                    "caption": clip.caption,
                    "origin": clip.origin,
                    "triage": clip.triage,
                    "references": refs,
                    "durationSeconds": clip.duration_seconds,
                }
            )
        report = prep.PrepReport(exported=len(rows), dropped=[], pairs=[])

    if parts.manifest:
        manifest: dict[str, Any] = {
            "kind": MANIFEST_KIND,
            "schemaVersion": MANIFEST_SCHEMA_VERSION,
            "name": dataset.name,
            "type": dataset.type,
            "triggerWord": dataset.trigger_word,
            "exportedAt": datetime.now(timezone.utc).isoformat(),
            "clips": manifest_clips,
        }
        (staging_dir / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    if parts.train_config:
        (staging_dir / "train_config.yaml").write_text(
            _default_config_yaml(dataset.type, config=config), encoding="utf-8"
        )
    if parts.readme:
        (staging_dir / "README.md").write_text(
            _readme(dataset, is_ic=is_ic, row_count=report.exported),
            encoding="utf-8",
        )
    if parts.model_card:
        (staging_dir / "MODEL_CARD.md").write_text(
            lora_model_card.build_template_card(dataset, config),
            encoding="utf-8",
        )
    return report


def _ic_manifest_clips(pairs: "list[EmittedPair]") -> list[dict[str, Any]]:
    """Lossless re-import entries for IC-LoRA: each pair becomes a reference
    clip (no refs) plus a target clip that references it, recreating the
    pairing when imported back into LTX Desktop."""
    out: list[dict[str, Any]] = []
    seen_refs: set[str] = set()
    for pair in pairs:
        if pair.reference_rel not in seen_refs:
            out.append(
                {
                    "file": pair.reference_rel,
                    "caption": "",
                    "origin": "imported",
                    "triage": None,
                    "references": [],
                    "durationSeconds": None,
                }
            )
            seen_refs.add(pair.reference_rel)
        out.append(
            {
                "file": pair.target_rel,
                "caption": pair.caption,
                "origin": "imported",
                "triage": None,
                "references": [pair.reference_rel],
                "durationSeconds": None,
            }
        )
    return out


def zip_dir(src_dir: Path, zip_path: Path) -> None:
    """Zip ``src_dir`` so the archive contains a single top-level folder."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src_dir.parent))


def find_manifest_root(base: Path) -> Path | None:
    """Locate the folder containing the manifest (handles a wrapping dir)."""
    if (base / MANIFEST_NAME).is_file():
        return base
    if base.is_dir():
        for child in sorted(base.iterdir()):
            if child.is_dir() and (child / MANIFEST_NAME).is_file():
                return child
    return None


def _is_within(parent: Path, child: Path) -> bool:
    """True if `child` is `parent` or nested under it (both resolved)."""
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def resolve_bundle_member(root: Path, relative_path: str) -> Path:
    """Resolve a manifest file reference without allowing root escape."""
    if not relative_path or "\x00" in relative_path:
        raise BundleError("Bundle contains an invalid empty file reference")
    root_resolved = root.resolve()
    relative = Path(relative_path)
    if relative.is_absolute() or relative.drive:
        raise BundleError(
            f"Refusing bundle file outside the import folder: {relative_path!r}"
        )
    lexical_target = root_resolved / relative
    for candidate in (lexical_target, *lexical_target.parents):
        if candidate == root_resolved:
            break
        if candidate.is_symlink():
            raise BundleError(
                f"Refusing symlinked bundle file: {relative_path!r}"
            )
    target = lexical_target.resolve()
    if not _is_within(root_resolved, target):
        raise BundleError(
            f"Refusing bundle file outside the import folder: {relative_path!r}"
        )
    return target


MAX_BUNDLE_FILES = 10_000
MAX_BUNDLE_UNCOMPRESSED_BYTES = 50 * 1024**3
MAX_BUNDLE_FILE_BYTES = 20 * 1024**3
MAX_BUNDLE_COMPRESSION_RATIO = 1_000


def _zip_member_kind(member: zipfile.ZipInfo) -> int:
    return (member.external_attr >> 16) & 0o170000


def safe_extractall(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract a bounded regular-file archive without following links."""
    dest_root = dest.resolve()
    members = zf.infolist()
    if len(members) > MAX_BUNDLE_FILES:
        raise BundleError(
            f"Bundle contains too many entries (maximum {MAX_BUNDLE_FILES})"
        )

    total_size = 0
    seen: set[str] = set()
    validated: list[tuple[zipfile.ZipInfo, Path]] = []
    for member in members:
        if "\x00" in member.filename:
            raise BundleError("Bundle contains an invalid file name")
        target = (dest / member.filename).resolve()
        if not _is_within(dest_root, target):
            raise BundleError(
                f"Refusing bundle entry outside the import folder: {member.filename!r}"
            )
        normalized = str(target).casefold()
        if normalized in seen:
            raise BundleError(f"Bundle contains a duplicate entry: {member.filename!r}")
        seen.add(normalized)
        kind = _zip_member_kind(member)
        if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise BundleError(
                f"Bundle contains an unsupported link or device: {member.filename!r}"
            )
        if member.flag_bits & 0x1:
            raise BundleError("Encrypted dataset bundles are not supported")
        if member.file_size > MAX_BUNDLE_FILE_BYTES:
            raise BundleError(
                f"Bundle entry is too large: {member.filename!r}"
            )
        total_size += member.file_size
        if total_size > MAX_BUNDLE_UNCOMPRESSED_BYTES:
            raise BundleError("Bundle expands beyond the allowed size")
        if (
            member.file_size > 10 * 1024**2
            and member.compress_size > 0
            and member.file_size / member.compress_size > MAX_BUNDLE_COMPRESSION_RATIO
        ):
            raise BundleError(
                f"Bundle entry has an unsafe compression ratio: {member.filename!r}"
            )
        validated.append((member, target))

    dest_root.mkdir(parents=True, exist_ok=True)
    extracted_total = 0
    for member, target in validated:
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        try:
            with zf.open(member) as source, target.open("xb") as output:
                while chunk := source.read(1024 * 1024):
                    written += len(chunk)
                    extracted_total += len(chunk)
                    if (
                        written > member.file_size
                        or extracted_total > MAX_BUNDLE_UNCOMPRESSED_BYTES
                    ):
                        raise BundleError(
                            f"Bundle entry exceeded its declared size: {member.filename!r}"
                        )
                    output.write(chunk)
        except (OSError, zipfile.BadZipFile) as exc:
            raise BundleError(
                f"Could not safely extract bundle entry {member.filename!r}: {exc}"
            ) from exc
        if written != member.file_size:
            raise BundleError(
                f"Bundle entry size did not match its manifest: {member.filename!r}"
            )


def read_manifest(root: Path) -> dict[str, Any]:
    """Parse + validate the manifest. Raises ``BundleError`` if invalid."""
    try:
        raw = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"Could not read {MANIFEST_NAME}: {exc}") from exc
    if not isinstance(raw, dict):
        raise BundleError("Not a valid LTX Desktop dataset bundle")
    data = cast(dict[str, Any], raw)
    if data.get("kind") != MANIFEST_KIND:
        raise BundleError("Not a valid LTX Desktop dataset bundle")
    if data.get("schemaVersion") != MANIFEST_SCHEMA_VERSION:
        raise BundleError(
            f"Unsupported bundle schema version {data.get('schemaVersion')!r}"
        )
    clips = data.get("clips")
    if not isinstance(clips, list):
        raise BundleError("Bundle manifest has no clips")
    clip_entries = cast(list[Any], clips)
    if len(clip_entries) > MAX_BUNDLE_FILES:
        raise BundleError(
            f"Bundle manifest contains too many clips (maximum {MAX_BUNDLE_FILES})"
        )
    if any(not isinstance(entry, dict) for entry in clip_entries):
        raise BundleError("Bundle manifest contains an invalid clip entry")
    return data


def coerce_origin(value: Any) -> ClipOrigin:
    return value if value in _VALID_ORIGINS else "imported"  # type: ignore[return-value]


def coerce_triage(value: Any) -> ClipTriage | None:
    return value if value in _VALID_TRIAGE else None  # type: ignore[return-value]


def manifest_type(data: dict[str, Any]) -> LoraDatasetType:
    return "ic_lora" if data.get("type") == "ic_lora" else "standard"


def _default_config_yaml(
    dataset_type: LoraDatasetType, *, config: TrainingConfig | None = None
) -> str:
    return build_training_yaml(
        config=config or TrainingConfig(),
        dataset_type=dataset_type,
        model_path="<PATH_TO_LTX2_MODEL.safetensors>",
        text_encoder_path="<PATH_TO_GEMMA_TEXT_ENCODER_DIR>",
        preprocessed_data_root="./.precomputed",
        output_dir="./outputs",
    )


def _readme(dataset: LoraDataset, *, is_ic: bool, row_count: int) -> str:
    trigger = dataset.trigger_word or "(none)"
    # IC-LoRA never trains audio; the trainer auto-detects the reference_video
    # column, so no reference flag is needed. Audio is on by default otherwise.
    audio_flag = "\\\n    --skip-audio " if is_ic else ""
    mode = "IC-LoRA (video_to_video)" if is_ic else "standard LoRA (text_to_video)"
    unit = "pair(s)" if is_ic else "clip(s)"
    clips_line = (
        "- `clips/` — the processed media (one fps, downscaled, rotation baked "
        "in, audio stripped). Files are named `NNNN_output_…` (the target) and "
        "`NNNN_reference_…` (the conditioning input) so each pair sorts "
        "together.\n"
        if is_ic
        else "- `clips/` — the media, numbered, with any edits already baked in.\n"
    )
    schema_line = (
        "- `dataset.json` — one record per pair: `{caption, video, "
        "reference_video}` (`video` = target/output, `reference_video` = "
        "conditioning/input).\n"
        if is_ic
        else "- `dataset.json` — caption + (relative) media paths the trainer reads.\n"
    )
    return (
        f"# {dataset.name}\n\n"
        f"LoRA training dataset exported from **LTX Desktop** — {row_count} "
        f"{unit}, type: {mode}, trigger word: `{trigger}`.\n\n"
        "## Contents\n\n"
        f"{schema_line}"
        f"{clips_line}"
        "- `train_config.yaml` — a ready-to-run trainer config (fill in the "
        "`<...>` model paths).\n"
        "- `ltxdesktop.json` — LTX Desktop manifest (used for re-import; the "
        "trainer ignores it).\n\n"
        "## Train it (LTX-2 trainer)\n\n"
        "Clone the trainer and, from its repo root, run:\n\n"
        "```bash\n"
        "# 1) Pre-compute latents + text embeddings\n"
        "uv run python scripts/process_dataset.py /path/to/dataset.json \\\n"
        '    --resolution-buckets "768x768x49" \\\n'
        "    --model-path <PATH_TO_LTX2_MODEL.safetensors> \\\n"
        f"    --text-encoder-path <PATH_TO_GEMMA_TEXT_ENCODER_DIR> {audio_flag}\n\n"
        "# 2) Train (edit train_config.yaml first to point at your model files)\n"
        "uv run python scripts/train.py /path/to/train_config.yaml\n"
        "```\n\n"
        "Resolution buckets are `WIDTHxHEIGHTxFRAMES` (e.g. `768x768x49`, "
        "`960x544x49`); pass several separated by `;`.\n\n"
        "## Re-import into LTX Desktop\n\n"
        'Use **LoRA Studio → Import dataset…** and pick this folder (or its '
        ".zip).\n"
    )

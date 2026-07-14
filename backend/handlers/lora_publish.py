"""Generate a publishable "model card" + asset bundle for a trained LoRA.

When people share a LoRA on Hugging Face / Civitai they ship a short paper-
style write-up: what it does, the trigger word, how it was trained, and a few
example generations. We already hold all of that (the run's `TrainingConfig`,
its `PreprocessedDataset`, and the source `LoraDataset`), so this module turns a
finished run into:

- a platform-tailored card (Hugging Face `README.md` with YAML front-matter,
  a Civitai description + structured metadata, and/or a portable Markdown card),
- the `.safetensors` weights,
- an `examples/` gallery of user-chosen showcase clips,
- a `publication.json` manifest.

Like `lora_export`, this is pure plumbing over fully-resolved paths so the
handler stays thin and the card text is trivially unit-testable. It never
gates publishing and never touches the network — the (later) auto-push services
own that.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from handlers.lora_export import BundleError, safe_dirname
from state.lora_training_state import (
    LoraDataset,
    LoraDatasetType,
    PreprocessedDataset,
    TrainingConfig,
    TrainingJob,
)

PublishPlatform = Literal["huggingface", "civitai", "portable"]
ALL_PLATFORMS: tuple[PublishPlatform, ...] = ("huggingface", "civitai", "portable")

# Default Hugging Face base-model repo + a human label. Editable per-publication
# since the exact repo someone fine-tuned from can differ.
DEFAULT_BASE_MODEL_REPO = "Lightricks/LTX-Video"
BASE_MODEL_LABEL = "LTX-2"

_VIDEO_EXTS: frozenset[str] = frozenset({".mp4", ".mov", ".webm", ".mkv", ".m4v"})


def _empty_str_list() -> list[str]:
    return []


# --- Inputs ---------------------------------------------------------------


@dataclass(frozen=True)
class PublicationExample:
    """A showcase clip the user picked from the LoRA's dataset.

    ``media_path`` is an absolute local file; ``caption`` is the prompt/desc
    shown beside it in the gallery.
    """

    media_path: str
    caption: str = ""


@dataclass
class PublicationMeta:
    """User-editable card fields, pre-seeded from the run (see ``suggest_meta``)."""

    title: str
    summary: str = ""
    description: str = ""
    author: str = ""
    license: str = "other"
    tags: list[str] = field(default_factory=_empty_str_list)
    base_model: str = DEFAULT_BASE_MODEL_REPO


# --- Card-side example (relative path) ------------------------------------


@dataclass(frozen=True)
class _CardExample:
    rel_path: str
    caption: str
    is_video: bool


def _is_video(path: str) -> bool:
    return Path(path).suffix.lower() in _VIDEO_EXTS


def plan_examples(
    examples: Sequence[PublicationExample],
) -> list[tuple[PublicationExample, _CardExample]]:
    """Assign each showcase clip a stable ``examples/NNNN_stem.ext`` name.

    Shared by preview (render only) and export (render + copy) so the gallery
    paths in the card always match the files actually written.
    """
    planned: list[tuple[PublicationExample, _CardExample]] = []
    for idx, ex in enumerate(examples, start=1):
        src = Path(ex.media_path)
        ext = src.suffix or ".mp4"
        rel = f"examples/{idx:04d}_{safe_dirname(src.stem)[:40]}{ext}"
        planned.append((ex, _CardExample(rel_path=rel, caption=ex.caption, is_video=_is_video(ex.media_path))))
    return planned


# --- Defaults / suggestions ----------------------------------------------


def _mode_label(dataset_type: LoraDatasetType) -> str:
    return "IC-LoRA (video-to-video)" if dataset_type == "ic_lora" else "standard LoRA (text-to-video)"


def _pipeline_tag() -> str:
    # Civitai/HF both bucket these as video generation; HF's closest tag is
    # text-to-video even for the edit (v2v) variant.
    return "text-to-video"


def suggest_tags(dataset: LoraDataset) -> list[str]:
    tags = ["lora", "ltx", "ltx-video", "ltx-desktop", _pipeline_tag()]
    if dataset.type == "ic_lora":
        tags.append("video-to-video")
    if dataset.trigger_word:
        tags.append(dataset.trigger_word)
    # Dedupe, keep order.
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        key = t.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(t.strip())
    return out


def suggest_meta(job: TrainingJob, dataset: LoraDataset) -> PublicationMeta:
    """Pre-fill the card fields so publishing is one edit-and-go, not a blank page."""
    title = (dataset.name or job.name or "Untitled LoRA").strip()
    mode = _mode_label(dataset.type)
    trig = f" Trigger word: `{dataset.trigger_word}`." if dataset.trigger_word else ""
    summary = f"An LTX-2 {mode} trained with LTX Desktop.{trig}".strip()
    return PublicationMeta(
        title=title,
        summary=summary,
        description="",
        author="",
        license="other",
        tags=suggest_tags(dataset),
        base_model=DEFAULT_BASE_MODEL_REPO,
    )


# --- Card building blocks -------------------------------------------------


def _fmt_duration(seconds: float) -> str:
    if seconds <= 0:
        return "—"
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}m {s}s"


def _dataset_stats(dataset: LoraDataset) -> tuple[int, float]:
    """(#training clips, total duration secs) for the kept, non-trashed set."""
    clips = [
        c
        for c in dataset.clips
        if not c.deleted_at and c.triage not in ("reject", "holdout")
    ]
    total = sum((c.duration_seconds or 0.0) for c in clips)
    return len(clips), total


def usage_snippet(dataset: LoraDataset, config: TrainingConfig) -> str:
    trigger = dataset.trigger_word
    res = f"{config.validation_video_width}×{config.validation_video_height}"
    lines = [
        f"- **Base model:** {BASE_MODEL_LABEL}",
        f"- **Resolution:** {res}, {config.validation_video_frames} frames @ {config.validation_frame_rate:g} fps",
        f"- **Inference steps:** {config.validation_inference_steps}",
        f"- **Guidance scale:** {config.validation_guidance_scale:g}",
    ]
    if trigger:
        lines.insert(0, f"- **Trigger word:** include `{trigger}` in your prompt")
    return "\n".join(lines)


def recipe_rows(config: TrainingConfig, preprocessed: PreprocessedDataset) -> list[tuple[str, str]]:
    """The training-details table, as ordered (label, value) pairs."""
    optimizer = config.optimizer_type or f"{config.preset} default"
    rows: list[tuple[str, str]] = [
        ("Base model", BASE_MODEL_LABEL),
        ("LoRA rank / alpha", f"{config.rank} / {config.alpha}"),
        ("Learning rate", f"{config.learning_rate:g}"),
        ("Training steps", str(config.steps)),
        ("Batch size", str(config.batch_size)),
        ("Grad accumulation", str(config.gradient_accumulation_steps)),
        ("Resolution buckets", preprocessed.resolution_buckets),
        ("Precision", config.mixed_precision_mode),
        ("Optimizer", optimizer),
        ("Scheduler", config.scheduler_type),
        ("Seed", str(config.seed)),
    ]
    if config.with_audio:
        rows.append(("Audio", "enabled"))
    return rows


def _front_matter(meta: PublicationMeta, dataset: LoraDataset) -> str:
    """Minimal, deterministic Hugging Face card metadata YAML."""
    tags = meta.tags or suggest_tags(dataset)
    lines = [
        "---",
        f"base_model: {meta.base_model}",
        "tags:",
        *[f"  - {t}" for t in tags],
        f"pipeline_tag: {_pipeline_tag()}",
        f"license: {meta.license or 'other'}",
        "---",
    ]
    return "\n".join(lines)


def _gallery_md(examples: Sequence[_CardExample]) -> str:
    if not examples:
        return "_No example media included._"
    blocks: list[str] = []
    for ex in examples:
        caption = ex.caption.strip() or "_(no caption)_"
        if ex.is_video:
            # HF + Civitai render raw HTML in cards; the link is a graceful
            # fallback for plain Markdown viewers.
            media = (
                f'<video controls width="480" src="{ex.rel_path}"></video>\n\n'
                f"[{ex.rel_path}]({ex.rel_path})"
            )
        else:
            media = f"![example]({ex.rel_path})"
        blocks.append(f"**{caption}**\n\n{media}")
    return "\n\n".join(blocks)


def build_model_card(
    *,
    platform: PublishPlatform,
    job: TrainingJob,
    preprocessed: PreprocessedDataset,
    dataset: LoraDataset,
    examples: Sequence[_CardExample],
    meta: PublicationMeta,
) -> str:
    """Render the Markdown card for one platform.

    Hugging Face gets YAML front-matter (its card metadata); Civitai and the
    portable card are plain Markdown (Civitai metadata travels in its JSON).
    """
    clip_count, duration = _dataset_stats(dataset)
    mode = _mode_label(dataset.type)
    trigger = dataset.trigger_word

    parts: list[str] = []
    if platform == "huggingface":
        parts.append(_front_matter(meta, dataset))

    parts.append(f"# {meta.title}")
    if meta.summary.strip():
        parts.append(meta.summary.strip())
    if meta.description.strip():
        parts.append(meta.description.strip())

    parts.append("## Trigger word")
    parts.append(
        f"Include `{trigger}` in your prompt to activate this LoRA."
        if trigger
        else "This LoRA has no dedicated trigger word — prompt it normally."
    )

    parts.append("## Recommended usage")
    parts.append(usage_snippet(dataset, job.config))

    parts.append("## Examples")
    parts.append(_gallery_md(examples))

    parts.append("## Training details")
    parts.append(f"Trained as a {mode} on **{clip_count}** clip(s) ({_fmt_duration(duration)} total).")
    table = ["| Setting | Value |", "| --- | --- |"]
    for label, value in recipe_rows(job.config, preprocessed):
        table.append(f"| {label} | {value} |")
    parts.append("\n".join(table))

    parts.append("## Intended use & limitations")
    parts.append(
        "Generated with a small fine-tuning dataset; outputs reflect that data and "
        "may not generalize. Review generations for your use case and respect the "
        "license of the base model and any source media."
    )

    parts.append("## Credits")
    author = f" by {meta.author}" if meta.author.strip() else ""
    parts.append(
        f"Trained{author} with [LTX Desktop](https://github.com/MountainPlatform300/LTX-Desktop) using the "
        f"LTX-2 trainer. Base model: `{meta.base_model}`."
    )

    return "\n\n".join(parts) + "\n"


def civitai_metadata(
    *,
    job: TrainingJob,
    preprocessed: PreprocessedDataset,
    dataset: LoraDataset,
    meta: PublicationMeta,
) -> dict[str, Any]:
    """Structured fields a Civitai upload (manual or API) expects."""
    return {
        "name": meta.title,
        "type": "LORA",
        "baseModel": BASE_MODEL_LABEL,
        "trainedWords": [dataset.trigger_word] if dataset.trigger_word else [],
        "tags": meta.tags or suggest_tags(dataset),
        "description": meta.summary,
        "trainingDetails": {label: value for label, value in recipe_rows(job.config, preprocessed)},
    }


# --- Bundle ---------------------------------------------------------------


def build_publication_bundle(
    *,
    platforms: Sequence[PublishPlatform],
    job: TrainingJob,
    preprocessed: PreprocessedDataset,
    dataset: LoraDataset,
    examples: Sequence[PublicationExample],
    meta: PublicationMeta,
    lora_path: str | None,
    staging_dir: Path,
) -> dict[str, Any]:
    """Write a complete publication into ``staging_dir``.

    Layout::

        <title>/
          README.md               # Hugging Face card (if selected)
          MODEL_CARD.md           # portable card (if selected)
          civitai_description.md   # Civitai body (if selected)
          civitai.json             # Civitai structured metadata (if selected)
          examples/                # chosen showcase clips
          <name>.safetensors       # the weights (if available locally)
          publication.json         # our manifest

    Returns a small manifest dict (counts + filenames). Raises ``BundleError``
    if a chosen example file is missing.
    """
    if not platforms:
        raise BundleError("Select at least one platform to publish to")

    staging_dir.mkdir(parents=True, exist_ok=True)
    planned = plan_examples(examples)

    # Copy showcase media.
    if planned:
        (staging_dir / "examples").mkdir(exist_ok=True)
        for ex, card in planned:
            src = Path(ex.media_path)
            if not src.is_file():
                raise BundleError(f"Example file is missing on disk: {ex.media_path}")
            shutil.copy2(src, staging_dir / card.rel_path)

    card_examples = [card for _, card in planned]

    written: list[str] = []
    if "huggingface" in platforms:
        card = build_model_card(
            platform="huggingface",
            job=job,
            preprocessed=preprocessed,
            dataset=dataset,
            examples=card_examples,
            meta=meta,
        )
        (staging_dir / "README.md").write_text(card, encoding="utf-8")
        written.append("README.md")
    if "portable" in platforms:
        card = build_model_card(
            platform="portable",
            job=job,
            preprocessed=preprocessed,
            dataset=dataset,
            examples=card_examples,
            meta=meta,
        )
        (staging_dir / "MODEL_CARD.md").write_text(card, encoding="utf-8")
        written.append("MODEL_CARD.md")
    if "civitai" in platforms:
        body = build_model_card(
            platform="civitai",
            job=job,
            preprocessed=preprocessed,
            dataset=dataset,
            examples=card_examples,
            meta=meta,
        )
        (staging_dir / "civitai_description.md").write_text(body, encoding="utf-8")
        meta_json = civitai_metadata(job=job, preprocessed=preprocessed, dataset=dataset, meta=meta)
        (staging_dir / "civitai.json").write_text(json.dumps(meta_json, indent=2), encoding="utf-8")
        written.extend(["civitai_description.md", "civitai.json"])

    # The weights themselves, when we have them downloaded locally.
    weights_name: str | None = None
    if lora_path:
        src = Path(lora_path)
        if src.is_file():
            weights_name = f"{safe_dirname(meta.title)}{src.suffix or '.safetensors'}"
            shutil.copy2(src, staging_dir / weights_name)

    manifest: dict[str, Any] = {
        "kind": "ltx-desktop-lora-publication",
        "schemaVersion": 1,
        "title": meta.title,
        "platforms": list(platforms),
        "trainingId": job.id,
        "datasetType": dataset.type,
        "triggerWord": dataset.trigger_word,
        "baseModel": meta.base_model,
        "license": meta.license,
        "tags": meta.tags or suggest_tags(dataset),
        "exampleCount": len(planned),
        "weightsFile": weights_name,
        "files": written,
        "publishedAt": datetime.now(timezone.utc).isoformat(),
    }
    (staging_dir / "publication.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

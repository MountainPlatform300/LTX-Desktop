"""Build a *training-ready* IC-LoRA bundle: paired, normalized, validated.

The LTX-2 "trainer-next" video-to-video workflow wants ONE record per pair::

    {"caption": "<describes the TARGET>",
     "video":           "<the TARGET / output clip>",
     "reference_video": "<the conditioning / driver / input clip>"}

The app stores a pair as a single target `LoraClip` whose
`reference_path`/`reference_paths` point at the input clip. This module turns
that into the schema above and — crucially — makes the *media itself* trainer-
ready with zero manual cleanup:

- both clips of a pair are re-encoded to one dataset-wide fps and one short-side
  resolution (rotation baked in, audio stripped) and trimmed to exactly the
  target bucket's frame count, so frame *i* of the target lines up with frame
  *i* of the reference (the trainer samples the first N frames at the clip fps);
- only complete, consistent pairs ship — input-only / unpaired rows are dropped;
- every surviving pair is validated (same fps / W×H / frame count, `8k+1` and
  ≥ the bucket, plus a non-empty, non-truncated, trigger-free caption).

Anything that can't be made consistent is dropped with a reason and reported,
never silently shipped. `render_media` maps a relative ``clips/<name>`` path to
the value written into the JSON (identity for a portable bundle; an absolute
remote path for upload staging).
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from handlers.lora_export import refs_of, safe_stem  # shared helpers
from services.clip_processor.clip_processor import (
    ClipProbeResult,
    ClipProcessor,
    ClipProcessorError,
)
from state.lora_training_state import LoraClip, LoraDataset

# Caption is considered "complete" if it ends with a sentence terminator. A
# truncated Gemini caption ("…foliage in") ends on a bare word, which this
# rejects. But many valid captions omit trailing punctuation ("A clean-shaven
# man smiles"), so a bare ending is only flagged when its final word is a
# function word that signals a cut-off — a dangling preposition, article, or
# conjunction is what a truncated caption actually looks like.
_SENTENCE_TERMINATORS: frozenset[str] = frozenset('.!?"\u201d)\u2026')
_TRUNCATION_TAIL_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "of", "in", "into", "on", "onto", "at",
        "by", "for", "from", "to", "with", "near", "as", "is", "are", "was",
        "were", "be", "being", "that", "this", "while", "about", "over",
    }
)

# Still-image media the trainer reads as a single frame. An image used as an
# IC-LoRA reference is looped into a full-length clip during normalization (and
# shipped as .mp4), so the pair satisfies the trainer's identical-length rule.
_IMAGE_SUFFIXES: frozenset[str] = frozenset({".png", ".jpg", ".jpeg"})
# How far a pair's target and reference aspect ratios may drift before we treat
# them as genuinely different shapes. Within this, both clips are scaled to one
# shared resolution (a sub-2% squash is imperceptible and beats dropping the
# pair); beyond it — e.g. a landscape target with a portrait reference — forcing
# a common size would badly distort one clip, so the pair is dropped instead.
_MAX_ASPECT_DRIFT: float = 0.15


@dataclass(frozen=True, slots=True)
class PrepOptions:
    """Knobs for normalizing + validating an IC-LoRA dataset.

    Defaults match the values that produced a clean beard-removal run:
    25 fps, short-side 576 (one above the 544 bucket short side to avoid VAE
    upscaling), and a 49-frame (`8k+1`) bucket. `forbidden_words` lets a caller
    reject concept words that must not leak into target captions (e.g. "beard"
    for a removal LoRA); the dataset's trigger word is always forbidden.
    """

    fps: float = 25.0
    short_side: int = 576
    bucket_frames: int = 49
    max_duration_seconds: float | None = None
    trigger_word: str | None = None
    forbidden_words: tuple[str, ...] = ()


def options_for_resolution_buckets(
    resolution_buckets: str, *, trigger_word: str | None
) -> PrepOptions:
    """Build a staging envelope that can satisfy every requested bucket."""
    parsed: list[tuple[int, int, int]] = []
    for raw in resolution_buckets.split(";"):
        width, height, frames = (int(part) for part in raw.strip().split("x"))
        parsed.append((width, height, frames))
    return PrepOptions(
        short_side=max(min(width, height) for width, height, _ in parsed),
        bucket_frames=max(frames for _, _, frames in parsed),
        trigger_word=trigger_word,
    )


@dataclass(frozen=True, slots=True)
class PrepDrop:
    """One pair (or unpaired clip) that was excluded, with the reason."""

    name: str
    reason: str


@dataclass(frozen=True, slots=True)
class EmittedPair:
    """A shipped pair's relative clip paths + caption (for the manifest)."""

    target_rel: str
    reference_rel: str
    caption: str


@dataclass(frozen=True, slots=True)
class PrepReport:
    exported: int
    dropped: list[PrepDrop] = field(default_factory=list[PrepDrop])
    pairs: list[EmittedPair] = field(default_factory=list[EmittedPair])

    def summary(self) -> str:
        lines = [f"{self.exported} pair(s) exported, {len(self.dropped)} dropped."]
        for d in self.dropped:
            lines.append(f"  - dropped {d.name}: {d.reason}")
        return "\n".join(lines)


def is_8k_plus_1(n: int) -> bool:
    """LTX-2 frame-count constraint: 1, 9, 17, 25, 33, 41, 49, …"""
    return n >= 1 and (n - 1) % 8 == 0


def caption_problem(
    caption: str, *, trigger_word: str | None, forbidden_words: tuple[str, ...]
) -> str | None:
    """Return why a target caption is unusable, or None if it's fine.

    Enforces (d): non-empty, complete (not truncated), and free of the trigger
    word + any configured concept words (matched as whole words).
    """
    c = caption.strip()
    if not c:
        return "empty caption"
    if c[-1] not in _SENTENCE_TERMINATORS:
        words = c.lower().split()
        tail = words[-1] if words else ""
        if tail in _TRUNCATION_TAIL_WORDS:
            return "caption looks truncated (does not end a sentence)"
    low = c.lower()
    forbidden: list[str] = list(forbidden_words)
    if trigger_word and trigger_word.strip():
        forbidden.append(trigger_word.strip())
    for word in forbidden:
        token = word.strip().lower()
        if token and re.search(rf"\b{re.escape(token)}\b", low):
            kind = "trigger word" if word.strip() == (trigger_word or "").strip() else "forbidden word"
            return f"caption contains {kind} {word.strip()!r}"
    return None


@dataclass(frozen=True, slots=True)
class _Pair:
    target: LoraClip
    reference: LoraClip


def collect_pairs(
    dataset: LoraDataset, clips: list[LoraClip]
) -> tuple[list[_Pair], list[PrepDrop]]:
    """Split the shipped clips into true (target, reference) pairs + drops.

    A *target* is any shipped clip with a resolvable reference whose file
    exists; its primary reference becomes `reference_video`. Clips that are
    only used *as* a reference are not separately dropped (they ride along as
    the pair's reference). Everything else — bearded input-only rows, missing
    files — is dropped with a reason (issue #2).
    """
    by_local: dict[str, LoraClip] = {c.local_path: c for c in dataset.clips}
    referenced: set[str] = set()
    for clip in clips:
        for ref in refs_of(clip):
            if ref in by_local:
                referenced.add(ref)

    pairs: list[_Pair] = []
    drops: list[PrepDrop] = []
    for clip in clips:
        resolvable = [by_local[r] for r in refs_of(clip) if r in by_local]
        if not resolvable:
            if clip.local_path in referenced:
                continue  # it's an input for some pair; ships as reference_video
            drops.append(PrepDrop(_label(clip), "unpaired (no reference) — input-only row"))
            continue
        reference = resolvable[0]
        if len(resolvable) > 1:
            # The released trainer conditions on a single reference. Surface the
            # extra inputs as drops rather than silently using only the first.
            extras = ", ".join(_label(r) for r in resolvable[1:])
            drops.append(
                PrepDrop(
                    _label(clip),
                    "example has multiple inputs; the trainer conditions on one "
                    f"reference, so only {_label(reference)} is used and these are "
                    f"dropped: {extras}",
                )
            )
        if not Path(clip.local_path).is_file():
            drops.append(PrepDrop(_label(clip), "target file missing on disk"))
            continue
        if not Path(reference.local_path).is_file():
            drops.append(PrepDrop(_label(clip), "reference file missing on disk"))
            continue
        pairs.append(_Pair(target=clip, reference=reference))
    return pairs, drops


def prepare_ic_lora_bundle(
    *,
    dataset: LoraDataset,
    clips: list[LoraClip],
    staging_dir: Path,
    processor: ClipProcessor,
    options: PrepOptions,
    render_media: Callable[[str], str],
) -> PrepReport:
    """Normalize + validate pairs into ``staging_dir`` and write ``dataset.json``.

    Returns a `PrepReport` (exported count, drops with reasons, emitted pairs).
    Heavy ffmpeg work — call without the state lock held.
    """
    clips_dir = staging_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = staging_dir / ".prep_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    pairs, drops = collect_pairs(dataset, clips)
    rows: list[dict[str, str]] = []
    emitted: list[EmittedPair] = []
    seq = 0
    try:
        for pair in pairs:
            label = _label(pair.target)
            ext_t = _ext(pair.target)
            ext_r = _ext(pair.reference)
            tmp_t = tmp_dir / f"t{ext_t}"
            tmp_r = tmp_dir / f"r{ext_r}"
            ref_is_image = _is_image(pair.reference)
            try:
                # Pick one resolution for the whole pair from the target's
                # (undistorted) geometry, then scale BOTH clips to it so they
                # match exactly. Probing the sources first lets us reject pairs
                # whose shapes are too different to align without bad distortion.
                src_t = processor.probe(video_path=pair.target.local_path)
                if ref_is_image:
                    # A still image has no orientation/length to reconcile and is
                    # pure conditioning, so scale it to the target's geometry
                    # (upscaling if needed) rather than letting a small image
                    # shrink the target or dropping the pair on aspect drift.
                    dims = single_dims(src_t, options.short_side)
                    if dims is None:
                        drops.append(PrepDrop(label, "could not read target resolution"))
                        continue
                else:
                    src_r = processor.probe(video_path=pair.reference.local_path)
                    dims = pair_dims(src_t, src_r, options.short_side)
                    if dims is None:
                        drops.append(
                            PrepDrop(
                                label,
                                "target and reference have different orientation/aspect "
                                f"({src_t.width}x{src_t.height} vs {src_r.width}x{src_r.height})",
                            )
                        )
                        continue
                _normalize(processor, pair.target.local_path, tmp_t, options, dims=dims)
                _normalize(processor, pair.reference.local_path, tmp_r, options, dims=dims)
                probe_t = processor.probe(video_path=str(tmp_t))
                probe_r = processor.probe(video_path=str(tmp_r))
            except ClipProcessorError as exc:
                drops.append(PrepDrop(label, f"ffmpeg failed: {exc.detail}"))
                continue

            problem = _pair_problem(probe_t, probe_r, options) or caption_problem(
                pair.target.caption,
                trigger_word=options.trigger_word,
                forbidden_words=options.forbidden_words,
            )
            if problem is not None:
                drops.append(PrepDrop(label, problem))
                continue

            seq += 1
            target_rel = f"clips/{seq:04d}_output_{safe_stem(Path(pair.target.local_path).stem)}{ext_t}"
            reference_rel = f"clips/{seq:04d}_reference_{safe_stem(Path(pair.reference.local_path).stem)}{ext_r}"
            shutil.move(str(tmp_t), str(staging_dir / target_rel))
            shutil.move(str(tmp_r), str(staging_dir / reference_rel))
            rows.append(
                {
                    "caption": pair.target.caption.strip(),
                    "video": render_media(target_rel),
                    "reference_video": render_media(reference_rel),
                }
            )
            emitted.append(EmittedPair(target_rel, reference_rel, pair.target.caption.strip()))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    (staging_dir / "dataset.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return PrepReport(exported=len(emitted), dropped=drops, pairs=emitted)


@dataclass(frozen=True, slots=True)
class HoldoutStagingReport:
    """Result of staging held-out clips' reference videos for validation.

    `staged` holds the clip ids whose reference video was normalized and
    written to `staging_dir/holdout/{id}.mp4`; `dropped` holds the clips that
    could not be staged (with reasons). Text-to-video holdout clips have no
    reference and are neither staged nor dropped — their caption alone is the
    validation prompt. `auto_picked` is the id of a training clip staged as a
    validation fallback when no clip was marked `holdout` (IC-LoRA only), so
    a run still gets a progress feed without the user curating holdouts.
    """

    staged: list[str] = field(default_factory=list[str])
    dropped: list[PrepDrop] = field(default_factory=list[PrepDrop])
    auto_picked: str | None = None


HOLDOUT_REFERENCE_DIR = "holdout"


def holdout_reference_filename(clip_id: str) -> str:
    """Filename of a staged holdout reference (``<id>.mp4``)."""
    return f"{clip_id}.mp4"


def holdout_reference_relpath(clip_id: str) -> str:
    """Relative path of a holdout clip's staged reference video (``holdout/<id>.mp4``).

    Single source of truth for the holdout-reference layout. The runner bakes
    this into a validation sample's reference condition as a POSIX remote path
    (``{remote_dataset_dir}/{holdout_reference_relpath(id)}``); the staging
    helper composes the same layout locally from `HOLDOUT_REFERENCE_DIR` +
    `holdout_reference_filename` (using separate `/` joins so a clip id that
    happens to be an absolute path collapses the same way the bundle paths do).
    Keeping the layout in one module ensures the staged file and the YAML
    reference path agree.
    """
    return f"{HOLDOUT_REFERENCE_DIR}/{holdout_reference_filename(clip_id)}"


def stage_holdout_references(
    *,
    dataset: LoraDataset,
    staging_dir: Path,
    processor: ClipProcessor,
    options: PrepOptions,
    auto_pick_when_empty: bool = False,
) -> HoldoutStagingReport:
    """Stage held-out clips' reference videos for IC-LoRA validation.

    Clips marked ``triage="holdout"`` are excluded from training (see the
    runner's ``_build_staging`` filter) but an IC-LoRA validation sample needs
    a reference (input) video to condition on. We normalize each holdout
    clip's primary reference to ``staging_dir/holdout/{clip.id}.mp4`` so it
    ships with the dataset upload and lands at
    ``{remote_dataset_dir}/holdout/{clip.id}.mp4`` — the path the runner bakes
    into the validation sample's reference condition at training start.

    Best-effort: a clip whose reference can't be probed/normalized is dropped
    from validation (reported), never failing the upload. Text-to-video
    holdout clips (no reference) are a no-op here — their caption alone is the
    validation prompt.

    When ``auto_pick_when_empty`` is set and no holdout clip was staged, the
    first training clip (not rejected/holdout, with a caption + reference) is
    staged the same way so IC-LoRA runs without a curated holdout still get a
    validation feed. That clip remains in the training set, so the feed
    monitors progress rather than generalization.
    """
    by_local: dict[str, LoraClip] = {c.local_path: c for c in dataset.clips}
    tmp_dir = staging_dir / ".holdout_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    staged: list[str] = []
    dropped: list[PrepDrop] = []
    try:
        for clip in dataset.clips:
            if clip.triage != "holdout" or clip.deleted_at:
                continue
            refs = [by_local[r] for r in refs_of(clip) if r in by_local]
            if not refs:
                continue  # t2v holdout: prompt-only, no reference to stage
            reference = refs[0]
            if not Path(reference.local_path).is_file():
                dropped.append(
                    PrepDrop(_label(clip), "holdout reference file missing on disk")
                )
                continue
            tmp = tmp_dir / f"{clip.id}.mp4"
            try:
                _normalize(processor, reference.local_path, tmp, options)
            except ClipProcessorError as exc:
                dropped.append(
                    PrepDrop(_label(clip), f"holdout reference ffmpeg failed: {exc.detail}")
                )
                continue
            dest = staging_dir / HOLDOUT_REFERENCE_DIR / holdout_reference_filename(clip.id)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tmp), str(dest))
            staged.append(clip.id)

        auto_picked: str | None = None
        if auto_pick_when_empty and not staged:
            pick = _auto_pick_validation_clip(dataset, by_local)
            if pick is not None:
                clip, reference = pick
                tmp = tmp_dir / f"{clip.id}.mp4"
                try:
                    _normalize(processor, reference.local_path, tmp, options)
                    dest = (
                        staging_dir
                        / HOLDOUT_REFERENCE_DIR
                        / holdout_reference_filename(clip.id)
                    )
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(tmp), str(dest))
                    auto_picked = clip.id
                    staged.append(clip.id)
                except ClipProcessorError as exc:
                    dropped.append(
                        PrepDrop(
                            _label(clip),
                            f"auto validation reference ffmpeg failed: {exc.detail}",
                        )
                    )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return HoldoutStagingReport(staged=staged, dropped=dropped, auto_picked=auto_picked)


def _auto_pick_validation_clip(
    dataset: LoraDataset, by_local: dict[str, LoraClip]
) -> tuple[LoraClip, LoraClip] | None:
    """First training clip with a caption + reference, for IC-LoRA fallback."""
    for clip in dataset.clips:
        if clip.deleted_at or clip.triage in ("reject", "holdout"):
            continue
        if not clip.caption.strip():
            continue
        refs = [by_local[r] for r in refs_of(clip) if r in by_local]
        if not refs:
            continue
        reference = refs[0]
        if not Path(reference.local_path).is_file():
            continue
        return clip, reference
    return None


def _normalize(
    processor: ClipProcessor,
    source: str,
    out_path: Path,
    options: PrepOptions,
    *,
    dims: tuple[int, int] | None = None,
) -> None:
    processor.normalize_for_training(
        source_path=source,
        out_path=str(out_path),
        fps=options.fps,
        short_side=options.short_side,
        frames=options.bucket_frames,
        max_duration_seconds=options.max_duration_seconds,
        exact_width=dims[0] if dims else None,
        exact_height=dims[1] if dims else None,
    )


def _even(value: float) -> int:
    return max(2, int(round(value / 2.0)) * 2)


def pair_dims(
    target: ClipProbeResult, reference: ClipProbeResult, short_side: int
) -> tuple[int, int] | None:
    """One shared output resolution for a pair, or None if they can't align.

    Uses the TARGET's aspect (the clip the model learns to produce, so it must
    not be distorted) at a short side that never upscales *either* clip. The
    reference is later scaled to these exact dims — fine while the two shapes
    are close, but rejected (None) for different orientations or large drift.
    """
    tw, th, rw, rh = target.width, target.height, reference.width, reference.height
    if min(tw, th, rw, rh) <= 0:
        return None
    t_ar = tw / th
    r_ar = rw / rh
    # Different orientation (one portrait, one landscape) — don't force-align.
    if (t_ar >= 1.0) != (r_ar >= 1.0):
        return None
    if abs(t_ar - r_ar) / min(t_ar, r_ar) > _MAX_ASPECT_DRIFT:
        return None
    # Never upscale either clip: cap the short side to the smaller of the two.
    short = min(short_side, min(tw, th), min(rw, rh))
    t_short = min(tw, th)
    return _even(tw * short / t_short), _even(th * short / t_short)


def single_dims(target: ClipProbeResult, short_side: int) -> tuple[int, int] | None:
    """Output resolution derived from the target alone (never upscaling it).

    Used when the reference is a still image: the image is then scaled to these
    exact dims (upscaling allowed — it's only conditioning), so the pair matches
    the trainer's identical-shape rule without a small image dragging the
    target's resolution down. Returns even dims, or None if unreadable.
    """
    w, h = target.width, target.height
    if min(w, h) <= 0:
        return None
    short = min(short_side, min(w, h))
    s = min(w, h)
    return _even(w * short / s), _even(h * short / s)


def _pair_problem(
    target: ClipProbeResult, reference: ClipProbeResult, options: PrepOptions
) -> str | None:
    """Per-pair consistency (issue #9) on the *normalized* outputs."""
    if target.frame_count <= 0 or reference.frame_count <= 0:
        return "could not read output frame count"
    if abs(target.fps - reference.fps) > 0.05:
        return f"fps mismatch within pair ({target.fps:g} vs {reference.fps:g})"
    if abs(target.fps - options.fps) > 0.05:
        return f"fps {target.fps:g} != target {options.fps:g}"
    if (target.width, target.height) != (reference.width, reference.height):
        return (
            f"resolution mismatch ({target.width}x{target.height} vs "
            f"{reference.width}x{reference.height}) — different aspect ratios"
        )
    if target.frame_count != reference.frame_count:
        return f"frame-count mismatch ({target.frame_count} vs {reference.frame_count})"
    if not is_8k_plus_1(target.frame_count):
        return f"frame count {target.frame_count} is not 8k+1"
    if target.frame_count < options.bucket_frames:
        return f"only {target.frame_count} frames; need >= {options.bucket_frames} for the bucket"
    return None


def _label(clip: LoraClip) -> str:
    return Path(clip.local_path).name


def _is_image(clip: LoraClip) -> bool:
    return Path(clip.local_path).suffix.lower() in _IMAGE_SUFFIXES


def _ext(clip: LoraClip) -> str:
    """Container extension for a clip's normalized output.

    Image references are looped into a short video, so they ship as ``.mp4``:
    the trainer reads a ``.png``/``.jpg``/``.jpeg`` path as a single frame,
    which would violate the identical-length IC-LoRA constraint.
    """
    if _is_image(clip):
        return ".mp4"
    return Path(clip.local_path).suffix or ".mp4"

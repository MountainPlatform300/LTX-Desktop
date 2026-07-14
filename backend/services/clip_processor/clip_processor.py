"""Local (desktop-side) clip-processing service protocol.

Dataset preparation for a LoRA happens entirely on the desktop before
anything is uploaded to the remote GPU. This service is the home for
those GPU-free, ffmpeg/cv2-backed operations. Phase 1 exposes only
`probe` (read a clip's duration / resolution / fps / audio); later
phases add trim, crop, and scene-splitting behind the same Protocol so
the handler/route layer never learns whether the work is local ffmpeg
or anything else.

Following the repo's service convention: a `Protocol` here, a real
ffmpeg implementation alongside, and a `Fake*` in `tests/fakes/` that
the test bundle swaps in (no GPU / no ffmpeg in CI).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ClipProcessorError(Exception):
    """Raised when a clip can't be read/processed.

    `status_code` lets the route layer map the failure to an HTTP
    status without leaking raw ffmpeg stderr — the message is curated.
    """

    def __init__(self, detail: str, *, status_code: int = 422) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class ClipProbeResult:
    """Objective, measured facts about a single clip.

    Pure measurement — no judgement about whether the clip is "good"
    for training. Quality heuristics live in the UI so they can evolve
    without a backend round-trip.
    """

    duration_seconds: float
    width: int
    height: int
    fps: float
    frame_count: int
    has_audio: bool
    video_codec: str | None


@dataclass(frozen=True, slots=True)
class TrimSpec:
    """Keep only [start, end) of the source timeline (seconds)."""

    start_seconds: float
    end_seconds: float


@dataclass(frozen=True, slots=True)
class CropSpec:
    """Pixel crop rectangle, applied after trim."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class ScaleSpec:
    """Target pixel resolution, applied after crop (e.g. bucket-snap)."""

    width: int
    height: int


@dataclass(frozen=True, slots=True)
class EditPlan:
    """The full non-destructive edit stack applied in one re-encode.

    Order of application: trim (input seek) → crop → scale → fps →
    reverse → speed. Audio mirrors the video (reverse/tempo) unless
    `mute` drops it. A plan where `is_empty` is True is a no-op and the
    caller should not write a derived file at all.
    """

    trim: TrimSpec | None = None
    crop: CropSpec | None = None
    scale: ScaleSpec | None = None
    fps: float | None = None
    speed: float | None = None
    mute: bool = False
    reverse: bool = False

    @property
    def is_empty(self) -> bool:
        return (
            self.trim is None
            and self.crop is None
            and self.scale is None
            and self.fps is None
            and (self.speed is None or self.speed == 1.0)
            and not self.mute
            and not self.reverse
        )


@dataclass(frozen=True, slots=True)
class SceneSpan:
    """A detected scene segment in the source timeline (seconds)."""

    start_seconds: float
    end_seconds: float


class ClipProcessor(Protocol):
    def probe(self, *, video_path: str) -> ClipProbeResult:
        ...

    def render(self, *, source_path: str, plan: EditPlan, out_path: str) -> None:
        """Render `source_path` to `out_path` applying `plan`.

        Re-encodes (the edits change the stream), mirroring video timing
        changes onto audio unless muted. The plan must not be empty — a
        no-op edit should be handled by the caller (don't write a derived
        file at all).
        """
        ...

    def normalize_for_training(
        self,
        *,
        source_path: str,
        out_path: str,
        fps: float,
        short_side: int,
        frames: int,
        max_duration_seconds: float | None = None,
        exact_width: int | None = None,
        exact_height: int | None = None,
    ) -> None:
        """Re-encode `source_path` into a training-ready clip at `out_path`.

        Produces a deterministic clip so paired IC-LoRA videos line up frame
        for frame (the trainer samples the first N frames at the clip's fps):

        - **fps** is forced to `fps` for the whole dataset (temporal align).
        - **short_side**: the shorter dimension is scaled to `short_side`,
          aspect preserved, dims rounded to even — and *never upscaled* (a
          clip already smaller is left at its size). Never crops.
        - **exact_width / exact_height**: when *both* are given they override
          `short_side` and the clip is scaled to exactly that resolution. This
          lets a caller force the two clips of an IC-LoRA pair to an identical
          W×H even when their source aspect ratios differ by a hair (a 1264×720
          target vs a 1920×1080 reference), so the pair stays consistent instead
          of being dropped. The tiny aspect squash this implies is intentional.
        - **frames**: output is capped to exactly `frames` frames (`-frames:v`),
          which the caller chooses as the target bucket size (`8k+1`). If the
          source has fewer frames after fps conversion the output will be
          shorter — the caller re-probes and drops such pairs.
        - rotation metadata is baked into the pixels and then stripped, so the
          file's W×H reflect what's displayed.
        - audio is dropped (`-an`); v2v LoRA ignores it.
        - `max_duration_seconds`, when set, trims the input first (guards huge
          clips from a full decode).

        Raises `ClipProcessorError` on failure.
        """
        ...

    def detect_scenes(self, *, video_path: str, threshold: float) -> list[SceneSpan]:
        """Detect scene-cut segments via ffmpeg's scene score filter.

        Returns one span per detected segment (always covering the whole
        clip); an empty list means probing failed / the clip is unreadable.
        """
        ...

    def extract_frame(self, *, video_path: str, time_seconds: float) -> bytes:
        """Extract a single frame at `time_seconds` as PNG bytes.

        Used by the AI dataset-prep tools (e.g. edit the first frame with
        an image model). Raises `ClipProcessorError` if the frame can't
        be read.
        """
        ...

    def generate_sprite(
        self, *, video_path: str, out_path: str, tile_count: int, tile_width: int
    ) -> int:
        """Render a horizontal filmstrip montage of `tile_count` frames.

        Powers the curation gallery's hover-scrub. Writes a single image
        (`tile_count` frames laid left-to-right, each `tile_width` px wide)
        to `out_path` and returns the number of tiles actually produced.
        Raises `ClipProcessorError`.
        """
        ...

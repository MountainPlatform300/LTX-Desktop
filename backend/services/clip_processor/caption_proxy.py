"""Shared caption-proxy helper for oversized video clips.

The Gemini inline (base64) captioning endpoint caps a request at ~20MB; base64
inflates raw bytes by ~33%, so clips over ~14MB are rejected by the captioner.
For auto-prompt (which feeds an arbitrary imported IC-LoRA reference video) and
for dataset captioning, we transcode a small, caption-only proxy — lower
resolution + fps cap + muted (when audio is irrelevant) — that fits the budget,
then caption the proxy instead. Captioning is also faster/cheaper on it.

The helper is stateless and lock-free: ffmpeg reads the file directly. The
caller owns the returned temp dir and must clean it up.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from services.interfaces import ClipProcessor, ClipProcessorError, EditPlan, ScaleSpec

logger = logging.getLogger(__name__)

# Inline request budget for the Generative Language API is ~20MB; base64
# inflates by ~33%, so keep the raw clip under ~12MB to stay safely inside it
# with headroom for the prompt.
CAPTION_PROXY_BUDGET_BYTES = 12 * 1024 * 1024
# Long-side resolutions tried in order; the first proxy that fits the budget
# wins. 768px keeps plenty of detail for captioning; lower tiers rescue very
# long clips. Captioning is also faster/cheaper on the smaller proxy.
CAPTION_PROXY_LONG_SIDES: tuple[int, ...] = (768, 512, 384)
CAPTION_PROXY_FPS_CAP = 16.0
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"})


def _even(value: float) -> int:
    """Round to the nearest positive even int (yuv420p needs even dims)."""
    return max(2, int(round(value / 2)) * 2)


def build_caption_proxy_if_oversized(
    clip_processor: ClipProcessor,
    video_path: str,
    *,
    with_audio: bool,
) -> tuple[Path, str] | None:
    """Return ``(temp_dir, proxy_path)`` for an oversized video, else None.

    Caller owns ``temp_dir`` cleanup. Returns None (caption the original) for
    images, small clips, or if probing/transcoding fails — the captioner's own
    size guard then surfaces a friendly error.
    """
    src = Path(video_path)
    if src.suffix.lower() in _IMAGE_SUFFIXES:
        return None
    try:
        if not src.is_file() or src.stat().st_size <= CAPTION_PROXY_BUDGET_BYTES:
            return None
    except OSError:
        return None

    try:
        probe = clip_processor.probe(video_path=video_path)
    except ClipProcessorError:
        return None  # let the captioner report the size problem
    long_side = max(probe.width, probe.height)
    if long_side <= 0:
        return None
    fps = CAPTION_PROXY_FPS_CAP if probe.fps and probe.fps > CAPTION_PROXY_FPS_CAP else None

    tmp_dir = Path(tempfile.mkdtemp(prefix="ltx-caption-"))
    out = tmp_dir / "proxy.mp4"
    try:
        for target in CAPTION_PROXY_LONG_SIDES:
            factor = min(1.0, target / long_side)
            width = _even(probe.width * factor)
            height = _even(probe.height * factor)
            clip_processor.render(
                source_path=video_path,
                plan=EditPlan(
                    scale=ScaleSpec(width=width, height=height),
                    fps=fps,
                    mute=not with_audio,
                ),
                out_path=str(out),
            )
            if out.stat().st_size <= CAPTION_PROXY_BUDGET_BYTES:
                logger.info(
                    "caption proxy path=%s %dx%d bytes=%d",
                    video_path,
                    width,
                    height,
                    out.stat().st_size,
                )
                return tmp_dir, str(out)
        # Smallest tier still over budget — hand it over anyway; better a shot
        # at captioning than a hard reject.
        return tmp_dir, str(out)
    except (ClipProcessorError, OSError):
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

"""Video captioning service (desktop-side, Gemini-backed)."""

from __future__ import annotations

from services.video_captioner.video_captioner import (
    VideoCaptioner,
    VideoCaptionerError,
)

__all__ = ["VideoCaptioner", "VideoCaptionerError"]

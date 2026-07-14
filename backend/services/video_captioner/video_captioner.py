"""Desktop-side video captioning service.

Captions a single local video clip for LoRA dataset prep. This runs on the
control plane (the desktop's local backend), not the remote GPU host: the
trainer's own `caption_videos.py` is the remote path, but captioning before
upload lets users review and edit captions in the GUI first — which the
LTX-2 trainer docs explicitly recommend.

The Protocol keeps the network side effect (a vision-model call) behind a
swappable boundary so tests use a deterministic fake instead of mocking HTTP.
"""

from __future__ import annotations

from typing import Protocol


class VideoCaptionerError(Exception):
    """A captioning attempt failed in a user-presentable way.

    `status_code` is the HTTP status the route should surface (e.g. 400 for a
    missing API key, 413 for an oversized clip, 502 for an upstream error).
    """

    def __init__(self, detail: str, *, status_code: int = 502) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


class VideoCaptioner(Protocol):
    def caption(
        self,
        *,
        video_path: str,
        api_key: str,
        with_audio: bool,
        instructions: str | None = None,
    ) -> str:
        """Return a natural-language caption describing the clip.

        `with_audio` asks the model to also describe speech and sound (for
        audio-video training). `instructions` overrides the default system
        prompt (used by the per-LoRA auto-prompt assistant). Raises
        `VideoCaptionerError` on any failure.
        """
        ...

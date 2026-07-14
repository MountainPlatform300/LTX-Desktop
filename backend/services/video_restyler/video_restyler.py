"""Video synthesis service for LoRA dataset prep (Fal LTX endpoints).

Two operations, both BYOK via the app's `fal_api_key`:

- `restyle`: video-to-video — re-render an existing clip under a text
  prompt (e.g. "make it claymation") to grow a stylised training set.
- `animate`: image-to-video — turn an (often Nano-Banana-edited) frame
  into a clip, realising the "edit the first frame, then generate the
  video" dataset-manipulation workflow.

API-only (no local backend); Protocol + real (`FalVideoRestyler`) + fake
follow the standard service convention.
"""

from __future__ import annotations

from typing import Protocol


class VideoRestylerError(Exception):
    def __init__(self, detail: str, *, status_code: int = 502) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


class VideoRestyler(Protocol):
    def restyle(
        self, *, video_bytes: bytes, prompt: str, api_key: str
    ) -> bytes:
        """Video-to-video. Returns the restyled clip bytes (mp4)."""
        ...

    def animate(
        self, *, image_bytes: bytes, prompt: str, api_key: str
    ) -> bytes:
        """Image-to-video. Returns the generated clip bytes (mp4)."""
        ...

    def motion_edit(
        self,
        *,
        video_bytes: bytes,
        image_bytes: bytes,
        prompt: str,
        video_strength: float,
        api_key: str,
    ) -> bytes:
        """Motion-locked edit (LTX-2 v2v + first-frame anchor).

        The original clip drives the motion (`video_bytes`); an edited
        still (`image_bytes`) anchors the first frame. `video_strength`
        in [0,1] trades motion/structure fidelity (high) for freedom to
        adopt the edited content (low). Realises the "edit a frame, then
        regenerate the clip with the same motion" paired-dataset flow.
        Returns the generated clip bytes (mp4).
        """
        ...

    def motion_transfer(
        self,
        *,
        image_bytes: bytes,
        video_bytes: bytes,
        prompt: str,
        character_orientation: str,
        api_key: str,
    ) -> bytes:
        """Kling motion-control: transfer the reference video's motion onto
        a character image. `character_orientation` is "video" (full body +
        camera follow the reference, <=30s) or "image" (preserve the image's
        framing, gesture transfer only, <=10s). Returns the clip bytes (mp4).
        """
        ...

    def kling_v2v_edit(
        self,
        *,
        video_bytes: bytes,
        image_bytes: bytes | None,
        prompt: str,
        keep_audio: bool,
        api_key: str,
    ) -> bytes:
        """Kling O3 video-to-video edit: re-render `video_bytes` under
        `prompt` (text-driven). When `image_bytes` is given it's passed as an
        appearance/style reference (referenced in the prompt as @Image1),
        e.g. a Nano-Banana-edited first frame. `keep_audio` preserves the
        source clip's audio. Returns the generated clip bytes (mp4).
        """
        ...

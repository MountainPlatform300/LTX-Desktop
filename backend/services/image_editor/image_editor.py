"""Image-editing service protocol (Nano Banana / Gemini image edit).

Used by LoRA dataset prep to edit a frame extracted from a clip — e.g.
remove a logo, add an attribute, swap a background — so the user can
build a manipulated training set. There is no local implementation; the
only backend is Fal (BYOK via the app's `fal_api_key`), so this is an
API-only service. Same Protocol + real + fake convention as the rest.
"""

from __future__ import annotations

from typing import Literal, Protocol

# Selectable Nano Banana tiers, mapped to Fal endpoints by the impl.
NanoBananaModel = Literal["nano-banana", "nano-banana-2", "nano-banana-pro"]


class ImageEditorError(Exception):
    def __init__(self, detail: str, *, status_code: int = 502) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


class ImageEditor(Protocol):
    def edit(
        self,
        *,
        image_bytes: bytes,
        prompt: str,
        model: NanoBananaModel,
        api_key: str,
    ) -> bytes:
        """Return the edited image bytes (PNG). Raises `ImageEditorError`."""
        ...

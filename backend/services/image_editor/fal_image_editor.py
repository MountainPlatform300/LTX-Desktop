"""Fal-backed `ImageEditor` (Nano Banana / Gemini image edit).

Sends the source frame as a base64 data URI in `image_urls` (Fal accepts
data URIs for image inputs), then downloads the edited result. Mirrors
the request/response shape of the existing `ZitAPIClientImpl` FAL flow.
"""

from __future__ import annotations

import base64
from typing import Any, cast

from services.http_client.http_client import HTTPClient
from services.image_editor.image_editor import ImageEditorError, NanoBananaModel
from services.services_utils import JSONValue

FAL_API_BASE_URL = "https://fal.run"

# Nano Banana tier -> Fal edit endpoint.
_MODEL_ENDPOINTS: dict[NanoBananaModel, str] = {
    "nano-banana": "/fal-ai/gemini-25-flash-image/edit",
    "nano-banana-2": "/fal-ai/nano-banana-2/edit",
    "nano-banana-pro": "/fal-ai/nano-banana-pro/edit",
}


class FalImageEditor:
    def __init__(self, http: HTTPClient, *, fal_api_base_url: str = FAL_API_BASE_URL) -> None:
        self._http = http
        self._base_url = fal_api_base_url.rstrip("/")

    def edit(
        self,
        *,
        image_bytes: bytes,
        prompt: str,
        model: NanoBananaModel,
        api_key: str,
    ) -> bytes:
        if not api_key:
            raise ImageEditorError(
                "Add a Fal API key in Settings to edit frames.", status_code=400
            )
        endpoint = _MODEL_ENDPOINTS.get(model, _MODEL_ENDPOINTS["nano-banana-2"])
        data_uri = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
        payload: dict[str, JSONValue] = {"prompt": prompt, "image_urls": [data_uri]}

        response = self._http.post(
            f"{self._base_url}{endpoint}",
            headers={"Authorization": f"Key {api_key}", "Content-Type": "application/json"},
            json_payload=payload,
            timeout=180,
        )
        if response.status_code != 200:
            detail = response.text[:500] if response.text else "Unknown error"
            raise ImageEditorError(f"Fal image edit failed ({response.status_code}): {detail}")

        image_url = self._extract_image_url(self._json_object(response.json()))
        download = self._http.get(image_url, timeout=120)
        if download.status_code != 200 or not download.content:
            raise ImageEditorError(
                f"Fal edited-image download failed ({download.status_code})"
            )
        return download.content

    @staticmethod
    def _extract_image_url(payload: dict[str, Any]) -> str:
        images = payload.get("images")
        if isinstance(images, list) and images:
            first = cast(list[object], images)[0]
            if isinstance(first, dict):
                url = cast(dict[str, Any], first).get("url")
                if isinstance(url, str) and url:
                    return url
            if isinstance(first, str) and first:
                return first
        for key in ("image_url", "url"):
            url = payload.get(key)
            if isinstance(url, str) and url:
                return url
        raise ImageEditorError("Fal response missing edited image url")

    @staticmethod
    def _json_object(payload: object) -> dict[str, Any]:
        if isinstance(payload, dict):
            return cast(dict[str, Any], payload)
        raise ImageEditorError("Unexpected Fal response format")

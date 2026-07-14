"""Gemini-backed video captioner.

Sends the clip inline (base64) to the Gemini `generateContent` REST endpoint
via the shared `HTTPClient`, mirroring the existing gap-prompt integration
(`handlers/suggest_gap_prompt_handler.py`) rather than pulling in a new SDK.

Inline upload caps the request at ~20MB, so we guard on raw file size and
raise a friendly `VideoCaptionerError` advising the user to trim/split — the
scene-split tool produces short clips that comfortably fit.
"""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from services.interfaces import HTTPClient, HttpResponseLike, HttpTimeoutError, JSONValue
from services.video_captioner.video_captioner import VideoCaptionerError

logger = logging.getLogger(__name__)

# Inline request budget for the Generative Language API is ~20MB; base64
# inflates by ~33%, so keep the raw clip under ~14MB to stay safely inside it.
_MAX_INLINE_BYTES = 14 * 1024 * 1024

_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

# Gemini intermittently returns transient server errors (notably 500 INTERNAL)
# and 429 rate limits for otherwise-valid requests; retry a few times with
# linear backoff before surfacing the failure.
_RETRYABLE_STATUS = frozenset({429, 500, 503})
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 1.0

_MIME_BY_SUFFIX = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".avi": "video/x-msvideo",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
    ".m4v": "video/mp4",
}

# Still-image inputs caption through the same multimodal endpoint; only the
# MIME type and the instructions differ from the video path.
_IMAGE_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


class _GeminiPart(BaseModel):
    text: str


class _GeminiContent(BaseModel):
    parts: list[_GeminiPart] = Field(min_length=1)


class _GeminiCandidate(BaseModel):
    content: _GeminiContent
    # "STOP" = natural end; "MAX_TOKENS" = hit the output budget (truncated).
    finishReason: str | None = None


class _GeminiResponsePayload(BaseModel):
    candidates: list[_GeminiCandidate] = Field(min_length=1)


def _instructions(*, with_audio: bool, is_image: bool) -> str:
    if is_image:
        return (
            "You are captioning a still image to train an image/video generation model. "
            "Write a single concise caption (1-3 sentences) describing what is visible: "
            "the main subject(s), the setting, lighting, colors, and composition. "
            "Output only the caption text - no labels, prefixes, or quotation marks."
        )
    parts = [
        "You are captioning a short video clip to train a video generation model.",
        "Write a single concise caption (1-3 sentences) describing what is visible:",
        "the main subject(s), their actions, the setting, lighting, colors, and any",
        "notable camera movement.",
    ]
    if with_audio:
        parts.append(
            "Also briefly describe the audio: speech (summarize, don't transcribe verbatim),"
            " music, and notable sound effects."
        )
    parts.append("Output only the caption text — no labels, prefixes, or quotation marks.")
    return " ".join(parts)


class GeminiVideoCaptioner:
    """Real `VideoCaptioner` implementation backed by Gemini 2.0 Flash."""

    def __init__(self, http: HTTPClient) -> None:
        self._http = http

    def caption(
        self,
        *,
        video_path: str,
        api_key: str,
        with_audio: bool,
        instructions: str | None = None,
    ) -> str:
        path = Path(video_path)
        if not path.is_file():
            raise VideoCaptionerError(f"Clip not found: {video_path}", status_code=400)

        size = path.stat().st_size
        if size > _MAX_INLINE_BYTES:
            mb = size / (1024 * 1024)
            raise VideoCaptionerError(
                f"Clip is too large to caption ({mb:.0f}MB). Trim or split it into "
                "shorter scenes first (under ~14MB).",
                status_code=413,
            )

        suffix = path.suffix.lower()
        is_image = suffix in _IMAGE_MIME_BY_SUFFIX
        mime_type = (
            _IMAGE_MIME_BY_SUFFIX.get(suffix, "image/png")
            if is_image
            else _MIME_BY_SUFFIX.get(suffix, "video/mp4")
        )
        try:
            encoded = base64.b64encode(path.read_bytes()).decode()
        except OSError as exc:
            raise VideoCaptionerError(f"Could not read clip: {exc}", status_code=400) from exc

        # A caller-supplied system prompt (the per-LoRA auto-prompt template)
        # overrides the default captioning instruction; otherwise we use the
        # dataset-captioning instruction.
        system_text = instructions if instructions is not None else _instructions(
            with_audio=with_audio, is_image=is_image
        )
        user_parts: list[JSONValue] = [
            {"text": system_text},
            {"inlineData": {"mimeType": mime_type, "data": encoded}},
        ]
        payload: dict[str, JSONValue] = {
            "contents": [{"role": "user", "parts": user_parts}],
            # gemini-2.5-flash is a *thinking* model: reasoning tokens come out
            # of the same maxOutputTokens budget. A verbose per-LoRA auto-prompt
            # template (8 structured fields + rules) makes the model think
            # heavily, blow past 1024 tokens before emitting any visible text,
            # and return finishReason=MAX_TOKENS — which surfaces as "caption
            # was truncated". Captioning/auto-prompt is explicit formatting, not
            # reasoning, so we disable thinking (predictable budget, faster,
            # cheaper) and raise the ceiling for user-edited verbose templates.
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 2048,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }

        logger.info(
            "gemini.caption request bytes=%d mime=%s with_audio=%s", size, mime_type, with_audio
        )
        status = 0
        response: HttpResponseLike | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = self._http.post(
                    _GEMINI_URL,
                    headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
                    json_payload=payload,
                    timeout=120,
                )
            except HttpTimeoutError as exc:
                raise VideoCaptionerError("Captioning timed out. Try a shorter clip.", status_code=504) from exc
            except Exception as exc:  # noqa: BLE001 — surface any transport failure as a clean error
                raise VideoCaptionerError(f"Captioning request failed: {exc}") from exc

            status = response.status_code
            if status == 200:
                break
            if status in _RETRYABLE_STATUS and attempt < _MAX_ATTEMPTS - 1:
                logger.warning(
                    "gemini.caption transient error status=%d attempt=%d/%d; retrying",
                    status,
                    attempt + 1,
                    _MAX_ATTEMPTS,
                )
                time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue
            break

        if status != 200:
            if status in _RETRYABLE_STATUS:
                raise VideoCaptionerError(
                    f"Gemini is temporarily unavailable ({status}). Please try again in a moment.",
                    status_code=502,
                )
            logger.warning("gemini.caption rejected request with status=%d", status)
            raise VideoCaptionerError(
                f"Gemini rejected the captioning request ({status}). Check the API key and request, then try again.",
                status_code=502,
            )

        assert response is not None
        try:
            parsed = _GeminiResponsePayload.model_validate(response.json())
        except ValidationError as exc:
            raise VideoCaptionerError("Could not parse the captioning response.", status_code=502) from exc

        candidate = parsed.candidates[0]
        caption = candidate.content.parts[0].text.strip()
        if not caption:
            raise VideoCaptionerError("The captioner returned an empty caption.", status_code=502)
        # A truncated caption (hit the token budget) is unusable for training —
        # surface it so it isn't silently saved and later rejected on export.
        if candidate.finishReason and candidate.finishReason.upper() == "MAX_TOKENS":
            raise VideoCaptionerError(
                "The caption was truncated (token limit). Try captioning a "
                "shorter clip or re-run captioning.",
                status_code=502,
            )
        return caption

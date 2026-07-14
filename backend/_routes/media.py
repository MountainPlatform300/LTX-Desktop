"""Route handlers for /api/media/* — used by external clients that
need ffmpeg-backed extraction (Premiere UXP plugin, today; future
Adobe Audition / After Effects panels likely tomorrow).

Both routes follow the same shape as the rest of the backend:

  - camelCase request/response bodies (Pydantic models in api_types).
  - 404/422 errors mapped via _routes._errors.HTTPError so the
    frontend (and the plugin's TypeScript client) sees structured
    `{ code, message }` payloads, not FastAPI's raw 422s.
  - Localhost-only by virtue of the existing CORS allowlist + auth
    middleware; we don't re-check `request.client.host` here.

We deliberately don't attempt to *validate the queue contract* in
these routes — the path returned by extract is just bytes-on-disk
that the next /api/queue/items POST will consume as `imagePath`
or `audioPath`. If a future schema change adds extra constraints
(e.g. "imagePath must be PNG"), it lives in the queue route's
validation, not here. Keeps the media handler free of queue-shape
coupling.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from _routes._errors import HTTPError
from api_types import (
    ExtractAudioRequest,
    ExtractAudioResponse,
    ExtractFrameRequest,
    ExtractFrameResponse,
)
from app_handler import AppHandler
from handlers import MediaExtractionError
from state import get_state_service

router = APIRouter(prefix="/api/media", tags=["media"])


@router.post("/extract-frame", response_model=ExtractFrameResponse)
def route_extract_frame(
    body: ExtractFrameRequest,
    handler: AppHandler = Depends(get_state_service),
) -> ExtractFrameResponse:
    """POST /api/media/extract-frame — pull a single frame to a PNG.

    Used by the Premiere plugin to convert "selected clip + playhead
    time" into a path the queue can consume as `imagePath`. Out-of-
    range times are silently clamped to last-frame by ffmpeg's seek
    behaviour, so the plugin doesn't need to query clip duration
    upfront — the latency cost of an extra round trip would dominate
    any UX win from rejecting bogus timestamps.

    Returns 422 when `sourcePath` doesn't exist, isn't a regular
    file, or ffmpeg can't decode it. The frontend should treat 422
    as "show the user 'this clip can't be used', don't retry."
    """
    try:
        path = handler.media.extract_frame(
            source_path=body.sourcePath,
            time_seconds=body.timeSeconds,
        )
    except MediaExtractionError as exc:
        raise HTTPError(
            422,
            exc.reason,
            code="MEDIA_EXTRACT_FAILED",
        ) from None
    return ExtractFrameResponse(path=str(path))


@router.post("/extract-audio", response_model=ExtractAudioResponse)
def route_extract_audio(
    body: ExtractAudioRequest,
    handler: AppHandler = Depends(get_state_service),
) -> ExtractAudioResponse:
    """POST /api/media/extract-audio — pull a slice of audio to WAV.

    `durationSeconds=0` means "to end of file"; positive values are
    bounded by both Pydantic (le=300) and a defence-in-depth
    ceiling in the handler. Output is mono / 48kHz / WAV — the
    format the a2v MLX pipeline ingests directly without another
    transcode.

    Same 422 contract as extract-frame: source-side problems
    surface as 422 with a stable error code; ffmpeg-internal
    failures (rare in practice) also use 422 since the client's
    only useful response is "show the user, don't retry."
    """
    try:
        path = handler.media.extract_audio(
            source_path=body.sourcePath,
            start_seconds=body.startSeconds,
            duration_seconds=body.durationSeconds,
        )
    except MediaExtractionError as exc:
        raise HTTPError(
            422,
            exc.reason,
            code="MEDIA_EXTRACT_FAILED",
        ) from None
    return ExtractAudioResponse(path=str(path))

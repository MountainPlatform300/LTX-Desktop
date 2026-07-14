"""Route handlers for GET/POST /api/settings."""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Depends, Request

from _routes._admin_guard import guard_admin_permission
from state.app_settings import SettingsResponse, UpdateSettingsRequest, to_settings_response
from api_types import StatusResponse
from state import get_state_service
from app_handler import AppHandler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["settings"])

CredentialName = Literal["ltx", "fal", "gemini", "pexels", "runpod", "hugging-face"]
_CREDENTIAL_FIELDS: dict[CredentialName, str] = {
    "ltx": "ltx_api_key",
    "fal": "fal_api_key",
    "gemini": "gemini_api_key",
    "pexels": "pexels_api_key",
    "runpod": "runpod_api_key",
    "hugging-face": "hf_token",
}


@router.get("/settings", response_model=SettingsResponse)
def route_get_settings(handler: AppHandler = Depends(get_state_service)) -> SettingsResponse:
    response = to_settings_response(handler.settings.get_settings_snapshot())
    response.models_dir = str(handler.settings.models_dir)
    return response


@router.post("/settings", response_model=StatusResponse)
def route_post_settings(
    req: UpdateSettingsRequest,
    request: Request,
    handler: AppHandler = Depends(get_state_service),
) -> StatusResponse:
    patch_data = req.model_dump(exclude_unset=True)
    if "models_dir" in patch_data or "modelsDir" in patch_data:
        guard_admin_permission(request)

    _, _after, changed_paths = handler.settings.update_settings(req)
    changed_roots = {path.split(".", 1)[0] for path in changed_paths}

    logger.info(
        "Applied settings patch (changed=%s)",
        ", ".join(sorted(changed_roots)) if changed_roots else "none",
    )

    return StatusResponse(status="ok")


@router.delete(
    "/settings/credentials/{credential}",
    response_model=StatusResponse,
)
def route_delete_credential(
    credential: CredentialName,
    handler: AppHandler = Depends(get_state_service),
) -> StatusResponse:
    """Remove one stored API credential without exposing its value."""
    handler.settings.clear_secret(_CREDENTIAL_FIELDS[credential])
    logger.info("Removed stored credential (%s)", credential)
    return StatusResponse(status="ok")

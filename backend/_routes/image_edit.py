"""Route handlers for /api/generate-image-edit."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import GenerateImageEditRequest, GenerateImageEditResponse
from state import get_state_service
from app_handler import AppHandler

router = APIRouter(prefix="/api", tags=["image-edit"])


@router.post("/generate-image-edit", response_model=GenerateImageEditResponse)
def route_generate_image_edit(
    req: GenerateImageEditRequest,
    handler: AppHandler = Depends(get_state_service),
) -> GenerateImageEditResponse:
    """POST /api/generate-image-edit — FLUX.2 [klein] 9B image generation/editing."""
    return handler.image_edit.generate(req)

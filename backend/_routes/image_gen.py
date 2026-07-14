"""Route handlers for /api/generate-image."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import GenerateImageModelsSpecsResponse, GenerateImageRequest, GenerateImageResponse
from state import get_state_service
from app_handler import AppHandler

router = APIRouter(prefix="/api", tags=["image"])


@router.get("/generate/image-models-specs", response_model=GenerateImageModelsSpecsResponse)
def route_image_model_specs(
    handler: AppHandler = Depends(get_state_service),
) -> GenerateImageModelsSpecsResponse:
    """GET /api/generate/image-models-specs — catalog of downloadable image models."""
    return handler.image_generation.get_model_specs()


@router.post("/generate-image", response_model=GenerateImageResponse)
def route_generate_image(
    req: GenerateImageRequest,
    handler: AppHandler = Depends(get_state_service),
) -> GenerateImageResponse:
    """POST /api/generate-image."""
    return handler.image_generation.generate(req)


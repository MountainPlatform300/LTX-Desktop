"""Canonical image-generation model catalog used by backend and frontend.

Mirrors ``api_model_specs`` for video: a single backend-owned catalog of
downloadable image models that the Gen Space image picker renders. Only the
two models with inference wired in the app are catalogued: Z-Image Turbo
(local text-to-image default) and FLUX.2 [klein] 9B (local instruction-based
image editing). Other open-weight image models were removed from the picker
to keep the surface focused; they can be re-added when their inference is
wired.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from api_types import GenerateImageModelsSpecsResponse, ImageModelSpecApi
from runtime_config.model_download_specs import (
    IMG_GEN_MODEL_CP_ID,
    ModelCheckpointID,
    get_model_cp_spec,
    is_cp_downloaded,
)

ImageModelInferenceStatus = Literal["available", "coming_soon"]


@dataclass(frozen=True, slots=True)
class ImageModelSpec:
    id: str
    display_name: str
    checkpoint_id: ModelCheckpointID
    description: str
    license: str
    gated: bool
    inference_status: ImageModelInferenceStatus
    default_resolution: tuple[int, int]
    supported_resolutions: tuple[tuple[int, int], ...]
    # True for instruction-based editing models served by the
    # /api/generate-image-edit endpoint (accept input reference images). Not
    # required for basic text-to-image, so the first-run recommendation skips
    # these even when `inference_status == "available"`.
    is_edit_model: bool = False


_Z_IMAGE_RESOLUTIONS: tuple[tuple[int, int], ...] = (
    (768, 768),
    (1024, 1024),
    (1024, 1536),
    (1536, 1024),
)

IMAGE_MODELS: tuple[ImageModelSpec, ...] = (
    ImageModelSpec(
        id="z-image-turbo",
        display_name="Z-Image Turbo",
        checkpoint_id=IMG_GEN_MODEL_CP_ID,
        description="Fast distilled text-to-image model. The default local image model.",
        license="Apache-2.0",
        gated=False,
        inference_status="available",
        default_resolution=(1024, 1024),
        supported_resolutions=_Z_IMAGE_RESOLUTIONS,
    ),
    ImageModelSpec(
        id="flux-2-klein-9b",
        display_name="FLUX.2 [klein] 9B",
        checkpoint_id="flux-2-klein-9b",
        description=(
            "FLUX.2 [klein] 9B \u2014 unified text-to-image and instruction-based "
            "image editing (single + multi-reference). The local image-editing "
            "model: add an input image in Generate Image to edit it. Gated on "
            "HuggingFace (non-commercial): sign in via Settings \u2192 HuggingFace "
            "before downloading."
        ),
        license="FLUX Non-Commercial License",
        gated=True,
        inference_status="available",
        default_resolution=(1024, 1024),
        supported_resolutions=_Z_IMAGE_RESOLUTIONS,
        is_edit_model=True,
    ),
)


def get_image_model_spec(model_id: str) -> ImageModelSpec | None:
    return next((spec for spec in IMAGE_MODELS if spec.id == model_id), None)


def get_default_image_model_spec() -> ImageModelSpec:
    # The first entry is the default (Z-Image Turbo) — keeps back-compat with
    # pre-catalog callers that omitted `model` and with IMG_GEN_MODEL_CP_ID.
    return IMAGE_MODELS[0]


def resolve_image_model_spec(model_id: str | None) -> ImageModelSpec:
    if model_id is None:
        return get_default_image_model_spec()
    spec = get_image_model_spec(model_id)
    if spec is None:
        return get_default_image_model_spec()
    return spec


def _to_api(spec: ImageModelSpec, *, models_dir: Path) -> ImageModelSpecApi:
    cp_spec = get_model_cp_spec(spec.checkpoint_id)
    return ImageModelSpecApi(
        id=spec.id,
        display_name=spec.display_name,
        checkpoint_id=spec.checkpoint_id,
        repo_id=cp_spec.repo_id,
        description=spec.description,
        license=spec.license,
        gated=spec.gated,
        inference_status=spec.inference_status,
        downloaded=is_cp_downloaded(models_dir, spec.checkpoint_id),
        default_resolution=spec.default_resolution,
        supported_resolutions=list(spec.supported_resolutions),
        size_bytes=cp_spec.expected_size_bytes,
        is_edit_model=spec.is_edit_model,
    )


def build_image_model_specs_response(models_dir: Path) -> GenerateImageModelsSpecsResponse:
    return GenerateImageModelsSpecsResponse(
        models=[_to_api(spec, models_dir=models_dir) for spec in IMAGE_MODELS],
    )

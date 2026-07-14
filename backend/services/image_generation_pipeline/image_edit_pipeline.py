"""Image editing pipeline protocol with multi-reference support.

FLUX.2 [klein] 9B unifies text-to-image and instruction-based editing in one
model: ``generate`` does txt2img, ``generate_with_references`` conditions on up
to 4 reference images (single/multi-reference editing). Same Protocol + real +
fake convention as the rest of the services.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from services.services_utils import ImagePipelineOutputLike, PILImageType


@runtime_checkable
class ImageEditPipeline(Protocol):
    @staticmethod
    def create(
        model_path: str,
        device: str | None = None,
    ) -> "ImageEditPipeline":
        ...

    def generate(
        self,
        prompt: str,
        height: int,
        width: int,
        guidance_scale: float,
        num_inference_steps: int,
        seed: int,
    ) -> ImagePipelineOutputLike:
        ...

    def generate_with_references(
        self,
        prompt: str,
        reference_images: Sequence[PILImageType],
        height: int,
        width: int,
        guidance_scale: float,
        num_inference_steps: int,
        seed: int,
    ) -> ImagePipelineOutputLike:
        ...

    def to(self, device: str) -> None:
        ...

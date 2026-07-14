"""FLUX.2 [klein] 9B image generation and editing pipeline wrapper.

Wraps ``diffusers.Flux2KleinPipeline`` so the rest of the app talks to it
through the ``ImageEditPipeline`` protocol. Klein is a unified model: pass
``image=None`` (via ``generate``) for text-to-image, or pass reference images
(via ``generate_with_references``) for single/multi-reference instruction
editing.

On CUDA/MPS the diffusers pipeline is run with ``enable_model_cpu_offload`` so
the 9B + 8B-Qwen3 stack can run on cards smaller than the full ~29GB footprint
(layers are swapped in/out of VRAM between steps). On CPU the pipeline is moved
whole, which is functionally correct but impractically slow — intended only for
tests/fakes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from PIL.Image import Image as PILImage

from services.services_utils import ImagePipelineOutputLike, PILImageType, get_device_type


@dataclass(slots=True)
class _KleinOutput:
    images: Sequence[PILImageType]


class KleinImageEditPipeline:
    @staticmethod
    def create(
        model_path: str,
        device: str | None = None,
    ) -> "KleinImageEditPipeline":
        return KleinImageEditPipeline(model_path=model_path, device=device)

    def __init__(self, model_path: str, device: str | None = None) -> None:
        from diffusers import Flux2KleinPipeline  # type: ignore[reportUnknownVariableType]

        self._device: str | None = None
        self._cpu_offload_active = False
        # low_cpu_mem_usage=False forces every component (notably the Qwen3
        # text encoder) to be fully materialized on CPU at load time instead of
        # the default meta-init + lazy-materialize path. The meta path is fragile
        # when the process is under memory pressure or after another heavy
        # pipeline (LTX/fp8) has run: accelerate can leave some Qwen3 params on
        # the `meta` device (tied-weight/data_ptr heuristics), and the very next
        # `enable_model_cpu_offload()` call — whose first step is `self.to("cpu")`
        # — then dies with "Cannot copy out of meta tensor". Full materialization
        # costs a bit more transient host RAM but removes the meta path entirely,
        # which is what makes a second Klein load (after an LTX animate) reliable.
        self.pipeline = Flux2KleinPipeline.from_pretrained(  # type: ignore[reportUnknownMemberType]
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
        if device is not None:
            self.to(device)

    def _resolve_generator_device(self) -> str:
        if self._cpu_offload_active:
            return "cuda"
        if self._device is not None:
            return self._device
        execution_device = getattr(self.pipeline, "_execution_device", None)
        return get_device_type(execution_device)

    @staticmethod
    def _normalize_output(output: object) -> ImagePipelineOutputLike:
        images = getattr(output, "images", None)
        if not isinstance(images, Sequence):
            raise RuntimeError("Unexpected Klein pipeline output format: missing images sequence")

        images_list = cast(Sequence[object], images)
        validated: list[PILImageType] = []
        for img in images_list:
            if not isinstance(img, PILImage):
                raise RuntimeError("Unexpected Klein pipeline output format: images must be PIL.Image instances")
            validated.append(img)
        return _KleinOutput(images=validated)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        height: int,
        width: int,
        guidance_scale: float,
        num_inference_steps: int,
        seed: int,
    ) -> ImagePipelineOutputLike:
        generator = torch.Generator(device=self._resolve_generator_device()).manual_seed(seed)
        pipeline = cast(Any, self.pipeline)
        output = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            output_type="pil",
            return_dict=True,
        )
        return self._normalize_output(output)

    @torch.inference_mode()
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
        """Generate an image conditioned on up to 4 reference images."""
        refs = list(reference_images)[:4]
        generator = torch.Generator(device=self._resolve_generator_device()).manual_seed(seed)
        pipeline = cast(Any, self.pipeline)
        output = pipeline(
            prompt=prompt,
            image=refs,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            output_type="pil",
            return_dict=True,
        )
        return self._normalize_output(output)

    def to(self, device: str) -> None:
        runtime_device = get_device_type(device)
        if runtime_device in ("cuda", "mps"):
            self.pipeline.enable_model_cpu_offload()  # type: ignore[reportUnknownMemberType]
            self._cpu_offload_active = True
        else:
            self._cpu_offload_active = False
            self.pipeline.to(runtime_device)  # type: ignore[reportUnknownMemberType]
        self._device = runtime_device

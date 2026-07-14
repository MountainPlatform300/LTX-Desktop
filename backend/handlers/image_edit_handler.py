"""Image editing orchestration handler (FLUX.2 [klein] 9B, local).

Routes a Klein edit/generate request through the single-flight generation slot
and the pipeline loader, mirroring `ImageGenerationHandler` for progress +
cancel semantics. Klein is local-only: under `force_api_generations` the
endpoint returns 501 so the UI can fall back to the remote text-to-image path
instead of silently no-op'ing.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from threading import RLock
from typing import TYPE_CHECKING

from PIL import Image

from _routes._errors import HTTPError
from api_types import GenerateImageEditCompleteResponse, GenerateImageEditCancelledResponse, GenerateImageEditRequest, GenerateImageEditResponse
from handlers.base import StateHandlerBase
from handlers.generation_handler import GenerationHandler
from handlers.pipelines_handler import PipelinesHandler
from state.app_state_types import AppState

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)

MAX_REFERENCE_IMAGES = 4
KLEIN_NATIVE_MAX_EDGE = 1024
KLEIN_DISTILLED_GUIDANCE = 1.0


def _round_down_16(value: float) -> int:
    return max(16, (int(value) // 16) * 16)


def _klein_native_dimensions(width: int, height: int) -> tuple[int, int]:
    """Keep diffusion in Klein's coherent native range.

    GenSpace's output presets can be much larger (2048p 16:9 is about 7.4 MP).
    Klein is substantially more reliable around its reference 1024px range, so
    larger outputs are generated there and resized after inference.
    """
    longest = max(width, height)
    if longest <= KLEIN_NATIVE_MAX_EDGE:
        return width, height
    scale = KLEIN_NATIVE_MAX_EDGE / longest
    return _round_down_16(width * scale), _round_down_16(height * scale)


class ImageEditHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        generation_handler: GenerationHandler,
        pipelines_handler: PipelinesHandler,
        config: RuntimeConfig,
    ) -> None:
        super().__init__(state, lock, config)
        self._generation = generation_handler
        self._pipelines = pipelines_handler

    def generate(self, req: GenerateImageEditRequest) -> GenerateImageEditResponse:
        if self.config.force_api_generations:
            raise HTTPError(
                501,
                "Local image editing isn't available while generations are forced to the API.",
                code="KLEIN_UNAVAILABLE",
            )

        if self._generation.is_generation_running():
            raise HTTPError(409, "Generation already in progress")

        self._generation.clear_cancel_token()

        width = _round_down_16(req.width)
        height = _round_down_16(req.height)
        native_width, native_height = _klein_native_dimensions(width, height)
        num_images = max(1, min(12, req.numImages))

        generation_id = uuid.uuid4().hex[:8]
        settings = self.state.app_settings.model_copy(deep=True)
        if settings.seed_locked:
            seed = settings.locked_seed
            logger.info("Using locked seed for Klein edit: %s", seed)
        elif self.config.dev_mode:
            seed = 1000
        else:
            seed = int(time.time()) % 2147483647

        ref_images: list[Image.Image] = []
        for ref_path in req.referenceImages[:MAX_REFERENCE_IMAGES]:
            try:
                ref_images.append(Image.open(ref_path).convert("RGB"))
            except Exception as exc:
                logger.warning("Could not load reference image %s: %s", ref_path, exc)

        try:
            klein = self._pipelines.load_klein_to_gpu()
            self._generation.start_generation(generation_id)
            self._generation.update_progress("loading_model", 5, 0, req.numSteps)

            outputs: list[str] = []
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            for i in range(num_images):
                if self._generation.is_generation_cancelled():
                    raise RuntimeError("Generation was cancelled")

                progress = 15 + int((i / num_images) * 80)
                self._generation.update_progress("inference", progress, i, num_images)

                if ref_images:
                    result = klein.generate_with_references(
                        prompt=req.prompt,
                        reference_images=ref_images,
                        height=native_height,
                        width=native_width,
                        guidance_scale=KLEIN_DISTILLED_GUIDANCE,
                        num_inference_steps=req.numSteps,
                        seed=seed + i,
                    )
                else:
                    result = klein.generate(
                        prompt=req.prompt,
                        height=native_height,
                        width=native_width,
                        guidance_scale=KLEIN_DISTILLED_GUIDANCE,
                        num_inference_steps=req.numSteps,
                        seed=seed + i,
                    )

                output_path = self.config.outputs_dir / f"klein_edit_{timestamp}_{uuid.uuid4().hex[:8]}.png"
                output_image = result.images[0]
                if (native_width, native_height) != (width, height):
                    self._generation.update_progress(
                        "upscaling", progress, i, num_images
                    )
                    output_image = output_image.resize(
                        (width, height), Image.Resampling.LANCZOS
                    )
                output_image.save(str(output_path))
                outputs.append(str(output_path))

            if self._generation.is_generation_cancelled():
                raise RuntimeError("Generation was cancelled")

            self._generation.update_progress("complete", 100, num_images, num_images)
            self._generation.complete_generation(outputs)
            # Klein runs with model-cpu-offload, so the weights are back on CPU
            # after inference — but the CUDA caching allocator still holds the
            # freed activation/latent blocks, pinning VRAM while the pipeline
            # sits idle. Release them so idle VRAM drops to ~0 (the pipeline
            # stays loaded in gpu_slot for reuse on the next edit).
            self._pipelines.release_gpu_cache()
            return GenerateImageEditCompleteResponse(status="complete", image_paths=outputs)
        except HTTPError:
            self._generation.fail_generation("Klein edit failed")
            raise
        except Exception as e:
            self._generation.fail_generation(str(e))
            if "cancelled" in str(e).lower():
                logger.info("Klein image edit cancelled by user")
                return GenerateImageEditCancelledResponse(status="cancelled")
            message = str(e)
            # A CUDA error mid-inference poisons the process's CUDA context for
            # the rest of its life; surface a clear restart instruction instead
            # of the raw, scary CUDA dump.
            if "CUDA error" in message or "cuda" in message.lower():
                raise HTTPError(
                    503,
                    "The GPU hit an error and can’t run this edit right now. "
                    "Restart the app to reset the GPU, then try again.",
                    code="GPU_ERROR",
                ) from e
            raise HTTPError(500, message) from e

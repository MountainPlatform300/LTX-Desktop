"""Pipeline lifecycle handler."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, cast

from _routes._errors import HTTPError
from api_types import LTXLocalModelId, ModelCheckpointID
from handlers.base import StateHandlerBase
from handlers.text_handler import TextHandler
from runtime_config.model_download_specs import (
    IMG_GEN_MODEL_CP_ID,
    get_downloaded_ltx_model_id,
    get_existing_cp_path,
    get_ltx_model_spec,
    is_cp_downloaded,
)
from runtime_config.runtime_policy import streaming_prefetch_count_for_mode
from services.interfaces import (
    A2VPipeline,
    DepthProcessorPipeline,
    FastVideoPipeline,
    ImageEditPipeline,
    ImageGenerationPipeline,
    GpuCleaner,
    IcLoraPipeline,
    PoseProcessorPipeline,
    RetakePipeline,
    VideoPipelineModelType,
)
from services.services_utils import device_supports_fp8, get_device_type
from state.app_state_types import (
    A2VPipelineState,
    AppState,
    CpuSlot,
    GpuGeneration,
    GenerationRunning,
    GpuSlot,
    ICLoraState,
    KleinPipelineState,
    PoseResources,
    RetakePipelineState,
    VideoPipelineState,
)

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)


def _pose_matches(
    current: PoseResources | None,
    want_pose: bool,
    pose_model_path: str | None,
) -> bool:
    """Cache-key predicate for pose resources on an IC-LoRA GpuSlot."""
    if not want_pose:
        # A non-pose request can reuse any state (pose_resources unused).
        return True
    # A pose request requires matching pose resources to be already loaded.
    return current is not None and current.pose_model_path == pose_model_path


class PipelinesHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        text_handler: TextHandler,
        gpu_cleaner: GpuCleaner,
        fast_video_pipeline_class: type[FastVideoPipeline],
        image_generation_pipeline_class: type[ImageGenerationPipeline],
        ic_lora_pipeline_class: type[IcLoraPipeline],
        depth_processor_pipeline_class: type[DepthProcessorPipeline],
        pose_processor_pipeline_class: type[PoseProcessorPipeline],
        a2v_pipeline_class: type[A2VPipeline],
        retake_pipeline_class: type[RetakePipeline],
        image_edit_pipeline_class: type[ImageEditPipeline] | None,
        config: RuntimeConfig,
    ) -> None:
        super().__init__(state, lock, config)
        self._text_handler = text_handler
        self._gpu_cleaner = gpu_cleaner
        self._fast_video_pipeline_class = fast_video_pipeline_class
        self._image_generation_pipeline_class = image_generation_pipeline_class
        self._ic_lora_pipeline_class = ic_lora_pipeline_class
        self._depth_processor_pipeline_class = depth_processor_pipeline_class
        self._pose_processor_pipeline_class = pose_processor_pipeline_class
        self._a2v_pipeline_class = a2v_pipeline_class
        self._retake_pipeline_class = retake_pipeline_class
        self._image_edit_pipeline_class = image_edit_pipeline_class
        self._runtime_device = get_device_type(self.config.device)

    def _ensure_no_running_generation(self) -> None:
        match self.state.active_generation:
            case GpuGeneration(state=GenerationRunning()) if self.state.gpu_slot is not None:
                raise RuntimeError("Generation already running; cannot swap pipelines")
            case _:
                return

    def _pipeline_matches_model_type(
        self, model_type: VideoPipelineModelType, *, lora_path: str | None
    ) -> bool:
        match self.state.gpu_slot:
            case GpuSlot(active_pipeline=VideoPipelineState(pipeline=pipeline, lora_path=active_lora)):
                return pipeline.pipeline_kind == model_type and active_lora == lora_path
            case _:
                return False

    def _assert_invariants(self) -> None:
        match self.state.gpu_slot:
            case GpuSlot(active_pipeline=active_pipeline):
                gpu_has_image_generation_pipeline = isinstance(active_pipeline, ImageGenerationPipeline)
            case _:
                gpu_has_image_generation_pipeline = False

        if gpu_has_image_generation_pipeline and self.state.cpu_slot is not None:
            raise RuntimeError("Invariant violation: image generation pipeline cannot be in both GPU and CPU slots")

    def _install_text_patches_if_needed(self) -> None:
        te = self.state.text_encoder
        if te is None:
            return
        te.service.install_patches(lambda: self.state)

    def _require_downloaded_ltx_model_id(self) -> LTXLocalModelId:
        model_id = get_downloaded_ltx_model_id(self.models_dir)
        if model_id is None:
            raise HTTPError(409, "NO_DOWNLOADED_LTX_MODEL")
        return model_id

    def _compile_if_enabled(self, state: VideoPipelineState) -> VideoPipelineState:
        if not self.state.app_settings.use_torch_compile:
            return state
        if state.is_compiled:
            return state
        if self._runtime_device == "mps":
            logger.info("Skipping torch.compile() for %s - not supported on MPS", state.pipeline.pipeline_kind)
            return state

        try:
            state.pipeline.compile_transformer()
            state.is_compiled = True
        except Exception as exc:
            logger.warning("Failed to compile transformer: %s", exc, exc_info=True)
        return state

    def _create_video_pipeline(
        self,
        model_type: VideoPipelineModelType,
        *,
        lora_path: str | None = None,
        lora_scale: float = 1.0,
    ) -> VideoPipelineState:
        gemma_root = self._text_handler.resolve_gemma_root()
        model_id = self._require_downloaded_ltx_model_id()
        spec = get_ltx_model_spec(model_id)
        checkpoint_path = str(get_existing_cp_path(self.models_dir, spec.model_cp))
        upsampler_path = str(get_existing_cp_path(self.models_dir, spec.upscale_cp))

        pipeline = self._fast_video_pipeline_class.create(
            checkpoint_path,
            gemma_root,
            upsampler_path,
            self.config.device,
            streaming_prefetch_count_for_mode(self.config.local_generations_mode),
            lora_path=lora_path,
            lora_scale=lora_scale,
        )

        state = VideoPipelineState(
            pipeline=pipeline,
            is_compiled=False,
            lora_path=lora_path,
            lora_scale=lora_scale,
        )
        return self._compile_if_enabled(state)

    def unload_gpu_pipeline(self) -> None:
        with self._lock:
            self._ensure_no_running_generation()
            self.state.gpu_slot = None
            self._assert_invariants()
        self._gpu_cleaner.cleanup()

    def release_gpu_cache(self) -> None:
        # Return the CUDA caching allocator's unused blocks to the driver while
        # keeping the resident pipeline loaded for reuse. After an inference the
        # allocator still holds the freed activation/latent blocks; with a
        # cpu-offloaded pipeline (Klein) the model itself is on CPU, so this
        # brings idle VRAM back to ~0 without paying for a reload on the next
        # request. Safe to call with a pipeline resident: it touches no state.
        self._gpu_cleaner.cleanup()

    def park_image_generation_pipeline_on_cpu(self) -> None:
        image_generation_pipeline: ImageGenerationPipeline | None = None

        with self._lock:
            if self.state.gpu_slot is None:
                return

            active = self.state.gpu_slot.active_pipeline
            if not isinstance(active, ImageGenerationPipeline):
                return

            if isinstance(self.state.active_generation, GpuGeneration) and isinstance(
                self.state.active_generation.state, GenerationRunning
            ):
                raise RuntimeError("Cannot park image generation pipeline while generation is running")

            image_generation_pipeline = active
            self.state.gpu_slot = None

        assert image_generation_pipeline is not None
        image_generation_pipeline.to("cpu")
        self._gpu_cleaner.cleanup()

        with self._lock:
            self.state.cpu_slot = CpuSlot(active_pipeline=image_generation_pipeline)
            self._assert_invariants()

    def load_image_generation_pipeline_to_gpu(self) -> ImageGenerationPipeline:
        with self._lock:
            if self.state.gpu_slot is not None:
                active = self.state.gpu_slot.active_pipeline
                if isinstance(active, ImageGenerationPipeline):
                    return active
                self._ensure_no_running_generation()

        image_generation_pipeline: ImageGenerationPipeline | None = None

        with self._lock:
            match self.state.cpu_slot:
                case CpuSlot(active_pipeline=stored):
                    image_generation_pipeline = stored
                    self.state.cpu_slot = None
                case _:
                    image_generation_pipeline = None

        if image_generation_pipeline is None:
            zit_path = get_existing_cp_path(self.models_dir, IMG_GEN_MODEL_CP_ID)
            image_generation_pipeline = self._image_generation_pipeline_class.create(str(zit_path), self._runtime_device)
        else:
            image_generation_pipeline.to(self._runtime_device)

        self._gpu_cleaner.cleanup()

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=image_generation_pipeline)
            self._assert_invariants()

        return image_generation_pipeline

    def _evict_gpu_pipeline_for_swap(self) -> None:
        should_park_image_generation_pipeline = False
        should_cleanup = False

        with self._lock:
            self._ensure_no_running_generation()
            if self.state.gpu_slot is None:
                return

            active = self.state.gpu_slot.active_pipeline
            if isinstance(active, ImageGenerationPipeline):
                should_park_image_generation_pipeline = True
            else:
                self.state.gpu_slot = None
                self._assert_invariants()
                should_cleanup = True

        if should_park_image_generation_pipeline:
            self.park_image_generation_pipeline_on_cpu()
        elif should_cleanup:
            self._gpu_cleaner.cleanup()

    def load_gpu_pipeline(
        self,
        model_type: VideoPipelineModelType,
        *,
        lora_path: str | None = None,
        lora_scale: float = 1.0,
    ) -> VideoPipelineState:
        self._install_text_patches_if_needed()

        state: VideoPipelineState | None = None
        with self._lock:
            if self._pipeline_matches_model_type(model_type, lora_path=lora_path):
                match self.state.gpu_slot:
                    case GpuSlot(active_pipeline=VideoPipelineState() as existing_state):
                        state = existing_state
                    case _:
                        pass

        if state is None:
            self._evict_gpu_pipeline_for_swap()
            state = self._create_video_pipeline(
                model_type, lora_path=lora_path, lora_scale=lora_scale
            )
            with self._lock:
                self.state.gpu_slot = GpuSlot(active_pipeline=state)
                self._assert_invariants()

        return state

    def _resolve_ic_lora_base(self, model_spec: object) -> tuple[Path, str | None]:
        """Pick the IC-LoRA base checkpoint + optional distilled LoRA overlay.

        Returns (base_checkpoint_path, distilled_lora_path_or_none). When the
        dev quality base is enabled AND both the dev checkpoint and the
        distilled v1.1 LoRA are downloaded, the dev checkpoint is used with the
        distilled LoRA stacked @0.5 (the ComfyUI dev + distilled-LoRA flow).
        Otherwise falls back to the distilled checkpoint with no overlay — the
        distilled LoRA is never applied to an already-distilled checkpoint.
        """
        quality_base_cp = getattr(model_spec, "quality_base_cp", None)
        distilled_lora_cp = getattr(model_spec, "distilled_lora_cp", None)
        if (
            self.state.app_settings.use_dev_quality_base
            and quality_base_cp is not None
            and distilled_lora_cp is not None
            and is_cp_downloaded(self.models_dir, quality_base_cp)
            and is_cp_downloaded(self.models_dir, distilled_lora_cp)
        ):
            return (
                get_existing_cp_path(self.models_dir, quality_base_cp),
                str(get_existing_cp_path(self.models_dir, distilled_lora_cp)),
            )
        base_cp: ModelCheckpointID = getattr(model_spec, "model_cp")
        return get_existing_cp_path(self.models_dir, base_cp), None

    def _expected_ic_lora_base_checkpoint_path(self) -> str:
        """Base path the cached IC-LoRA state SHOULD have (cache-key helper).

        Non-raising: returns "" when no LTX model is downloaded yet so the guard
        simply misses and the reload path raises the proper 409.
        """
        model_id = get_downloaded_ltx_model_id(self.models_dir)
        if model_id is None:
            return ""
        base_path, _ = self._resolve_ic_lora_base(get_ltx_model_spec(model_id))
        return str(base_path)

    def load_ic_lora(
        self,
        lora_path: str,
        depth_model_path: str,
        *,
        lora_scale: float = 1.0,
        pose_model_path: str | None = None,
        person_detector_model_path: str | None = None,
    ) -> ICLoraState:
        self._install_text_patches_if_needed()

        # Pose resources are loaded only when a pose-conditioned request asks
        # for them; they're part of the cache key so a non-pose request never
        # gets handed a state whose pose_resources are (un)loaded the wrong way.
        want_pose = pose_model_path is not None and person_detector_model_path is not None

        with self._lock:
            match self.state.gpu_slot:
                case GpuSlot(
                    active_pipeline=ICLoraState(
                        lora_path=current_lora_path,
                        lora_scale=current_lora_scale,
                        depth_model_path=current_depth_model_path,
                        pose_resources=current_pose,
                        base_checkpoint_path=current_base,
                    ) as state
                ) if (
                    current_lora_path == lora_path
                    and current_lora_scale == lora_scale
                    and current_depth_model_path == depth_model_path
                    and _pose_matches(current_pose, want_pose, pose_model_path)
                    and current_base == self._expected_ic_lora_base_checkpoint_path()
                ):
                    return state
                case _:
                    pass

        self._evict_gpu_pipeline_for_swap()
        model_id = self._require_downloaded_ltx_model_id()
        model_spec = get_ltx_model_spec(model_id)
        base_checkpoint_path, distilled_lora_path = self._resolve_ic_lora_base(model_spec)

        pipeline = self._ic_lora_pipeline_class.create(
            str(base_checkpoint_path),
            self._text_handler.resolve_gemma_root(),
            str(get_existing_cp_path(self.models_dir, model_spec.upscale_cp)),
            lora_path,
            self.config.device,
            streaming_prefetch_count_for_mode(self.config.local_generations_mode),
            distilled_lora_path=distilled_lora_path,
            lora_scale=lora_scale,
        )
        depth_pipeline = self._depth_processor_pipeline_class.create(depth_model_path, self.config.device)

        pose_resources: PoseResources | None = None
        if want_pose:
            pose_pipeline = self._pose_processor_pipeline_class.create(
                cast(str, pose_model_path),
                cast(str, person_detector_model_path),
                self.config.device,
            )
            pose_resources = PoseResources(
                pipeline=pose_pipeline,
                person_detector_model_path=cast(str, person_detector_model_path),
                pose_model_path=cast(str, pose_model_path),
            )

        state = ICLoraState(
            pipeline=pipeline,
            lora_path=lora_path,
            lora_scale=lora_scale,
            depth_pipeline=depth_pipeline,
            depth_model_path=depth_model_path,
            pose_resources=pose_resources,
            base_checkpoint_path=str(base_checkpoint_path),
        )

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=state)
            self._assert_invariants()
        return state

    def load_a2v_pipeline(self) -> A2VPipelineState:
        self._install_text_patches_if_needed()

        with self._lock:
            match self.state.gpu_slot:
                case GpuSlot(active_pipeline=A2VPipelineState() as state):
                    return state
                case _:
                    pass

        self._evict_gpu_pipeline_for_swap()
        model_id = self._require_downloaded_ltx_model_id()
        model_spec = get_ltx_model_spec(model_id)

        pipeline = self._a2v_pipeline_class.create(
            str(get_existing_cp_path(self.models_dir, model_spec.model_cp)),
            self._text_handler.resolve_gemma_root(),
            str(get_existing_cp_path(self.models_dir, model_spec.upscale_cp)),
            self.config.device,
            streaming_prefetch_count_for_mode(self.config.local_generations_mode),
        )
        state = A2VPipelineState(pipeline=pipeline)

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=state)
            self._assert_invariants()
        return state

    def load_retake_pipeline(self, *, distilled: bool = True) -> RetakePipelineState:
        self._install_text_patches_if_needed()

        quantized = device_supports_fp8(self.config.device)

        with self._lock:
            match self.state.gpu_slot:
                case GpuSlot(
                    active_pipeline=RetakePipelineState(distilled=current_distilled, quantized=current_quantized) as state
                ) if current_distilled == distilled and current_quantized == quantized:
                    return state
                case _:
                    pass

        self._evict_gpu_pipeline_for_swap()

        from ltx_core.quantization import QuantizationPolicy

        quantization = QuantizationPolicy.fp8_cast() if quantized else None
        model_id = self._require_downloaded_ltx_model_id()
        model_spec = get_ltx_model_spec(model_id)
        pipeline = self._retake_pipeline_class.create(
            checkpoint_path=str(get_existing_cp_path(self.models_dir, model_spec.model_cp)),
            gemma_root=self._text_handler.resolve_gemma_root(),
            device=self.config.device,
            streaming_prefetch_count=streaming_prefetch_count_for_mode(self.config.local_generations_mode),
            loras=[],
            quantization=quantization,
        )
        state = RetakePipelineState(pipeline=pipeline, distilled=distilled, quantized=quantized)

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=state)
            self._assert_invariants()
        return state

    def load_klein_to_gpu(self) -> ImageEditPipeline:
        # FLUX.2 [klein] 9B image editing pipeline. Like the other heavy
        # pipelines, it's loaded on demand and evicts whatever is resident
        # (via _evict_gpu_pipeline_for_swap). Reused if already active.
        with self._lock:
            if self.state.gpu_slot is not None:
                active = self.state.gpu_slot.active_pipeline
                if isinstance(active, KleinPipelineState):
                    return active.pipeline
                self._ensure_no_running_generation()

        self._evict_gpu_pipeline_for_swap()

        if self._image_edit_pipeline_class is None:
            raise HTTPError(
                501,
                "FLUX.2 Klein image editing isn't available in this build",
                code="KLEIN_UNAVAILABLE",
            )

        if not is_cp_downloaded(self.models_dir, "flux-2-klein-9b"):
            raise HTTPError(
                409,
                "FLUX.2 Klein 9B isn't downloaded. Download it from the Model Status menu.",
                code="KLEIN_NOT_DOWNLOADED",
            )

        klein_path = get_existing_cp_path(self.models_dir, "flux-2-klein-9b")
        assert self._image_edit_pipeline_class is not None
        klein_service = self._create_klein_with_retry(
            self._image_edit_pipeline_class, str(klein_path)
        )

        self._gpu_cleaner.cleanup()

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=KleinPipelineState(pipeline=klein_service))
            self._assert_invariants()

        return klein_service

    def _create_klein_with_retry(
        self,
        pipeline_cls: type[ImageEditPipeline],
        klein_path: str,
    ) -> ImageEditPipeline:
        """Create the Klein pipeline, recovering once from a CUDA hiccup.

        ``Flux2KleinPipeline.from_pretrained`` + ``enable_model_cpu_offload``
        do the first CUDA context work. If the driver / CUDA context is in a
        transient bad state (often the aftermath of a prior OOM or a driver
        blip — presents as ``CUDA error: unknown error``), the first attempt
        fails before anything is resident. Dropping any partial GPU state,
        clearing the caching allocator, and retrying once fixes the recoverable
        cases. A second failure means the CUDA context is poisoned for the
        lifetime of this process — surface a clear restart instruction instead
        of a raw CUDA dump so the user knows what to do.
        """
        try:
            return pipeline_cls.create(klein_path, self._runtime_device)
        except Exception as first_err:
            logger.warning(
                "Klein pipeline load failed (%s); clearing GPU and retrying once",
                first_err,
            )
            with self._lock:
                self.state.gpu_slot = None
            self._gpu_cleaner.cleanup()
            try:
                return pipeline_cls.create(klein_path, self._runtime_device)
            except Exception as second_err:
                message = str(second_err)
                if "CUDA error" in message or "cuda" in message.lower():
                    raise HTTPError(
                        503,
                        "The GPU hit an error and can’t run this edit right now. "
                        "Restart the app to reset the GPU, then try again.",
                        code="GPU_ERROR",
                    ) from second_err
                raise HTTPError(500, message) from second_err

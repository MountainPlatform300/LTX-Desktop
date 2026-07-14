"""Application state composition root and dependency wiring."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from state.app_settings import AppSettings
from handlers import (
    DownloadHandler,
    GenerationHandler,
    HealthHandler,
    HuggingFaceAuthHandler,
    IcLoraHandler,
    ImageEditHandler,
    ImageGenerationHandler,
    ModelsHandler,
    PipelinesHandler,
    SuggestGapPromptHandler,
    RetakeHandler,
    RuntimePolicyHandler,
    SettingsHandler,
    TextHandler,
    VideoGenerationHandler,
)
from handlers.media_handler import MediaHandler
from handlers.lora_training_handler import LoraTrainingHandler
from handlers.lora_inference_registry import LoraInferenceRegistry
from handlers.lora_inference_handler import LoraInferenceHandler
from handlers.lora_prompt_template import LoraPromptTemplateStore
from handlers.imported_lora_library import ImportedLoraLibrary
from handlers.lora_clip_jobs_runner import ClipJobsRunner
from handlers.lora_derivation_runner import LoraDerivationRunner
from handlers.lora_training_runner import LoraTrainingRunner
from handlers.queue_handler import QueueHandler
from handlers.queue_runner import QueueRunResult, QueueRunner
from runtime_config.runtime_config import RuntimeConfig
from services.interfaces import (
    A2VPipeline,
    DepthProcessorPipeline,
    FastVideoPipeline,
    ZitAPIClient,
    ImageGenerationPipeline,
    ImageEditPipeline,
    GpuCleaner,
    ClipProcessor,
    GpuInfo,
    HTTPClient,
    IcLoraPipeline,
    ImageEditor,
    LTXAPIClient,
    LoraPromptProfiler,
    ModelDownloader,
    PoseProcessorPipeline,
    RetakePipeline,
    TaskRunner,
    TextEncoder,
    TrainerTarget,
    VideoCaptioner,
    VideoProcessor,
    VideoRestyler,
    PexelsClient,
)
from services.trainer_target.local_trainer_target import LocalTrainerTarget
from services.lora_prompt_profiler.gemini_lora_prompt_profiler import (
    GeminiLoraPromptProfiler,
)
from services.lora_prompt_profiler.lora_prompt_profiler import NullLoraPromptProfiler
from state.app_state_types import AppState, TextEncoderState
from state.app_settings import should_video_generate_with_ltx_api
from state.queue_state import QueuePayload

from _routes._errors import HTTPError
from api_model_specs import validate_generate_video_request
from api_types import (
    CancelResponse,
    GenerateImageCompleteResponse,
    GenerateImageEditCompleteResponse,
    GenerateVideoCompleteResponse,
    LoraGenerateCompleteResponse,
    LoraStandardGenerateRequest,
)


class AppHandler:
    """Composition-only state service exposing typed domain handlers."""

    def __init__(
        self,
        config: RuntimeConfig,
        default_settings: AppSettings,
        http: HTTPClient,
        gpu_cleaner: GpuCleaner,
        model_downloader: ModelDownloader,
        gpu_info: GpuInfo,
        video_processor: VideoProcessor,
        text_encoder: TextEncoder,
        task_runner: TaskRunner,
        ltx_api_client: LTXAPIClient,
        zit_api_client: ZitAPIClient,
        fast_video_pipeline_class: type[FastVideoPipeline],
        image_generation_pipeline_class: type[ImageGenerationPipeline],
        ic_lora_pipeline_class: type[IcLoraPipeline],
        depth_processor_pipeline_class: type[DepthProcessorPipeline],
        pose_processor_pipeline_class: type[PoseProcessorPipeline],
        a2v_pipeline_class: type[A2VPipeline],
        retake_pipeline_class: type[RetakePipeline],
        image_edit_pipeline_class: type[ImageEditPipeline] | None,
        trainer_target: TrainerTarget,
        video_captioner: VideoCaptioner,
        clip_processor: ClipProcessor,
        image_editor: ImageEditor,
        video_restyler: VideoRestyler,
        pexels_client: PexelsClient,
        local_trainer: LocalTrainerTarget,
        lora_prompt_profiler: LoraPromptProfiler,
    ) -> None:
        self.config = config

        # Exposed for tests and diagnostics.
        self.http = http
        self.gpu_cleaner = gpu_cleaner
        self.model_downloader = model_downloader
        self.gpu_info = gpu_info
        self.video_processor = video_processor
        self.task_runner = task_runner
        self.ltx_api_client = ltx_api_client
        self.zit_api_client = zit_api_client
        self.fast_video_pipeline_class = fast_video_pipeline_class
        self.image_generation_pipeline_class = image_generation_pipeline_class
        self.ic_lora_pipeline_class = ic_lora_pipeline_class
        self.depth_processor_pipeline_class = depth_processor_pipeline_class
        self.pose_processor_pipeline_class = pose_processor_pipeline_class
        self.a2v_pipeline_class = a2v_pipeline_class
        self.retake_pipeline_class = retake_pipeline_class
        self.image_edit_pipeline_class = image_edit_pipeline_class
        self.trainer_target = trainer_target
        self.video_captioner = video_captioner
        self.clip_processor = clip_processor
        self.image_editor = image_editor
        self.video_restyler = video_restyler
        self.pexels_client = pexels_client
        self.local_trainer = local_trainer
        self.lora_prompt_profiler = lora_prompt_profiler

        self._lock = threading.RLock()

        self.state = AppState(
            downloading_session=None,
            gpu_slot=None,
            active_generation=None,
            cpu_slot=None,
            text_encoder=TextEncoderState(service=text_encoder),
            app_settings=default_settings.model_copy(deep=True),
        )

        # ============================================================
        # Handlers (wired in dependency order)
        # ============================================================

        self.settings = SettingsHandler(
            state=self.state,
            lock=self._lock,
            config=config,
        )

        self.models = ModelsHandler(
            state=self.state,
            lock=self._lock,
            config=config,
        )

        self.hf_auth = HuggingFaceAuthHandler(
            state=self.state,
            lock=self._lock,
            config=config,
        )

        self.downloads = DownloadHandler(
            state=self.state,
            lock=self._lock,
            models_handler=self.models,
            model_downloader=model_downloader,
            task_runner=task_runner,
            config=config,
        )

        self.text = TextHandler(
            state=self.state,
            lock=self._lock,
            config=config,
        )

        self.pipelines = PipelinesHandler(
            state=self.state,
            lock=self._lock,
            text_handler=self.text,
            gpu_cleaner=gpu_cleaner,
            fast_video_pipeline_class=fast_video_pipeline_class,
            image_generation_pipeline_class=image_generation_pipeline_class,
            ic_lora_pipeline_class=ic_lora_pipeline_class,
            depth_processor_pipeline_class=depth_processor_pipeline_class,
            pose_processor_pipeline_class=pose_processor_pipeline_class,
            a2v_pipeline_class=a2v_pipeline_class,
            retake_pipeline_class=retake_pipeline_class,
            image_edit_pipeline_class=image_edit_pipeline_class,
            config=config,
        )

        self.generation = GenerationHandler(state=self.state, lock=self._lock, config=config)

        self.video_generation = VideoGenerationHandler(
            state=self.state,
            lock=self._lock,
            generation_handler=self.generation,
            pipelines_handler=self.pipelines,
            text_handler=self.text,
            ltx_api_client=ltx_api_client,
            config=config,
        )

        self.image_generation = ImageGenerationHandler(
            state=self.state,
            lock=self._lock,
            config=config,
            generation_handler=self.generation,
            pipelines_handler=self.pipelines,
            zit_api_client=zit_api_client,
        )

        # Local instruction-based image editing (FLUX.2 [klein] 9B). Shares the
        # single-flight generation slot + pipeline loader with the other image
        # surfaces; local-only, so the handler 501s under force_api_generations.
        self.image_edit = ImageEditHandler(
            state=self.state,
            lock=self._lock,
            generation_handler=self.generation,
            pipelines_handler=self.pipelines,
            config=config,
        )

        # Durable batch generation queue. The handler owns the on-disk
        # ledger (queue.json) and the runner is the sole driver of the
        # video/image generation handlers for queued work — it
        # cooperates with other single-flight surfaces (retake / IC-LoRA
        # / editor regen) via the `is_slot_free` pre-check and the
        # `busy` dispatch outcome. Wired after the generation handlers
        # so the dispatch + validator closures can reference them.
        self.queue = QueueHandler(
            state=self.state,
            lock=self._lock,
            config=config,
            validate_payload=self._validate_queue_payload,
        )

        self.queue_runner = QueueRunner(
            queue_handler=self.queue,
            dispatch_fn=self._dispatch_queue_payload,
            is_slot_free=self._is_generation_slot_free,
        )

        self.health = HealthHandler(
            state=self.state,
            lock=self._lock,
            models_handler=self.models,
            gpu_info=gpu_info,
            config=config,
        )

        self.runtime_policy = RuntimePolicyHandler(config=config)

        self.media = MediaHandler(config=config)

        self.suggest_gap_prompt = SuggestGapPromptHandler(
            state=self.state,
            lock=self._lock,
            config=config,
            http=http,
        )

        self.retake = RetakeHandler(
            state=self.state,
            lock=self._lock,
            ltx_api_client=ltx_api_client,
            config=config,
            generation_handler=self.generation,
            pipelines_handler=self.pipelines,
            text_handler=self.text,
        )

        self.ic_lora = IcLoraHandler(
            state=self.state,
            lock=self._lock,
            generation_handler=self.generation,
            pipelines_handler=self.pipelines,
            text_handler=self.text,
            video_processor=video_processor,
            media_handler=self.media,
            config=config,
        )

        # LoRA trainer: durable control plane for remote GPU training.
        # Three ledgers (datasets / preprocessed / training) under the
        # shared lock; the runner reconciles in-flight remote jobs in
        # the background and is started from the FastAPI lifespan, same
        # as the queue runner.
        self.lora_training = LoraTrainingHandler(
            state=self.state,
            lock=self._lock,
            config=config,
            video_captioner=video_captioner,
            clip_processor=clip_processor,
            image_editor=image_editor,
            video_restyler=video_restyler,
            pexels_client=pexels_client,
            local_trainer=local_trainer,
            image_edit_handler=self.image_edit,
        )

        self.lora_training_runner = LoraTrainingRunner(
            handler=self.lora_training,
            trainer_target=trainer_target,
            settings_handler=self.settings,
            config=config,
            clip_processor=clip_processor,
            # Local WSL2 training shares the single GPU with this inference
            # server; free the resident 22B model before a local run so the
            # trainer gets the full card instead of crashing on VRAM starvation.
            free_inference_gpu=self.pipelines.unload_gpu_pipeline,
        )

        # Local, GPU-free sprite/preview generation for the curation
        # gallery. Runs on its own bounded pool so it never queues behind
        # the remote training reconciler.
        self.lora_clip_jobs_runner = ClipJobsRunner(
            handler=self.lora_training,
            clip_processor=clip_processor,
        )

        # Multi-stage AI target/variant generation. Runs serially on its
        # own thread: optional Nano Banana frame edit, then a local
        # IC-LoRA depth/canny drive (waits for the single-flight GPU) or a
        # remote Kling motion-control drive.
        self.lora_derivation_runner = LoraDerivationRunner(
            handler=self.lora_training,
            ic_lora=self.ic_lora,
            generation=self.generation,
        )

        # In-app LoRA inference registry (Gen Space "Apply LoRA" picker source).
        # Derived from the training ledger + imported-LoRA library + models dir,
        # so a freshly completed training job or a newly imported adapter appears
        # without an explicit bridge call.
        self.imported_lora_library = ImportedLoraLibrary(
            state=self.state,
            lock=self._lock,
            config=config,
        )

        # Per-LoRA prompt-writing-assistant template overrides (durable, keyed
        # by registry entry id). Overlayed onto every registry entry, so a
        # user edit of one LoRA's system prompt persists across restarts and
        # works for any entry kind (official / user-trained / imported).
        self.lora_prompt_templates = LoraPromptTemplateStore(
            state=self.state,
            lock=self._lock,
            config=config,
        )

        self.lora_inference_registry = LoraInferenceRegistry(
            training_handler=self.lora_training,
            imported_library=self.imported_lora_library,
            template_store=self.lora_prompt_templates,
            models_dir=self.models.models_dir,
        )

        # In-app LoRA inference generate: routes a registry-picked LoRA to the
        # right pipeline by variant (standard → fast video, union_control /
        # video_input_ic_lora → IC-LoRA). The queue dispatches `kind="lora"`
        # payloads here; the synchronous route is the same call.
        self.lora_inference = LoraInferenceHandler(
            state=self.state,
            lock=self._lock,
            config=config,
            registry=self.lora_inference_registry,
            video_generation=self.video_generation,
            ic_lora=self.ic_lora,
            video_captioner=video_captioner,
            template_store=self.lora_prompt_templates,
            clip_processor=clip_processor,
            imported_library=self.imported_lora_library,
            prompt_profiler=self.lora_prompt_profiler,
            training_handler=self.lora_training,
        )

        self.downloads.cleanup_downloading_dir()

        self.load_persistent_state(default_settings)

    def load_persistent_state(self, default_settings: AppSettings) -> None:
        """Load persisted state from disk (settings, HF auth token, etc.)."""
        self.settings.load_settings(default_settings)
        self.hf_auth.load_token()
        # LoRA ledgers are durable; load applies crash recovery
        # (re-poll handled remote jobs, reset handle-less in-flight items)
        # before the reconciler thread starts.
        self.lora_training.load_state()
        # Imported-LoRA library ledger (user-supplied adapter weights) — durable,
        # pruned of entries whose weights file has disappeared on disk.
        self.imported_lora_library.load_state()
        # Per-LoRA prompt-template overrides (durable, keyed by entry id).
        self.lora_prompt_templates.load_state()
        # The generation queue is durable; load re-marks any item left
        # `running` by a prior crash as `pending` so the runner re-claims
        # it once the lifespan starts the runner thread.
        self.queue.load_queue()

    # ------------------------------------------------------------------
    # Queue dispatch + validation closures
    # ------------------------------------------------------------------
    # These translate the queue's narrow contract (a `QueuePayload` in,
    # a `QueueRunResult` out; a free/busy slot probe) into calls on the
    # concrete generation handlers. Keeping the translation in the
    # composition root means the runner and handler stay decoupled from
    # the video/image response unions and from the spec validator, and
    # tests can substitute fakes without touching either.

    def _is_generation_slot_free(self) -> bool:
        return not self.generation.is_generation_running()

    def cancel_generation(self):
        """Cancel the active generation and any running queue item.

        Flips the `GenerationHandler` cancel flag (observed by the
        inference loop's pre/post-call checks) AND marks the running
        queue ledger entry as cancelled directly. The direct ledger
        cancellation is the key piece for a *stuck* generation: when the
        underlying CUDA call hangs and never returns to the runner, the
        runner can't observe the cancel to call `cancel_running` itself,
        so without this the item would stay `running` on disk and crash
        recovery would re-queue it on the next restart — re-running the
        stuck generation forever. Marking it cancelled here means a
        force-close self-heals instead.

        Returns the `CancelResponse` from the generation handler.
        """
        response: CancelResponse = self.generation.cancel_generation()
        self.queue.cancel_running_item()
        return response

    def _validate_queue_payload(self, payload: QueuePayload) -> None:
        """Boundary validation for enqueue / pending-edit.

        Video requests are validated against the same spec used inside
        `VideoGenerationHandler.generate` (so a request accepted by the
        queue is guaranteed to pass the generate-time check too, modulo
        the api/local decision which depends on live settings).
        Image requests are already pydantic-constrained (width/height/
        steps/numImages bounds) and the image handler clamps values
        itself, so there's no extra validator to run here.
        """
        if payload.kind == "video":
            use_api_specs = should_video_generate_with_ltx_api(
                force_api_generations=self.config.force_api_generations,
                settings=self.state.app_settings,
            )
            error = validate_generate_video_request(
                payload.request, use_api_specs=use_api_specs
            )
            if error is not None:
                raise ValueError(error)
        elif payload.kind == "lora":
            # A standard LoRA wraps a GenerateVideoRequest; validate it with the
            # local-pipeline spec (LoRA inference never runs against the API).
            # union_control / video_input_ic_lora requests are validated inside
            # the IC-LoRA handler.
            req = payload.request
            if isinstance(req, LoraStandardGenerateRequest):
                if should_video_generate_with_ltx_api(
                    force_api_generations=self.config.force_api_generations,
                    settings=self.state.app_settings,
                ):
                    raise ValueError("LoRA inference is not available for API generations")
                error = validate_generate_video_request(req.request, use_api_specs=False)
                if error is not None:
                    raise ValueError(error)

    def _dispatch_queue_payload(self, payload: QueuePayload) -> QueueRunResult:
        """Run one queued payload against the right generation handler
        and normalize the outcome into a `QueueRunResult`.

        The "busy" outcome covers the single-flight race: a non-queue
        surface (retake / IC-LoRA / editor regen) grabbed the generation
        slot between the runner's `is_slot_free` pre-check and the
        generation handler's internal `start_generation` guard. Both
        the video path (RuntimeError) and the image path (HTTPError 409)
        surface that race; we map either to `busy` so the runner
        re-queues the item without consuming a retry.
        """
        try:
            if payload.kind == "video":
                resp = self.video_generation.generate(payload.request)
                if isinstance(resp, GenerateVideoCompleteResponse):
                    return QueueRunResult(
                        status="complete", output_path=resp.video_path
                    )
                return QueueRunResult(status="cancelled")
            if payload.kind == "lora":
                resp = self.lora_inference.generate(payload.request)
                if isinstance(resp, LoraGenerateCompleteResponse):
                    return QueueRunResult(
                        status="complete", output_path=resp.videoPath
                    )
                return QueueRunResult(status="cancelled")
            if payload.kind == "image_edit":
                # FLUX.2 [klein] 9B local edit. Same single-flight GPU slot as
                # the other generators, so the "already in progress" race maps
                # to `busy` via the shared HTTPError(409) handler below.
                resp = self.image_edit.generate(payload.request)
                if isinstance(resp, GenerateImageEditCompleteResponse):
                    first = resp.image_paths[0] if resp.image_paths else ""
                    return QueueRunResult(status="complete", output_path=first)
                return QueueRunResult(status="cancelled")
            resp = self.image_generation.generate(payload.request)
            if isinstance(resp, GenerateImageCompleteResponse):
                first = resp.image_paths[0] if resp.image_paths else ""
                return QueueRunResult(status="complete", output_path=first)
            return QueueRunResult(status="cancelled")
        except HTTPError as exc:
            if exc.status_code == 409 and "already in progress" in exc.detail.lower():
                return QueueRunResult(status="busy")
            return QueueRunResult(status="failed", error=exc.detail)
        except RuntimeError as exc:
            if "already in progress" in str(exc).lower():
                return QueueRunResult(status="busy")
            return QueueRunResult(status="failed", error=str(exc))
        except Exception as exc:
            return QueueRunResult(status="failed", error=str(exc))


@dataclass
class ServiceBundle:
    http: HTTPClient
    gpu_cleaner: GpuCleaner
    model_downloader: ModelDownloader
    gpu_info: GpuInfo
    video_processor: VideoProcessor
    text_encoder: TextEncoder
    task_runner: TaskRunner
    ltx_api_client: LTXAPIClient
    zit_api_client: ZitAPIClient
    fast_video_pipeline_class: type[FastVideoPipeline]
    image_generation_pipeline_class: type[ImageGenerationPipeline]
    ic_lora_pipeline_class: type[IcLoraPipeline]
    depth_processor_pipeline_class: type[DepthProcessorPipeline]
    pose_processor_pipeline_class: type[PoseProcessorPipeline]
    a2v_pipeline_class: type[A2VPipeline]
    retake_pipeline_class: type[RetakePipeline]
    image_edit_pipeline_class: type[ImageEditPipeline] | None
    trainer_target: TrainerTarget
    video_captioner: VideoCaptioner
    clip_processor: ClipProcessor
    image_editor: ImageEditor
    video_restyler: VideoRestyler
    pexels_client: PexelsClient
    lora_prompt_profiler: LoraPromptProfiler = field(default_factory=NullLoraPromptProfiler)
    # The same local (WSL2) target wired into `trainer_target`'s routing map,
    # exposed separately so the handler can run its read-only eligibility
    # probe without reaching through the router. Defaulted so existing test
    # bundles (which don't exercise local training) construct unchanged; the
    # default is side-effect-safe (no work until probed).
    local_trainer: LocalTrainerTarget = field(default_factory=LocalTrainerTarget)


def build_default_service_bundle(config: RuntimeConfig) -> ServiceBundle:
    """Build real runtime services with lazy heavy imports isolated from tests."""
    from services.fast_video_pipeline.ltx_fast_video_pipeline import LTXFastVideoPipeline
    from services.zit_api_client.zit_api_client_impl import ZitAPIClientImpl
    from services.gpu_cleaner.torch_cleaner import TorchCleaner
    from services.gpu_info.gpu_info_impl import GpuInfoImpl
    from services.http_client.http_client_impl import HTTPClientImpl
    from services.a2v_pipeline.ltx_a2v_pipeline import LTXa2vPipeline
    from services.depth_processor_pipeline.midas_dpt_pipeline import MidasDPTPipeline
    from services.ic_lora_pipeline.ltx_ic_lora_pipeline import LTXIcLoraPipeline
    from services.image_generation_pipeline.zit_image_generation_pipeline import ZitImageGenerationPipeline
    from services.image_generation_pipeline.klein_image_edit_pipeline import KleinImageEditPipeline
    from services.ltx_api_client.ltx_api_client_impl import LTXAPIClientImpl
    from services.model_downloader.hugging_face_downloader import HuggingFaceDownloader
    from services.retake_pipeline.ltx_retake_pipeline import LTXRetakePipeline
    from services.pose_processor_pipeline.dw_pose_pipeline import DWPosePipeline
    from services.task_runner.threading_runner import ThreadingRunner
    from services.text_encoder.ltx_text_encoder import LTXTextEncoder
    from services.clip_processor.ffmpeg_clip_processor import FfmpegClipProcessor
    from services.image_editor.fal_image_editor import FalImageEditor
    from services.trainer_target.runpod_trainer_target import RunPodTrainerTarget
    from services.trainer_target.routing_trainer_target import RoutingTrainerTarget
    from services.video_captioner.gemini_video_captioner import GeminiVideoCaptioner
    from services.video_processor.video_processor_impl import VideoProcessorImpl
    from services.video_restyler.fal_video_restyler import FalVideoRestyler
    from services.pexels_client.pexels_client_impl import PexelsClientImpl

    http = HTTPClientImpl()

    # Build the local (WSL2) target once and share the SAME instance between
    # the routing map (which drives training) and the bundle's `local_trainer`
    # field (which the handler uses for its eligibility probe).
    local_trainer = LocalTrainerTarget()

    return ServiceBundle(
        http=http,
        gpu_cleaner=TorchCleaner(device=config.device),
        model_downloader=HuggingFaceDownloader(),
        gpu_info=GpuInfoImpl(),
        video_processor=VideoProcessorImpl(),
        text_encoder=LTXTextEncoder(
            device=config.device,
            http=http,
            ltx_api_base_url=config.ltx_api_base_url,
        ),
        task_runner=ThreadingRunner(),
        ltx_api_client=LTXAPIClientImpl(http=http, ltx_api_base_url=config.ltx_api_base_url),
        zit_api_client=ZitAPIClientImpl(http=http),
        fast_video_pipeline_class=LTXFastVideoPipeline,
        image_generation_pipeline_class=ZitImageGenerationPipeline,
        ic_lora_pipeline_class=LTXIcLoraPipeline,
        depth_processor_pipeline_class=MidasDPTPipeline,
        pose_processor_pipeline_class=DWPosePipeline,
        a2v_pipeline_class=LTXa2vPipeline,
        retake_pipeline_class=LTXRetakePipeline,
        image_edit_pipeline_class=KleinImageEditPipeline,
        trainer_target=RoutingTrainerTarget(
            {
                "runpod": RunPodTrainerTarget(
                    ssh_key_dir=config.app_data_dir / "lora" / "ssh"
                ),
                "local": local_trainer,
            }
        ),
        video_captioner=GeminiVideoCaptioner(http=http),
        clip_processor=FfmpegClipProcessor(),
        image_editor=FalImageEditor(http=http),
        video_restyler=FalVideoRestyler(http=http),
        pexels_client=PexelsClientImpl(http=http),
        lora_prompt_profiler=GeminiLoraPromptProfiler(http=http),
        local_trainer=local_trainer,
    )


def build_initial_state(
    config: RuntimeConfig,
    default_settings: AppSettings,
    service_bundle: ServiceBundle | None = None,
) -> AppHandler:
    bundle = service_bundle or build_default_service_bundle(config)

    return AppHandler(
        config=config,
        default_settings=default_settings,
        http=bundle.http,
        gpu_cleaner=bundle.gpu_cleaner,
        model_downloader=bundle.model_downloader,
        gpu_info=bundle.gpu_info,
        video_processor=bundle.video_processor,
        text_encoder=bundle.text_encoder,
        task_runner=bundle.task_runner,
        ltx_api_client=bundle.ltx_api_client,
        zit_api_client=bundle.zit_api_client,
        fast_video_pipeline_class=bundle.fast_video_pipeline_class,
        image_generation_pipeline_class=bundle.image_generation_pipeline_class,
        ic_lora_pipeline_class=bundle.ic_lora_pipeline_class,
        depth_processor_pipeline_class=bundle.depth_processor_pipeline_class,
        pose_processor_pipeline_class=bundle.pose_processor_pipeline_class,
        a2v_pipeline_class=bundle.a2v_pipeline_class,
        retake_pipeline_class=bundle.retake_pipeline_class,
        image_edit_pipeline_class=bundle.image_edit_pipeline_class,
        trainer_target=bundle.trainer_target,
        video_captioner=bundle.video_captioner,
        clip_processor=bundle.clip_processor,
        image_editor=bundle.image_editor,
        video_restyler=bundle.video_restyler,
        pexels_client=bundle.pexels_client,
        local_trainer=bundle.local_trainer,
        lora_prompt_profiler=bundle.lora_prompt_profiler,
    )

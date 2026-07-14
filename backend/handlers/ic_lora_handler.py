"""IC-LoRA endpoints orchestration handler."""

from __future__ import annotations

import base64
import logging
import shutil
import time
import uuid
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from api_types import (
    ConditioningType,
    IcLoraExtractRequest,
    IcLoraExtractResponse,
    IcLoraGenerateCancelledResponse,
    IcLoraGenerateCompleteResponse,
    IcLoraGenerateRequest,
    IcLoraGenerateResponse,
    ImageConditioningInput,
)
from _routes._errors import HTTPError
from handlers.base import StateHandlerBase
from handlers.generation_handler import GenerationHandler
from handlers.media_handler import MediaHandler
from handlers.pipelines_handler import PipelinesHandler
from handlers.text_handler import TextHandler
from runtime_config.model_download_specs import (
    DEPTH_PROCESSOR_CP_ID,
    PERSON_DETECTOR_CP_ID,
    POSE_PROCESSOR_CP_ID,
    get_downloaded_ltx_model_id,
    get_existing_cp_path,
    get_ltx_model_spec,
)
from runtime_config.runtime_config import RuntimeConfig
from state.conditioning_cache import ConditioningCacheEntry, ConditioningCacheKey
from services.interfaces import VideoProcessor
from services.services_utils import FrameArray, VideoCaptureLike
from state.app_state_types import AppState, ICLoraState

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)

# IC-LoRA reference standardization + resolution strategy.
#
# The union / video-input IC-LoRA adapters were trained on LTX's canonical
# distribution: 24fps, ~16:9 or ~9:16, durations on the 8k+1 temporal lattice,
# at the 540p bucket (960x576). Feeding an arbitrary imported reference (any
# fps / AR / length) drifts off that distribution and the output distorts, so
# we conform the reference before inference.
#
# STAGE-1 (the diffusion pass that decides identity/structure) ALWAYS runs at
# the 540p bucket — the adapter's training resolution. Higher OUTPUT
# resolutions are reached with the spatial upsampler (Stage 2, x2), exactly
# like the t2v fast pipeline (DistilledPipeline), instead of diffusing natively
# at high res. A native 1080p first-pass is ~4x the pixels of 540p and overflows
# a 32GB GPU: it silently spills into shared system memory (over PCIe) and
# crawls (~1600s/step vs ~3s). Upsampling from a 540p stage-1 keeps VRAM
# bounded and the adapter on-distribution.
#
# The vendored two-stage pipeline always runs stage 1 at (height//2, width//2)
# and (when not skipped) upsamples stage 2 to (height, width). So the handler
# hands it a 2x-of-540p target (1920x1152 / 1152x1920): with the upsampler
# skipped the output is the 540p bucket; with it enabled the output is
# 1920x1152. 720p is downscaled from that 1920x1152 result (the x2 upsampler
# can't hit 720p directly). Bucket dims are 64-multiples so the stage-1 latent
# (bucket // 32) is even; the two-stage patchify (p=2) can't split an odd axis
# (960x544 gave a 17-tall latent that crashed rearrange), which is why 540p is
# 960x576 rather than 960x544.
_IC_LORA_FPS: int = 24
# Stage-1 diffusion bucket — ALWAYS 540p, regardless of the requested output.
_IC_LORA_STAGE1_BUCKET: dict[str, tuple[int, int]] = {
    "16:9": (960, 576),
    "9:16": (576, 960),
}
# Final output size per requested resolution, AR-consistent with the 5:3
# stage-1 bucket. 540p is the raw stage-1; 720p/1080p come from the x2
# upsampler (1920x1152 / 1152x1920), with 720p downscaled to fit.
_IC_LORA_OUTPUT_BY_RESOLUTION: dict[str, dict[str, tuple[int, int]]] = {
    "540p": {"16:9": (960, 576), "9:16": (576, 960)},
    "720p": {"16:9": (1280, 768), "9:16": (768, 1280)},
    "1080p": {"16:9": (1920, 1152), "9:16": (1152, 1920)},
}
_IC_LORA_DEFAULT_RESOLUTION: str = "540p"
# Supported durations (seconds @ 24fps). Capped at 10s for VRAM.
_IC_LORA_DURATIONS: tuple[int, ...] = (5, 6, 8, 10)

# How often the frame-standardization loops poll the cancel flag. The loops are
# handler-owned (unlike the blocking pipeline call) so they CAN be interrupted;
# checking every N frames keeps a Stop responsive without pegging the lock.
_CANCEL_CHECK_EVERY_FRAMES: int = 8


def _even(value: float) -> int:
    """Round to the nearest non-negative even int (mp4v/x264 need even dims)."""
    v = max(0, int(round(value)))
    return v if v % 2 == 0 else v + 1


def _ic_lora_render_plan(
    resolution: str, refine: bool, ar_label: str
) -> tuple[int, int, bool, tuple[int, int] | None]:
    """Map a requested output resolution to a pipeline plan.

    Returns ``(stage1_w, stage1_h, use_upsampler, downscale_to)``:
      * ``stage1_*`` is always the 540p bucket (the adapter's training res), so
        the identity-deciding diffusion pass stays on-distribution and VRAM
        stays bounded.
      * ``use_upsampler`` True runs Stage 2 (x2 spatial upsample, output
        1920x1152); False emits the raw 540p stage-1 (960x576).
      * ``downscale_to`` is a final (w, h) to re-encode the x2 result down to
        (720p), or None to keep the upsampler's native 1920x1152 output.

    720p/1080p never diffuse natively at high res — they upsample a 540p
    stage-1, matching the t2v fast pipeline. ``refine`` forces the upsampler on
    even at 540p (540p -> ~1080p).
    """
    if resolution not in _IC_LORA_OUTPUT_BY_RESOLUTION:
        raise HTTPError(
            400,
            f"Unsupported IC-LoRA resolution: {resolution}. "
            f"Supported: {', '.join(_IC_LORA_OUTPUT_BY_RESOLUTION)}.",
        )
    s1_w, s1_h = _IC_LORA_STAGE1_BUCKET[ar_label]
    use_upsampler = resolution != "540p" or refine
    if not use_upsampler:
        return s1_w, s1_h, False, None
    upsampled = (s1_w * 2, s1_h * 2)
    # `refine` on a 540p request means "upscale it" — keep the x2 output
    # (1920x1152), don't downscale back to the 540p we started from.
    if resolution == "540p":
        return s1_w, s1_h, True, None
    target = _IC_LORA_OUTPUT_BY_RESOLUTION[resolution][ar_label]
    downscale_to = None if target == upsampled else target
    return s1_w, s1_h, True, downscale_to


def _standardized_write_size(
    input_width: int,
    input_height: int,
    gen_width: int,
    gen_height: int,
) -> tuple[int, int]:
    """Downscale size for the standardized reference before the VAE encode.

    The pipeline resizes-to-cover + center-crops the conditioning video to the
    generation target internally, so writing the standardized reference at the
    *native* resolution (e.g. 4K) just burns VAE time on pixels that are
    immediately thrown away — this is the dominant cost of a `video_input`
    IC-LoRA generation.

    We pre-scale (aspect ratio preserved, never upscaling) by the smallest
    factor that still *covers* the generation target in both axes, i.e.
    ``max(gen_w/in_w, gen_h/in_h)`` capped at 1.0. Because the pre-scaled frame
    still covers the target, the pipeline's internal resize stays a downscale,
    so the output framing is identical to feeding native — only wasted
    resolution is removed.

    ``(gen_width, gen_height)`` is the *stage-1 bucket* (e.g. 960x576), which is
    the resolution the reference actually conditions (stage 1), matching the
    ComfyUI removebeard graph that resizes the reference to the bucket's shorter
    side. Targeting the bucket (rather than the 2x generation-latent target) is
    the quality-preserving floor for the conditioning encode.
    """
    if input_width <= 0 or input_height <= 0 or gen_width <= 0 or gen_height <= 0:
        return _even(input_width), _even(input_height)
    cover_scale = max(gen_width / input_width, gen_height / input_height)
    scale = min(cover_scale, 1.0)  # never upscale a small reference
    return _even(input_width * scale), _even(input_height * scale)


class IcLoraHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        generation_handler: GenerationHandler,
        pipelines_handler: PipelinesHandler,
        text_handler: TextHandler,
        video_processor: VideoProcessor,
        media_handler: MediaHandler,
        config: RuntimeConfig,
    ) -> None:
        super().__init__(state, lock, config)
        self._generation = generation_handler
        self._pipelines = pipelines_handler
        self._text = text_handler
        self._video_processor = video_processor
        self._media = media_handler

    def _maybe_mux_audio(
        self, output_path: Path, reference_path: str, preserve_audio: bool
    ) -> Path:
        """Mux the reference's audio onto the generated output when requested.

        Returns the path to use as the generation result — either the muxed
        file (output + reference audio, trimmed to the output length) or, on
        any failure / when the flag is off / when the reference has no audio,
        the original video-only output. Failures are logged and degraded
        gracefully so an audio issue never blocks a successful generation.
        """
        if not preserve_audio:
            return output_path
        try:
            return self._media.mux_reference_audio(str(output_path), reference_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[ic-lora] preserve-audio mux failed, returning video-only: %s", exc
            )
            return output_path

    def _downscale_output(self, output_path: Path, size: tuple[int, int]) -> Path:
        """Re-encode the generated video down to ``size`` (w, h).

        Used for 720p, which is downscaled from the x2 upsampler's 1920x1152
        result (the x2 upsampler can't hit 720p directly). On failure we log and
        keep the upsampled output — a wrong-but-usable size beats a hard fail on
        an otherwise-complete generation.
        """
        try:
            scaled = self._media.downscale_video(str(output_path), size[0], size[1])
            if scaled != output_path:
                output_path.unlink(missing_ok=True)
            return scaled
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[ic-lora] downscale to %dx%d failed, keeping upsampled output: %s",
                size[0], size[1], exc,
            )
            return output_path

    def _build_conditioning_frame(
        self,
        frame: FrameArray,
        conditioning_type: ConditioningType,
        ic_state: ICLoraState | None = None,
    ) -> FrameArray:
        match conditioning_type:
            case "canny":
                return self._video_processor.apply_canny(frame)
            case "depth":
                if ic_state is None:
                    raise HTTPError(500, "Depth conditioning requires loaded IC-LoRA resources")
                return self._video_processor.apply_depth(frame, ic_state.depth_pipeline)
            case "pose":
                if ic_state is None or ic_state.pose_resources is None:
                    raise HTTPError(500, "Pose conditioning requires loaded IC-LoRA pose resources")
                return self._video_processor.apply_pose(frame, ic_state.pose_resources.pipeline)
            case _:
                raise HTTPError(400, f"Unsupported conditioning_type: {conditioning_type}")

    def _require_ic_lora_model_paths(
        self, conditioning_type: ConditioningType
    ) -> tuple[Path, Path, Path | None, Path | None]:
        """Resolve (lora_path, depth_model_path, pose_model_path, person_detector_model_path).

        Depth processor is always loaded (the union IC-LoRA pipeline pairs with it
        for cache identity); pose resources are loaded only for pose conditioning.
        """
        model_id = get_downloaded_ltx_model_id(self.models_dir)
        if model_id is None:
            raise HTTPError(409, "NO_DOWNLOADED_LTX_MODEL")
        ic_loras_spec = get_ltx_model_spec(model_id).ic_loras_spec
        match conditioning_type:
            case "canny":
                lora_cp_id = ic_loras_spec.canny_cp
            case "depth":
                lora_cp_id = ic_loras_spec.depth_cp
            case "pose":
                lora_cp_id = ic_loras_spec.pose_cp
            case _:
                raise HTTPError(400, f"Unsupported conditioning_type: {conditioning_type}")
        lora_path = get_existing_cp_path(self.models_dir, lora_cp_id)
        depth_model_path = get_existing_cp_path(self.models_dir, DEPTH_PROCESSOR_CP_ID)
        pose_model_path: Path | None = None
        person_detector_model_path: Path | None = None
        if conditioning_type == "pose":
            pose_model_path = get_existing_cp_path(self.models_dir, POSE_PROCESSOR_CP_ID)
            person_detector_model_path = get_existing_cp_path(self.models_dir, PERSON_DETECTOR_CP_ID)
        return lora_path, depth_model_path, pose_model_path, person_detector_model_path

    def extract_conditioning(self, req: IcLoraExtractRequest) -> IcLoraExtractResponse:
        video_file = Path(req.video_path)
        if not video_file.exists():
            raise HTTPError(400, f"Video not found: {req.video_path}")

        cap = self._video_processor.open_video(str(video_file))
        info = self._video_processor.get_video_info(cap)
        target_frame = int(req.frame_time * float(info["fps"]))
        frame = self._video_processor.read_frame(cap, frame_idx=target_frame)
        self._video_processor.release(cap)

        if frame is None:
            raise HTTPError(400, "Could not read frame from video")

        ic_state: ICLoraState | None = None
        if req.conditioning_type in ("depth", "pose"):
            lora_path, depth_model_path, pose_model_path, person_detector_model_path = (
                self._require_ic_lora_model_paths(req.conditioning_type)
            )
            ic_state = self._pipelines.load_ic_lora(
                str(lora_path),
                str(depth_model_path),
                pose_model_path=str(pose_model_path) if pose_model_path else None,
                person_detector_model_path=str(person_detector_model_path)
                if person_detector_model_path
                else None,
            )

        result = self._build_conditioning_frame(frame, req.conditioning_type, ic_state)

        conditioning = self._video_processor.encode_frame_jpeg(result, quality=85)
        original = self._video_processor.encode_frame_jpeg(frame, quality=85)

        return IcLoraExtractResponse(
            conditioning="data:image/jpeg;base64," + base64.b64encode(conditioning).decode("utf-8"),
            original="data:image/jpeg;base64," + base64.b64encode(original).decode("utf-8"),
            conditioning_type=req.conditioning_type,
            frame_time=req.frame_time,
        )

    def _resolve_seed(self) -> int:
        settings = self.state.app_settings
        if settings.seed_locked:
            return settings.locked_seed
        if self.config.dev_mode:
            return 1000
        return int(time.time()) % 2147483647

    def _snap_duration(self, src_duration: float) -> int:
        """Snap a source duration (seconds) to the nearest supported IC-LoRA duration."""
        return min(_IC_LORA_DURATIONS, key=lambda d: abs(d - src_duration))

    def _canonical_shape(
        self,
        input_width: int,
        input_height: int,
        src_fps: float,
        src_frame_count: int,
        target_duration: int | None = None,
    ) -> tuple[int, int, int, float]:
        """Conform a reference video to the model's training distribution.

        Returns (stage1_w, stage1_h, num_frames, fps) where:
          * (stage1_w, stage1_h) is always the 540p bucket for the input's AR
            (~16:9 -> 960x576, ~9:16 -> 576x960). This is the stage-1 diffusion
            resolution — the adapter's training res — regardless of the
            requested OUTPUT resolution, which is reached via the upsampler.
            See ``_ic_lora_render_plan``.
          * fps is fixed at 24 (the training fps).
          * duration is snapped to a supported value, capped at 10s — unless
            ``target_duration`` overrides it (validated against the supported
            set), which is how the Gen Space UI exposes length control.
          * num_frames is on the 8k+1 VAE temporal lattice.
        """
        ar_label = "16:9" if input_width >= input_height else "9:16"
        width, height = _IC_LORA_STAGE1_BUCKET[ar_label]
        safe_fps = src_fps if src_fps > 0 else float(_IC_LORA_FPS)
        if target_duration is not None:
            if target_duration not in _IC_LORA_DURATIONS:
                raise HTTPError(
                    400,
                    f"Unsupported IC-LoRA duration: {target_duration}. "
                    f"Supported: {', '.join(str(d) for d in _IC_LORA_DURATIONS)}s.",
                )
            duration = target_duration
        else:
            src_duration = src_frame_count / safe_fps
            duration = self._snap_duration(src_duration)
        num_frames = ((duration * _IC_LORA_FPS) // 8) * 8 + 1
        return width, height, num_frames, float(_IC_LORA_FPS)

    def _iter_resampled_frames(
        self,
        cap: VideoCaptureLike,
        src_fps: float,
        src_frame_count: int,
        num_frames: int,
    ) -> Iterator[FrameArray]:
        """Yield exactly ``num_frames`` frames resampled to 24fps motion.

        Frames are picked by nearest-index mapping ``src_idx = round(i * src_fps / 24)``.
        If the source is shorter than the target duration, the last readable
        frame is repeated (freeze-frame padding) so the conditioning tensor
        always matches the output temporal grid.
        """
        last_frame: FrameArray | None = None
        for i in range(num_frames):
            src_idx = round(i * src_fps / _IC_LORA_FPS) if src_fps > 0 else i
            frame: FrameArray | None = None
            if src_idx < src_frame_count:
                frame = self._video_processor.read_frame(cap, frame_idx=src_idx)
            if frame is None:
                if last_frame is None:
                    raise HTTPError(400, "Could not read any frames from reference video")
                frame = last_frame
            else:
                last_frame = frame
            yield frame

    def generate(self, req: IcLoraGenerateRequest) -> IcLoraGenerateResponse:
        if self._generation.is_generation_running():
            raise HTTPError(409, "Generation already in progress")

        # Clear any stale pre-load cancel token from a prior run before this
        # attempt begins. See GenerationHandler.clear_cancel_token.
        self._generation.clear_cancel_token()

        video_path = Path(req.video_path)
        if not video_path.exists():
            raise HTTPError(400, f"Video not found: {req.video_path}")
        lora_path, depth_model_path, pose_model_path, person_detector_model_path = (
            self._require_ic_lora_model_paths(req.conditioning_type)
        )

        generation_id = uuid.uuid4().hex[:8]
        t_total_start = time.perf_counter()
        logger.info("[ic-lora] Generation started (conditioning=%s)", req.conditioning_type)

        try:
            t_load_start = time.perf_counter()
            ic_state = self._pipelines.load_ic_lora(
                str(lora_path),
                str(depth_model_path),
                pose_model_path=str(pose_model_path) if pose_model_path else None,
                person_detector_model_path=str(person_detector_model_path)
                if person_detector_model_path
                else None,
            )
            t_load_end = time.perf_counter()
            logger.info("[ic-lora] Pipeline load: %.2fs", t_load_end - t_load_start)

            self._generation.start_generation(generation_id)
            self._generation.update_progress("loading_model", 5, 0, 1)

            s = self.state.app_settings
            use_api = not self._text.should_use_local_encoding()
            encoding_method = "api" if use_api else "local"
            t_text_start = time.perf_counter()
            self._text.prepare_text_encoding(req.prompt, enhance_prompt=use_api and s.prompt_enhancer_enabled_t2v)
            t_text_end = time.perf_counter()
            logger.info("[ic-lora] Text encoding (%s): %.2fs", encoding_method, t_text_end - t_text_start)

            cap = self._video_processor.open_video(str(video_path))
            if not cap.isOpened():
                raise HTTPError(400, f"Cannot open video: {video_path}")
            info = self._video_processor.get_video_info(cap)
            input_width = int(info["width"])
            input_height = int(info["height"])
            src_frame_count = int(info["frame_count"])
            src_fps = float(info["fps"])

            # Conform the reference to the model's training distribution
            # (24fps, AR-bucketed, 8k+1 frames). (width, height) is the 540p
            # stage-1 diffusion bucket; the requested output resolution is
            # reached via the upsampler (Stage 2) + optional downscale, so
            # 720p/1080p never diffuse natively at high res. See
            # `_ic_lora_render_plan`.
            width, height, num_frames, fps = self._canonical_shape(
                input_width, input_height, src_fps, src_frame_count, req.duration
            )
            ar_label = "16:9" if input_width >= input_height else "9:16"
            _s1_w, _s1_h, use_upsampler, downscale_to = _ic_lora_render_plan(
                req.resolution, req.refine, ar_label
            )

            source_stat = video_path.stat()
            cache_key = ConditioningCacheKey(
                video_path=str(video_path.resolve()),
                conditioning_type=req.conditioning_type,
                source_mtime_ns=source_stat.st_mtime_ns,
                source_size=source_stat.st_size,
                width=width,
                height=height,
                frame_count=num_frames,
                fps=fps,
            )
            cached = ic_state.conditioning_cache.get(cache_key)

            t_preprocess_start = 0.0
            t_preprocess_end = 0.0

            if cached is not None:
                self._video_processor.release(cap)
                control_video_path = cached.control_video_path
                frame_count = cached.frame_count
                fps = cached.fps
                logger.info("[ic-lora] Conditioning cache hit for %s/%s", video_path.name, req.conditioning_type)
            else:
                if src_frame_count <= 0:
                    self._video_processor.release(cap)
                    raise HTTPError(400, f"Reference video has no frames: {video_path}")
                t_preprocess_start = time.perf_counter()

                control_video_path = str(
                    self.config.outputs_dir / f"_control_{req.conditioning_type}_{uuid.uuid4().hex[:8]}.mp4"
                )
                write_size = _standardized_write_size(
                    input_width, input_height, width, height
                )
                needs_downscale = write_size != (input_width, input_height)
                writer = self._video_processor.create_writer(
                    control_video_path,
                    fourcc="mp4v",
                    fps=fps,
                    size=write_size,
                )

                # Build the control signal from the *standardized* frame stream
                # so canny/depth/pose land on the same 24fps / num_frames grid as
                # the output latent (the old path read native-fps frames and fed
                # the raw frame_count, which misaligned the temporal compression).
                # Per-frame depth/pose inference is slow, so poll the cancel flag
                # here (this loop is interruptible; the pipeline call is not).
                preprocess_failed = False
                try:
                    for frame_index, frame in enumerate(
                        self._iter_resampled_frames(
                            cap, src_fps, src_frame_count, num_frames
                        )
                    ):
                        if (
                            frame_index % _CANCEL_CHECK_EVERY_FRAMES == 0
                            and self._generation.is_generation_cancelled()
                        ):
                            raise RuntimeError("Generation was cancelled")
                        if needs_downscale:
                            frame = self._video_processor.resize_frame(
                                frame, write_size
                            )
                        control_frame = self._build_conditioning_frame(
                            frame, req.conditioning_type, ic_state
                        )
                        writer.write(control_frame)
                except Exception:
                    preprocess_failed = True
                    raise
                finally:
                    self._video_processor.release(cap)
                    self._video_processor.release(writer)
                    if preprocess_failed:
                        Path(control_video_path).unlink(missing_ok=True)
                t_preprocess_end = time.perf_counter()
                logger.info(
                    "[ic-lora] Preprocessing (%s, %d frames @ %dfps, %dx%d -> %dx%d): %.2fs",
                    req.conditioning_type, num_frames, int(fps),
                    input_width, input_height, write_size[0], write_size[1],
                    t_preprocess_end - t_preprocess_start,
                )

                ic_state.conditioning_cache.put(
                    cache_key, ConditioningCacheEntry(control_video_path, num_frames, fps)
                )
                frame_count = num_frames

            images: list[ImageConditioningInput] = [
                ImageConditioningInput(path=img.path, frame_idx=int(img.frame), strength=float(img.strength))
                for img in req.images
            ]

            self._generation.update_progress("inference", 15, 0, 1)

            # Honor a cancel that landed during load / text / preprocess.
            # Without this check (and the post-inference one below), the
            # IC-LoRA path ignores the cancel flag entirely, runs the
            # pipeline to completion, and returns `complete` — so the queue
            # marks the item completed and the output appears despite the
            # user clicking Stop. See video/image/retake handlers for the
            # same cancel-poll pattern.
            if self._generation.is_generation_cancelled():
                raise RuntimeError("Generation was cancelled")

            output_path = (
                self.config.outputs_dir / f"ic_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.mp4"
            )

            # Stage-1 always diffuses at the 540p bucket (width, height); the
            # upsampler (skip_stage_2=False) lifts the output to 1920x1152, and
            # 720p is downscaled from that below. See `_ic_lora_render_plan`.
            t_inference_start = time.perf_counter()
            logger.info(
                "[ic-lora] Pipeline generate: stage-1 %dx%d, upsampler=%s -> %dx%d%s",
                width, height, use_upsampler,
                width * 2 if use_upsampler else width,
                height * 2 if use_upsampler else height,
                f", downscale->{downscale_to[0]}x{downscale_to[1]}" if downscale_to else "",
            )
            ic_state.pipeline.generate(
                prompt=req.prompt,
                seed=self._resolve_seed(),
                height=height * 2,
                width=width * 2,
                num_frames=frame_count,
                frame_rate=fps,
                images=images,
                video_conditioning=[(control_video_path, req.conditioning_strength)],
                output_path=str(output_path),
                skip_stage_2=not use_upsampler,
            )
            t_inference_end = time.perf_counter()
            logger.info("[ic-lora] Inference: %.2fs", t_inference_end - t_inference_start)

            # The blocking pipeline call can't be interrupted mid-step, so
            # a cancel that arrived during inference is only observable now.
            # Drop the half-finished output and route through the cancelled
            # response so the runner calls `cancel_running` instead of
            # `complete_running`.
            if self._generation.is_generation_cancelled():
                if output_path.exists():
                    output_path.unlink(missing_ok=True)
                raise RuntimeError("Generation was cancelled")

            if downscale_to is not None:
                output_path = self._downscale_output(output_path, downscale_to)

            # Resolve the path the queue will actually receive before naming
            # the control sibling. Preserve-audio may turn `output.mp4` into
            # `output_audio.mp4`; deriving the sibling from the pre-mux path
            # made the frontend look for a control file that did not exist.
            final_path = self._maybe_mux_audio(
                output_path, req.video_path, req.preserve_audio
            )

            # Copy the control video (canny/depth/pose) to a stable path next to
            # the final output so the frontend can surface it as a "control"
            # view. The working control is written by OpenCV as `mp4v`, which
            # Chromium does not reliably decode, so transcode the preview to
            # H.264/yuv420p. Keep a raw-copy fallback so preview encoding can
            # never fail an otherwise-successful generation.
            control_sibling = final_path.with_name(f"{final_path.stem}_control.mp4")
            try:
                self._media.transcode_video_for_browser(
                    control_video_path, str(control_sibling)
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[ic-lora] Could not transcode control preview; "
                    "falling back to raw copy: %s",
                    exc,
                )
                try:
                    shutil.copyfile(control_video_path, control_sibling)
                except Exception as copy_exc:  # noqa: BLE001
                    logger.warning(
                        "[ic-lora] Could not copy control video beside output: %s",
                        copy_exc,
                    )

            t_total_end = time.perf_counter()
            preprocess_time = (t_preprocess_end - t_preprocess_start) if cached is None else 0.0
            logger.info(
                "[ic-lora] Total generation: %.2fs (load=%.2fs, text=%.2fs, preprocess=%.2fs, inference=%.2fs)",
                t_total_end - t_total_start,
                t_load_end - t_load_start,
                t_text_end - t_text_start,
                preprocess_time,
                t_inference_end - t_inference_start,
            )

            self._generation.update_progress("complete", 100, 1, 1)
            self._generation.complete_generation(str(final_path))
            return IcLoraGenerateCompleteResponse(status="complete", video_path=str(final_path))

        except HTTPError:
            self._generation.fail_generation("IC-LoRA generation failed")
            raise
        except Exception as exc:
            self._generation.fail_generation(str(exc))
            if "cancelled" in str(exc).lower():
                return IcLoraGenerateCancelledResponse(status="cancelled")
            raise HTTPError(500, f"Generation error: {exc}") from exc
        finally:
            self._text.clear_api_embeddings()

    def generate_video_input(
        self,
        *,
        lora_path: str,
        lora_scale: float,
        prompt: str,
        video_path: str,
        conditioning_strength: float,
        negative_prompt: str,
        target_duration: int | None = None,
        preserve_audio: bool = False,
        refine: bool = False,
        resolution: str = _IC_LORA_DEFAULT_RESOLUTION,
    ) -> IcLoraGenerateResponse:
        """Run a user-trained IC-LoRA conditioned on a reference video.

        The reference is first standardized to the model's training distribution
        (24fps, AR-bucketed to ~16:9 / ~9:16, duration snapped to a supported
        value on the 8k+1 temporal lattice) before being fed to the pipeline's
        ``video_conditioning`` channel. Without this step an arbitrary imported
        reference (any fps / AR / length) drifts off-distribution and the
        adapter distorts instead of applying. The adapter path is supplied by
        the caller (resolved from the inference registry); the base distilled
        checkpoint + depth processor are loaded the same way as the union path.
        """
        del negative_prompt  # negative prompt is owned by the pipeline defaults
        if self._generation.is_generation_running():
            raise HTTPError(409, "Generation already in progress")

        # Clear any stale pre-load cancel token from a prior run before this
        # attempt begins. See GenerationHandler.clear_cancel_token.
        self._generation.clear_cancel_token()

        ref = Path(video_path)
        if not ref.exists():
            raise HTTPError(400, f"Video not found: {video_path}")

        # load_ic_lora always pairs with a depth processor (cache identity); for
        # video_input it's loaded but unused. Resolve it the same way the union
        # path does, reusing the canny branch's model-path resolution.
        _, depth_model_path, _, _ = self._require_ic_lora_model_paths("canny")

        generation_id = uuid.uuid4().hex[:8]
        t_total_start = time.perf_counter()
        logger.info("[ic-lora] video_input generation started (lora=%s)", Path(lora_path).name)

        try:
            t_load_start = time.perf_counter()
            ic_state = self._pipelines.load_ic_lora(
                lora_path,
                str(depth_model_path),
                lora_scale=lora_scale,
            )
            t_load_end = time.perf_counter()
            logger.info("[ic-lora] Pipeline load: %.2fs", t_load_end - t_load_start)

            self._generation.start_generation(generation_id)
            self._generation.update_progress("loading_model", 5, 0, 1)

            s = self.state.app_settings
            use_api = not self._text.should_use_local_encoding()
            encoding_method = "api" if use_api else "local"
            t_text_start = time.perf_counter()
            self._text.prepare_text_encoding(
                prompt, enhance_prompt=use_api and s.prompt_enhancer_enabled_t2v
            )
            t_text_end = time.perf_counter()
            logger.info(
                "[ic-lora] Text encoding (%s): %.2fs",
                encoding_method,
                t_text_end - t_text_start,
            )

            t_standardize_start = time.perf_counter()
            cap = self._video_processor.open_video(str(ref))
            if not cap.isOpened():
                raise HTTPError(400, f"Cannot open video: {ref}")
            try:
                info = self._video_processor.get_video_info(cap)
                input_width = int(info["width"])
                input_height = int(info["height"])
                src_frame_count = int(info["frame_count"])
                src_fps = float(info["fps"])

                if src_frame_count <= 0:
                    raise HTTPError(400, f"Reference video has no frames: {ref}")

                # Conform the reference to the model's training distribution.
                # (width, height) is the 540p stage-1 bucket; the requested
                # output resolution is reached via the upsampler + optional
                # downscale (see `_ic_lora_render_plan`).
                width, height, num_frames, fps = self._canonical_shape(
                    input_width, input_height, src_fps, src_frame_count, target_duration
                )
                ar_label = "16:9" if input_width >= input_height else "9:16"
                _s1_w, _s1_h, use_upsampler, downscale_to = _ic_lora_render_plan(
                    resolution, refine, ar_label
                )

                # Write a standardized reference (24fps, AR-bucketed duration,
                # 8k+1 frames), downscaled toward the *stage-1 bucket* so the
                # pipeline's VAE doesn't encode throwaway pixels — the dominant
                # cost of a video_input generation.
                #
                # The reference is conditioning for stage 1, which runs at the
                # bucket (width, height), NOT the 2x target we hand the pipeline
                # for the generation latent. The ComfyUI removebeard graph proves
                # this: it generates stage 1 at the 960x544 bucket and resizes the
                # reference to shorter-side 544 (bucket) before the IC-LoRA guide
                # encode. So we target the bucket here, not width*2/height*2.
                #
                # Empirical check (see the two logs below): if the vendored
                # pipeline encodes the reference at the *file* resolution, this
                # cuts the encode ~4x. If instead it upscales the reference to
                # cover the passed 2x target, the encode time won't move and the
                # real lever is the passed resolution / a reference-downscale
                # factor — which the next run's timing will tell us.
                write_size = _standardized_write_size(
                    input_width, input_height, width, height
                )
                needs_downscale = write_size != (input_width, input_height)
                source_stat = ref.stat()
                cache_key = ConditioningCacheKey(
                    video_path=str(ref.resolve()),
                    conditioning_type="video_input",
                    source_mtime_ns=source_stat.st_mtime_ns,
                    source_size=source_stat.st_size,
                    width=write_size[0],
                    height=write_size[1],
                    frame_count=num_frames,
                    fps=fps,
                )
                cached_reference = ic_state.conditioning_cache.get(cache_key)
                if cached_reference is not None:
                    standardized_path = cached_reference.control_video_path
                    logger.info(
                        "[ic-lora] Reference cache hit for %s (%dx%d, %d frames)",
                        ref.name,
                        write_size[0],
                        write_size[1],
                        num_frames,
                    )
                else:
                    standardized_path = str(
                        self.config.outputs_dir
                        / f"_ic_lora_ref_{uuid.uuid4().hex[:8]}.mp4"
                    )
                    writer = self._video_processor.create_writer(
                        standardized_path,
                        fourcc="mp4v",
                        fps=fps,
                        size=write_size,
                    )
                    standardize_failed = False
                    try:
                        for frame_index, frame in enumerate(
                            self._iter_resampled_frames(
                                cap, src_fps, src_frame_count, num_frames
                            )
                        ):
                            # This loop is interruptible (unlike the blocking
                            # pipeline call), so honor a Stop promptly instead of
                            # standardizing every frame of a long reference first.
                            if (
                                frame_index % _CANCEL_CHECK_EVERY_FRAMES == 0
                                and self._generation.is_generation_cancelled()
                            ):
                                raise RuntimeError("Generation was cancelled")
                            if needs_downscale:
                                frame = self._video_processor.resize_frame(
                                    frame, write_size
                                )
                            writer.write(frame)
                    except Exception:
                        standardize_failed = True
                        raise
                    finally:
                        self._video_processor.release(writer)
                        if standardize_failed:
                            Path(standardized_path).unlink(missing_ok=True)
                    ic_state.conditioning_cache.put(
                        cache_key,
                        ConditioningCacheEntry(standardized_path, num_frames, fps),
                    )
            finally:
                self._video_processor.release(cap)

            t_standardize_end = time.perf_counter()
            logger.info(
                "[ic-lora] Reference standardize (%d frames @ %dfps, %dx%d -> %dx%d%s, cache=%s): %.2fs",
                num_frames,
                int(fps),
                input_width,
                input_height,
                write_size[0],
                write_size[1],
                "" if needs_downscale else ", no downscale",
                "hit" if cached_reference is not None else "miss",
                t_standardize_end - t_standardize_start,
            )

            output_path = (
                self.config.outputs_dir
                / f"ic_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.mp4"
            )

            self._generation.update_progress("inference", 15, 0, 1)

            # Honor a cancel that landed during load / text / standardize.
            # See `generate` above for why this pair of checks is required.
            if self._generation.is_generation_cancelled():
                raise RuntimeError("Generation was cancelled")

            # The pipeline call is one blocking span covering three sub-stages we
            # can't time from here (they live in the vendored ltx_pipelines): the
            # reference VAE-encode, then stage-1 diffusion, then decode. The
            # vendor logs a timestamped "Added N video conditioning(s)" when the
            # encode finishes and tqdm bars when diffusion starts, so the split
            # is recoverable from the log: inference-start -> "Added ..." is the
            # reference encode; "Added ..." -> first step bar is pre-diffusion
            # prep; the bars are diffusion.
            logger.info(
                "[ic-lora] Pipeline generate: reference file %dx%d, stage-1 bucket "
                "%dx%d, output %dx%d%s (%d frames, upsampler=%s)…",
                write_size[0],
                write_size[1],
                width,
                height,
                width * 2 if use_upsampler else width,
                height * 2 if use_upsampler else height,
                f", downscale->{downscale_to[0]}x{downscale_to[1]}" if downscale_to else "",
                num_frames,
                use_upsampler,
            )
            t_inference_start = time.perf_counter()
            ic_state.pipeline.generate(
                prompt=prompt,
                seed=self._resolve_seed(),
                height=height * 2,
                width=width * 2,
                num_frames=num_frames,
                frame_rate=fps,
                images=[],
                video_conditioning=[(standardized_path, conditioning_strength)],
                output_path=str(output_path),
                skip_stage_2=not use_upsampler,
            )
            t_inference_end = time.perf_counter()
            logger.info(
                "[ic-lora] video_input pipeline complete: %.2fs "
                "(load=%.2fs, text=%.2fs, standardize=%.2fs, pipeline=%.2fs)",
                t_inference_end - t_total_start,
                t_load_end - t_load_start,
                t_text_end - t_text_start,
                t_standardize_end - t_standardize_start,
                t_inference_end - t_inference_start,
            )

            # A cancel that arrived during the blocking inference call is
            # only visible now. Drop the output and return the cancelled
            # response so the runner cancels the queue item instead of
            # completing it (which would surface the output despite cancel).
            if self._generation.is_generation_cancelled():
                if output_path.exists():
                    output_path.unlink(missing_ok=True)
                raise RuntimeError("Generation was cancelled")

            t_postprocess_start = time.perf_counter()
            if downscale_to is not None:
                output_path = self._downscale_output(output_path, downscale_to)

            final_path = self._maybe_mux_audio(
                output_path, video_path, preserve_audio
            )
            t_total_end = time.perf_counter()
            logger.info(
                "[ic-lora] video_input total: %.2fs "
                "(load=%.2fs, text=%.2fs, standardize=%.2fs, pipeline=%.2fs, "
                "postprocess=%.2fs, reference_cache=%s)",
                t_total_end - t_total_start,
                t_load_end - t_load_start,
                t_text_end - t_text_start,
                t_standardize_end - t_standardize_start,
                t_inference_end - t_inference_start,
                t_total_end - t_postprocess_start,
                "hit" if cached_reference is not None else "miss",
            )
            self._generation.update_progress("complete", 100, 1, 1)
            self._generation.complete_generation(str(final_path))
            return IcLoraGenerateCompleteResponse(status="complete", video_path=str(final_path))

        except HTTPError:
            self._generation.fail_generation("IC-LoRA video_input generation failed")
            raise
        except Exception as exc:
            self._generation.fail_generation(str(exc))
            if "cancelled" in str(exc).lower():
                return IcLoraGenerateCancelledResponse(status="cancelled")
            raise HTTPError(500, f"Generation error: {exc}") from exc
        finally:
            self._text.clear_api_embeddings()

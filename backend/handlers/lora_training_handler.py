"""Durable LoRA-trainer control plane.

Owns three on-disk ledgers under ``APP_DATA_DIR`` — datasets,
preprocessed datasets, and training jobs — and the state-machine
transitions over them. Mirrors `QueueHandler`: atomic JSON persistence
(temp file + ``os.replace`` + fsync), deep-copied snapshots out, crash
recovery on load, and a wakeup event the background reconciler blocks
on.

Split of responsibilities (same as the queue):
- **User-facing CRUD** (create/update/delete dataset, request upload,
  start preprocessing, start training, request cancel) — wrapped 1:1 by
  the HTTP routes.
- **Reconciler-facing transitions** (`mark_dataset_uploaded`,
  `set_preprocess_processing`, `mark_training_completed`, ...) — only
  the `LoraTrainingRunner` calls these; the user API never moves an
  entity into a remote-in-flight state directly.

Crash recovery on load: an entity in a remote-in-flight state that
still carries a `TargetHandle` is kept as-is so the reconciler re-polls
the remote job. One with no handle (process died between "mark
in-flight" and "store handle") is reset to its pending/retryable state.

All heavy/remote work lives in the runner; this handler only validates,
mutates, and persists under the shared app lock.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypeVar

from pydantic import BaseModel

from api_types import CreateLoraDerivationJobRequest, GenerateImageEditRequest, LoraClipProbeApi
from handlers import lora_dataset_prep as LoraDatasetPrep
from handlers import lora_command_builder as paths
from handlers import lora_export, lora_publish
from _routes._errors import HTTPError
from handlers.base import StateHandlerBase, with_state_lock
from services.trainer_target.local_trainer_target import (
    LocalTrainerEligibility,
    LocalTrainerTarget,
)
from services.interfaces import (
    ClipProbeResult,
    ClipProcessor,
    ClipProcessorError,
    EditPlan,
    ImageEditor,
    ImageEditorError,
    NanoBananaModel,
    ScaleSpec,
    SceneSpan,
    TrimSpec,
    VideoCaptioner,
    VideoCaptionerError,
    VideoRestyler,
    VideoRestylerError,
    PexelsClient,
    PexelsError,
    PexelsSearchResult,
)
from services.clip_processor.caption_proxy import build_caption_proxy_if_oversized
from state.app_state_types import AppState, HfAuthenticated
from state.lora_clip_jobs_state import ClipJob, ClipJobKind, ClipJobsState
from state.lora_derivation_jobs_state import DerivationJob, DerivationJobsState
from state.lora_training_state import (
    AutoPipelineSpec,
    GpuStatus,
    LoraClip,
    LoraDataset,
    LoraDatasetsState,
    LoraDatasetType,
    LoraFolder,
    LoraTrainingProfile,
    LoraTrainingProfilesState,
    PendingTraining,
    PreprocessedDataset,
    PreprocessedState,
    TargetHandle,
    TrainerProvider,
    WorkspacePolicy,
    TrainingConfig,
    TrainingJob,
    TrainingPreset,
    TrainingState,
    RunpodSelection,
    SavedModelState,
    SavedModelVolumeMetadata,
    VALIDATION_FEED_MAX_ITEMS,
    ValidationFeedItem,
    CheckpointArtifact,
    ValidationSampleRef,
    BUILTIN_STANDARD_ID,
    BUILTIN_DETAILED_RANK64_ID,
    BUILTIN_LOW_VRAM_ID,
    BUILTIN_LOW_VRAM_INT4_ID,
    default_training_profiles,
    legacy_detailed_rank64_config,
)

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig
    from handlers.image_edit_handler import ImageEditHandler

logger = logging.getLogger(__name__)

_MODEL = TypeVar("_MODEL", bound=BaseModel)

_RESOLUTION_BUCKET_RE = re.compile(r"^(\d+)x(\d+)x(\d+)$")

# Reconciler transient-failure budget: how many consecutive retryable failures
# an entity (dataset upload / preprocess / training job) tolerates before the
# handler escalates to a terminal `failed`/`upload_failed`. Picked to absorb
# transient SSH/RunPod blips without dooming a multi-hour training run, while
# still surfacing a stuck entity in minutes (one tick ~= the poll interval).
_TRANSIENT_FAILURE_BUDGET: int = 5

ReconcileEntityKind = Literal["dataset", "preprocess", "training"]

# Kling O3 video-to-video edit rejects sources longer than 10.05s. Trim to a
# hair under that so the uploaded clip is always accepted.
_KLING_O3_MAX_DURATION_SECONDS: float = 10.0
# Kling O3 also rejects clips wider than 2160px (`video_too_large`). Downscale
# over-wide sources to fit, preserving aspect ratio.
_KLING_O3_MAX_WIDTH: int = 2160


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_NANO_BANANA_MODELS: frozenset[str] = frozenset(
    {"nano-banana", "nano-banana-2", "nano-banana-pro"}
)


def _coerce_nano_banana(value: str) -> NanoBananaModel:
    """Map a stored setting string to a valid tier, defaulting safely."""
    if value in _NANO_BANANA_MODELS:
        return value  # type: ignore[return-value]
    return "nano-banana-2"


class ContentAnchor(NamedTuple):
    """Result of `prepare_content_anchor`.

    `source_frame_path` is the verbatim still fed to the editor (the true
    "before"); `anchor_path` is what drives the motion stage — the same
    source when no edit ran, or the Nano-Banana-edited PNG otherwise.
    """

    source_frame_path: str
    anchor_path: str


class LoraEntityNotFoundError(LookupError):
    """Raised when a dataset / preprocessed / training id doesn't exist."""


class LoraTransitionError(ValueError):
    """Raised when a state transition or input is rejected."""


def validate_resolution_buckets(value: str) -> None:
    """Enforce the trainer's VAE constraints on a "WxHxF[;WxHxF...]" string.

    Spatial dims must be multiples of 32 and frame counts must satisfy
    ``frames % 8 == 1`` (LTX-2 VAE downsampling). Raises
    `LoraTransitionError` with a user-readable message on any violation.
    """
    buckets = [b for b in value.split(";") if b.strip()]
    if not buckets:
        raise LoraTransitionError("resolutionBuckets must not be empty")
    for bucket in buckets:
        match = _RESOLUTION_BUCKET_RE.match(bucket.strip())
        if match is None:
            raise LoraTransitionError(
                f"Invalid resolution bucket {bucket!r}; expected WxHxF "
                "(e.g. 768x448x49)"
            )
        width, height, frames = (int(g) for g in match.groups())
        if width % 32 != 0 or height % 32 != 0:
            raise LoraTransitionError(
                f"Bucket {bucket!r}: width and height must be multiples of 32"
            )
        if frames % 8 != 1:
            raise LoraTransitionError(
                f"Bucket {bucket!r}: frames must satisfy frames % 8 == 1 "
                "(e.g. 1, 9, 17, 25, ..., 89)"
            )


class LoraTrainingHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        config: "RuntimeConfig",
        video_captioner: VideoCaptioner,
        clip_processor: ClipProcessor,
        image_editor: ImageEditor,
        video_restyler: VideoRestyler,
        pexels_client: PexelsClient,
        local_trainer: LocalTrainerTarget,
        image_edit_handler: "ImageEditHandler | None" = None,
    ) -> None:
        super().__init__(state, lock, config)
        self._captioner = video_captioner
        self._clip_processor = clip_processor
        self._image_editor = image_editor
        self._video_restyler = video_restyler
        self._pexels = pexels_client
        # Local FLUX.2 [klein] 9B image editor (optional): when a frame edit
        # requests `engine="klein"`, the edit is delegated here so it reuses
        # the single-flight GPU slot + cancel/progress machinery instead of
        # reaching around it. None in builds without the Klein pipeline.
        self._image_edit = image_edit_handler
        # The local (WSL2) backend, used here only for its read-only
        # eligibility probe; the runner drives the actual training through
        # the routing target.
        self._local_trainer = local_trainer
        base = config.app_data_dir
        # Rendered, edited/scene-split derivatives. Lives outside the
        # ledger JSONs so a GC pass can prune orphans without touching state.
        self._derived_dir: Path = base / "lora" / "derived"
        # Curation preview assets (sprites/posters) generated by clip-jobs.
        self._thumbs_dir: Path = base / "lora" / "thumbs"
        # Assets downloaded from the Pexels stock-media browser.
        self._pexels_dir: Path = base / "lora" / "pexels"
        self._datasets_file: Path = base / "lora_datasets.json"
        self._preprocessed_file: Path = base / "lora_preprocessed.json"
        self._training_file: Path = base / "lora_training.json"
        self._saved_models_file: Path = base / "lora_saved_models.json"
        self._profiles_file: Path = base / "lora_training_profiles.json"
        self._clip_jobs_file: Path = base / "lora_clip_jobs.json"
        self._derivation_file: Path = base / "lora_derivation_jobs.json"
        self._datasets = LoraDatasetsState()
        self._preprocessed = PreprocessedState()
        self._training = TrainingState()
        self._saved_models = SavedModelState()
        self._profiles = LoraTrainingProfilesState()
        self._clip_jobs = ClipJobsState()
        self._derivation = DerivationJobsState()
        # Set by mutations that may create reconciler work (request
        # upload, start preprocessing, start training, request cancel).
        self._wakeup_event = threading.Event()
        # Separate wakeup for the (fast, local) clip-jobs runner so its
        # sprite work never waits behind long remote training reconciles.
        self._clip_jobs_wakeup = threading.Event()
        # Wakeup for the target/variant derivation runner (multi-stage AI
        # pipeline: frame edit -> local IC-LoRA drive or remote Kling).
        self._derivation_wakeup = threading.Event()

    @property
    def wakeup_event(self) -> threading.Event:
        return self._wakeup_event

    @property
    def clip_jobs_wakeup_event(self) -> threading.Event:
        return self._clip_jobs_wakeup

    @property
    def derivation_wakeup_event(self) -> threading.Event:
        return self._derivation_wakeup

    # ------------------------------------------------------------------
    # Captioning (desktop-side, no lock held during the network call)
    # ------------------------------------------------------------------

    def caption_clip(self, *, video_path: str, with_audio: bool) -> str:
        """Auto-caption a single local clip via the vision model.

        Reads the Gemini key from settings, then runs the captioner outside
        the lock (it's a slow network call). Raises `VideoCaptionerError` —
        the route maps its `status_code` to the HTTP response.
        """
        api_key = self.state.app_settings.gemini_api_key
        if not api_key:
            raise VideoCaptionerError(
                "Add a Gemini API key in Settings to auto-caption clips.",
                status_code=400,
            )
        logger.info("lora.caption_clip start path=%s with_audio=%s", video_path, with_audio)
        # Gemini's inline upload caps the request at ~20MB. Rather than reject
        # large clips, transcode a small, caption-only proxy (lower resolution +
        # fps) into a temp file and caption that — the model doesn't need full
        # resolution to describe a clip. Original stays untouched.
        proxy_dir: Path | None = None
        caption_path = video_path
        try:
            proxy = self._caption_proxy_if_oversized(video_path, with_audio=with_audio)
            if proxy is not None:
                proxy_dir, caption_path = proxy
            caption = self._captioner.caption(
                video_path=caption_path,
                api_key=api_key,
                with_audio=with_audio,
            )
        finally:
            if proxy_dir is not None:
                shutil.rmtree(proxy_dir, ignore_errors=True)
        logger.info("lora.caption_clip ok path=%s chars=%d", video_path, len(caption))
        return caption

    def _caption_proxy_if_oversized(
        self, video_path: str, *, with_audio: bool
    ) -> tuple[Path, str] | None:
        """Return ``(temp_dir, proxy_path)`` for an oversized video, else None.

        Thin delegate to the shared `caption_proxy` helper (also used by the
        auto-prompt path) so the proxy logic lives in one place. Caller owns
        ``temp_dir`` cleanup.
        """
        return build_caption_proxy_if_oversized(
            self._clip_processor, video_path, with_audio=with_audio
        )

    @staticmethod
    def _even(value: float) -> int:
        """Round to the nearest positive even int (yuv420p needs even dims)."""
        return max(2, int(round(value / 2)) * 2)

    # ------------------------------------------------------------------
    # Probing (desktop-side, stateless, no lock held)
    # ------------------------------------------------------------------

    def probe_clip(self, *, video_path: str) -> ClipProbeResult:
        """Measure a single local clip (duration / resolution / fps / audio).

        Stateless and lock-free: ffmpeg reads the file directly. The route
        persists the result onto the clip via the dataset edit path, so the
        durable ledger always carries the measured probe. Raises
        `ClipProcessorError` (mapped to HTTP by the route).
        """
        logger.info("lora.probe_clip start path=%s", video_path)
        probe = self._clip_processor.probe(video_path=video_path)
        logger.info(
            "lora.probe_clip ok path=%s %dx%d %.2fs %.2ffps audio=%s",
            video_path,
            probe.width,
            probe.height,
            probe.duration_seconds,
            probe.fps,
            probe.has_audio,
        )
        return probe

    # ------------------------------------------------------------------
    # Local-training eligibility (read-only WSL2 capability probe)
    # ------------------------------------------------------------------

    def local_trainer_eligibility(self) -> LocalTrainerEligibility:
        """Report whether local (WSL2) training is possible on this machine.

        Delegates to the local target's side-effect-safe probe (which never
        raises and runs no state mutation); the route maps the result to its
        API model. Lock-free: the probe only shells out to `wsl.exe`.
        """
        return self._local_trainer.probe_eligibility()

    # ------------------------------------------------------------------
    # Editing (desktop-side, stateless ffmpeg; non-destructive)
    # ------------------------------------------------------------------

    def apply_clip_edits(
        self, *, source_path: str, plan: EditPlan
    ) -> tuple[str, ClipProbeResult]:
        """Render an edited derivative of `source_path` from an `EditPlan`.

        Returns the derived file path and its fresh probe. The original is
        never modified — the frontend keeps `source_path` and re-renders
        from it whenever the edit stack changes. Raises `ClipProcessorError`.
        """
        out_path = self._allocate_derived_path(suffix=".mp4")
        logger.info("lora.apply_clip_edits start source=%s plan=%s", source_path, plan)
        self._clip_processor.render(
            source_path=source_path, plan=plan, out_path=str(out_path)
        )
        probe = self._clip_processor.probe(video_path=str(out_path))
        logger.info(
            "lora.apply_clip_edits ok source=%s out=%s %dx%d %.2fs",
            source_path,
            out_path,
            probe.width,
            probe.height,
            probe.duration_seconds,
        )
        return str(out_path), probe

    def split_scenes(
        self, *, source_path: str, threshold: float
    ) -> list[tuple[str, SceneSpan, ClipProbeResult]]:
        """Detect scene cuts and render each segment to its own clip.

        Returns (rendered_path, span, probe) per segment. A clip with no
        detectable cuts yields a single segment covering the whole source.
        Raises `ClipProcessorError`.
        """
        logger.info(
            "lora.split_scenes start source=%s threshold=%.2f", source_path, threshold
        )
        spans = self._clip_processor.detect_scenes(
            video_path=source_path, threshold=threshold
        )
        logger.info(
            "lora.split_scenes detected source=%s segments=%d", source_path, len(spans)
        )
        results: list[tuple[str, SceneSpan, ClipProbeResult]] = []
        for span in spans:
            out_path = self._allocate_derived_path(suffix=".mp4")
            self._clip_processor.render(
                source_path=source_path,
                plan=EditPlan(
                    trim=TrimSpec(start_seconds=span.start_seconds, end_seconds=span.end_seconds)
                ),
                out_path=str(out_path),
            )
            probe = self._clip_processor.probe(video_path=str(out_path))
            results.append((str(out_path), span, probe))
        return results

    # ------------------------------------------------------------------
    # AI dataset prep (Fal, BYOK; stateless, no lock held)
    # ------------------------------------------------------------------

    def edit_frame(
        self,
        *,
        source_path: str,
        time_seconds: float,
        prompt: str,
        model: NanoBananaModel | None,
        engine: str = "fal",
    ) -> str:
        """Extract a frame, edit it, and save it as a PNG.

        Returns the derived image path. Used to fix/manipulate a still
        (e.g. remove an object) that can then drive image-to-video.

        `engine` selects the editor: "fal" (default) edits with Nano Banana
        (remote; `model` falls back to the saved Nano Banana setting when
        omitted); "klein" edits with the local FLUX.2 [klein] 9B pipeline
        (ignores `model`). Raises `ClipProcessorError` / `ImageEditorError`
        for the Fal path, `HTTPError` for the Klein path.
        """
        logger.info(
            "lora.edit_frame start source=%s t=%.2f engine=%s prompt=%r",
            source_path, time_seconds, engine, prompt[:80],
        )
        frame = self._clip_processor.extract_frame(
            video_path=source_path, time_seconds=time_seconds
        )
        if engine == "klein":
            # Klein edits from a file path: persist the extracted frame, then
            # delegate to the image-edit handler (single-flight GPU slot).
            src_path = self._allocate_derived_path(suffix=".png")
            src_path.write_bytes(frame)
            out_path = self._edit_still_with_klein(prompt, str(src_path))
            logger.info("lora.edit_frame ok source=%s engine=klein out=%s", source_path, out_path)
            return out_path

        api_key = self.state.app_settings.fal_api_key
        chosen = model or _coerce_nano_banana(
            self.state.app_settings.lora_nano_banana_model
        )
        edited = self._image_editor.edit(
            image_bytes=frame, prompt=prompt, model=chosen, api_key=api_key or ""
        )
        out_path = self._allocate_derived_path(suffix=".png")
        out_path.write_bytes(edited)
        logger.info("lora.edit_frame ok source=%s out=%s bytes=%d", source_path, out_path, len(edited))
        return str(out_path)

    def _edit_still_with_klein(self, prompt: str, source_frame_path: str) -> str:
        """Edit a still with local FLUX.2 [klein] 9B, returning a derived PNG path.

        Delegates to `ImageEditHandler` so the edit runs through the
        single-flight GPU slot (`load_klein_to_gpu`) with cancel/progress
        semantics, then copies the Klein output into the LoRA derived dir so
        the LoRA GC pass manages it alongside other derivatives. Raises
        `HTTPError(501)` if Klein isn't wired, `HTTPError(409)` if a
        generation is already running, or the underlying edit error.
        """
        if self._image_edit is None:
            raise HTTPError(
                501,
                "Local FLUX.2 [klein] 9B editing isn't available in this build.",
                code="KLEIN_UNAVAILABLE",
            )
        width, height = self._klein_dimensions_for(source_frame_path)
        resp = self._image_edit.generate(
            GenerateImageEditRequest(
                prompt=prompt,
                width=width,
                height=height,
                numSteps=4,
                numImages=1,
                referenceImages=[source_frame_path],
            )
        )
        if resp.status != "complete" or not resp.image_paths:
            raise ImageEditorError(
                "Klein edit produced no image (cancelled?).", status_code=499
            )
        klein_out = Path(resp.image_paths[0])
        out_path = self._allocate_derived_path(suffix=".png")
        shutil.copyfile(klein_out, out_path)
        return str(out_path)

    @staticmethod
    def _klein_dimensions_for(source_frame_path: str) -> tuple[int, int]:
        """Pick Klein output dimensions that preserve the source frame's aspect
        ratio (Klein accepts arbitrary W×H). The longest side is capped at 1024
        and both sides are rounded to a multiple of 16 (Klein requires it);
        a missing/unreadable frame falls back to 1024×1024. Previously this was
        hardcoded to 1024×1024, which squared off portrait/landscape inputs.
        """
        try:
            from PIL import Image  # local import; PIL is a Klein/diffusers dep

            with Image.open(source_frame_path) as img:
                src_w, src_h = img.size
        except Exception:
            return 1024, 1024
        if src_w <= 0 or src_h <= 0:
            return 1024, 1024
        longest = max(src_w, src_h)
        target_longest = 1024
        scale = target_longest / longest
        w = max(16, round((src_w * scale) / 16) * 16)
        h = max(16, round((src_h * scale) / 16) * 16)
        return w, h

    def prepare_content_anchor(
        self,
        *,
        driver_path: str,
        frame_path: str | None,
        frame_time_seconds: float,
        edit_prompt: str,
        model: NanoBananaModel | None,
        edit_engine: str = "fal",
    ) -> ContentAnchor:
        """Resolve the still that anchors the target's content (stages 1-2).

        - If `frame_path` is given (still entry) it's the base image; else a
          frame is extracted from the driver at `frame_time_seconds`.
        - If `edit_prompt` is non-empty the base image is edited (Nano Banana
          for `edit_engine="fal"`, local FLUX.2 [klein] 9B for "klein");
          otherwise it's used as-is.

        Returns a `ContentAnchor` carrying both the verbatim source still
        (`source_frame_path`, the exact "before") and the resolved anchor
        (`anchor_path`, possibly the edited "after"). Persisting the source
        lets the review UI show a true before/after of the same frame.
        Raises `ClipProcessorError` / `ImageEditorError` (Fal) or `HTTPError`
        (Klein).
        """
        if frame_path is not None:
            base_bytes = Path(frame_path).read_bytes()
            source_frame_path = frame_path
        else:
            base_bytes = self._clip_processor.extract_frame(
                video_path=driver_path, time_seconds=frame_time_seconds
            )
            # Persist the extracted frame so the UI can show it as the "before".
            source_path = self._allocate_derived_path(suffix=".png")
            source_path.write_bytes(base_bytes)
            source_frame_path = str(source_path)

        if not edit_prompt.strip():
            # No edit: the anchor *is* the source frame.
            return ContentAnchor(source_frame_path, source_frame_path)

        if edit_engine == "klein":
            out_path = self._edit_still_with_klein(edit_prompt, source_frame_path)
            logger.info(
                "lora.prepare_content_anchor ok driver=%s engine=klein edited=%s",
                driver_path, out_path,
            )
            return ContentAnchor(source_frame_path, out_path)

        api_key = self.state.app_settings.fal_api_key
        chosen = model or _coerce_nano_banana(self.state.app_settings.lora_nano_banana_model)
        edited = self._image_editor.edit(
            image_bytes=base_bytes, prompt=edit_prompt, model=chosen, api_key=api_key or ""
        )
        out_path = self._allocate_derived_path(suffix=".png")
        out_path.write_bytes(edited)
        logger.info(
            "lora.prepare_content_anchor ok driver=%s edited=%s bytes=%d",
            driver_path, out_path, len(edited),
        )
        return ContentAnchor(source_frame_path, str(out_path))

    def animate_image(self, *, image_path: str, prompt: str) -> tuple[str, ClipProbeResult]:
        """Image-to-video: turn a still (often an edited frame) into a clip.

        Returns the derived clip path and its probe. Raises
        `VideoRestylerError` / `ClipProcessorError`.
        """
        api_key = self.state.app_settings.fal_api_key
        image_bytes = Path(image_path).read_bytes()
        logger.info(
            "lora.animate_image start image=%s bytes=%d prompt=%r",
            image_path,
            len(image_bytes),
            prompt[:80],
        )
        video = self._video_restyler.animate(
            image_bytes=image_bytes, prompt=prompt, api_key=api_key or ""
        )
        out_path, probe = self._write_derived_clip(video)
        logger.info("lora.animate_image ok image=%s out=%s %.2fs", image_path, out_path, probe.duration_seconds)
        return out_path, probe

    def restyle_clip(self, *, source_path: str, prompt: str) -> tuple[str, ClipProbeResult]:
        """Video-to-video: re-render an existing clip under a text prompt.

        Returns the derived clip path and its probe. Raises
        `VideoRestylerError` / `ClipProcessorError`.
        """
        api_key = self.state.app_settings.fal_api_key
        video_bytes = Path(source_path).read_bytes()
        logger.info(
            "lora.restyle_clip start source=%s bytes=%d prompt=%r",
            source_path,
            len(video_bytes),
            prompt[:80],
        )
        video = self._video_restyler.restyle(
            video_bytes=video_bytes, prompt=prompt, api_key=api_key or ""
        )
        out_path, probe = self._write_derived_clip(video)
        logger.info("lora.restyle_clip ok source=%s out=%s %.2fs", source_path, out_path, probe.duration_seconds)
        return out_path, probe

    def motion_edit_clip(
        self,
        *,
        source_path: str,
        reference_image_path: str | None,
        prompt: str,
        engine: Literal["ltx_v2v", "kling_motion", "kling_o3"],
        video_strength: float,
        character_orientation: str,
        keep_audio: bool = True,
    ) -> tuple[str, ClipProbeResult]:
        """Motion-locked paired generation: drive motion from the original
        clip while anchoring content to an edited still.

        Returns the derived (target) clip path and its probe. The caller
        persists it as a clip whose `referencePath` points back at the
        original, forming an aligned pair. Raises `VideoRestylerError` /
        `ClipProcessorError`.
        """
        api_key = self.state.app_settings.fal_api_key or ""
        video_bytes = Path(source_path).read_bytes()
        # Kling O3 may run image-free (pure video + prompt); the LTX / Kling-
        # motion engines always anchor on the edited still.
        image_bytes = (
            Path(reference_image_path).read_bytes() if reference_image_path else None
        )
        logger.info(
            "lora.motion_edit_clip start engine=%s source=%s ref=%s video_bytes=%d image_bytes=%s prompt=%r",
            engine,
            source_path,
            reference_image_path,
            len(video_bytes),
            len(image_bytes) if image_bytes is not None else "none",
            prompt[:80],
        )
        if engine == "kling_o3":
            # Kling O3 rejects clips longer than ~10s (`video_duration_too_long`).
            # Trim the source to the cap before uploading so long Pexels/imported
            # clips don't fail outright.
            video_bytes = self._cap_video_for_kling_o3(source_path, video_bytes)
            video = self._video_restyler.kling_v2v_edit(
                video_bytes=video_bytes,
                image_bytes=image_bytes,
                prompt=prompt,
                keep_audio=keep_audio,
                api_key=api_key,
            )
        elif image_bytes is None:
            raise VideoRestylerError(
                f"A reference image is required for the {engine} engine.",
                status_code=400,
            )
        elif engine == "kling_motion":
            video = self._video_restyler.motion_transfer(
                image_bytes=image_bytes,
                video_bytes=video_bytes,
                prompt=prompt,
                character_orientation=character_orientation,
                api_key=api_key,
            )
        else:
            video = self._video_restyler.motion_edit(
                video_bytes=video_bytes,
                image_bytes=image_bytes,
                prompt=prompt,
                video_strength=video_strength,
                api_key=api_key,
            )
        out_path, probe = self._write_derived_clip(video)
        logger.info(
            "lora.motion_edit_clip ok engine=%s out=%s %.2fs", engine, out_path, probe.duration_seconds
        )
        return out_path, probe

    def _cap_video_for_kling_o3(self, source_path: str, video_bytes: bytes) -> bytes:
        """Return clip bytes within Kling O3's duration and width limits.

        Kling O3 hard-rejects videos over ~10s (`video_duration_too_long`,
        max 10.05s) and wider than 2160px (`video_too_large`). When the source
        exceeds either limit we trim to `_KLING_O3_MAX_DURATION_SECONDS` and/or
        downscale to `_KLING_O3_MAX_WIDTH` (preserving aspect ratio) and upload
        that instead; clips within both limits pass through untouched. A probe
        failure also passes through so the upload (and any clearer downstream
        error) still proceeds.
        """
        try:
            probe = self._clip_processor.probe(video_path=source_path)
        except ClipProcessorError:
            return video_bytes
        needs_trim = probe.duration_seconds > _KLING_O3_MAX_DURATION_SECONDS
        needs_scale = probe.width > _KLING_O3_MAX_WIDTH
        if not needs_trim and not needs_scale:
            return video_bytes
        plan = EditPlan(
            trim=(
                TrimSpec(start_seconds=0.0, end_seconds=_KLING_O3_MAX_DURATION_SECONDS)
                if needs_trim
                else None
            ),
            scale=(
                ScaleSpec(
                    width=_KLING_O3_MAX_WIDTH,
                    height=self._even(probe.height * (_KLING_O3_MAX_WIDTH / probe.width)),
                )
                if needs_scale and probe.width > 0
                else None
            ),
        )
        capped_path = self._allocate_derived_path(suffix=".mp4")
        logger.info(
            "lora.kling_o3 cap source=%s %.2fs %dx%d -> trim=%s scale=%s (Kling O3 limits)",
            source_path,
            probe.duration_seconds,
            probe.width,
            probe.height,
            needs_trim,
            needs_scale,
        )
        self._clip_processor.render(
            source_path=source_path,
            plan=plan,
            out_path=str(capped_path),
        )
        return Path(capped_path).read_bytes()

    def _write_derived_clip(self, video: bytes) -> tuple[str, ClipProbeResult]:
        out_path = self._allocate_derived_path(suffix=".mp4")
        out_path.write_bytes(video)
        probe = self._clip_processor.probe(video_path=str(out_path))
        return str(out_path), probe

    def _allocate_derived_path(self, *, suffix: str) -> Path:
        self._derived_dir.mkdir(parents=True, exist_ok=True)
        return self._derived_dir / f"{uuid.uuid4().hex}{suffix}"

    # ------------------------------------------------------------------
    # Pexels stock-media browser (BYOK; no lock held during network I/O)
    # ------------------------------------------------------------------

    def search_pexels(
        self,
        *,
        query: str,
        media: Literal["video", "photo"],
        page: int,
        per_page: int,
        orientation: str,
    ) -> PexelsSearchResult:
        """Search Pexels for stock photos/videos to add to a collection."""
        api_key = self.state.app_settings.pexels_api_key or ""
        if not api_key:
            raise PexelsError(
                "Add a Pexels API key in Settings to browse stock media.",
                status_code=400,
            )
        return self._pexels.search(
            query=query,
            media=media,
            page=page,
            per_page=per_page,
            orientation=orientation,
            api_key=api_key,
        )

    def download_pexels_asset(
        self, *, url: str, kind: Literal["video", "photo"], ext: str
    ) -> tuple[str, ClipProbeResult | None]:
        """Download a chosen Pexels asset into app storage.

        Returns the local path and (for videos) its ffmpeg probe so the
        frontend can register it as a dataset clip. Photos return `None`
        for the probe (no ffmpeg metadata). Raises `PexelsError` /
        `ClipProcessorError`.
        """
        api_key = self.state.app_settings.pexels_api_key or ""
        data = self._pexels.download(url=url, api_key=api_key)
        suffix = f".{ext.lstrip('.')}" if ext else (".mp4" if kind == "video" else ".jpg")
        self._pexels_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._pexels_dir / f"{uuid.uuid4().hex}{suffix}"
        out_path.write_bytes(data)
        probe = (
            self._clip_processor.probe(video_path=str(out_path))
            if kind == "video"
            else None
        )
        logger.info(
            "lora.pexels_download ok kind=%s out=%s bytes=%d", kind, out_path, len(data)
        )
        return str(out_path), probe

    # ------------------------------------------------------------------
    # Clip-prep jobs (durable ledger; executed by ClipJobsRunner)
    # ------------------------------------------------------------------

    @with_state_lock
    def enqueue_clip_jobs(
        self, *, source_paths: list[str], kind: ClipJobKind
    ) -> list[ClipJob]:
        """Create (or reuse) prep jobs for a batch of source clips.

        Idempotent per (source_path, kind): if a non-terminal job already
        exists for a path it's returned as-is rather than duplicated, so a
        polling UI can safely re-enqueue. Wakes the clip-jobs runner.
        """
        existing = {
            (j.source_path, j.kind): j
            for j in self._clip_jobs.jobs
            if j.status in ("pending", "running")
        }
        result: list[ClipJob] = []
        created = False
        for path in source_paths:
            reused = existing.get((path, kind))
            if reused is not None:
                result.append(reused)
                continue
            job = ClipJob(
                id=uuid.uuid4().hex,
                kind=kind,
                source_path=path,
                status="pending",
                created_at=_now_iso(),
            )
            self._clip_jobs.jobs.append(job)
            existing[(path, kind)] = job
            result.append(job)
            created = True
        if created:
            self._persist_clip_jobs_unlocked()
            self._clip_jobs_wakeup.set()
        return [j.model_copy(deep=True) for j in result]

    @with_state_lock
    def get_clip_jobs_state(self) -> ClipJobsState:
        return self._clip_jobs.model_copy(deep=True)

    @with_state_lock
    def claim_pending_clip_jobs(self) -> list[ClipJob]:
        """Mark all pending jobs `running` and return snapshots.

        The runner submits these to its bounded pool; concurrency is
        capped by the pool, not here. Marking at claim time prevents the
        runner from double-submitting a job across ticks.
        """
        claimed: list[ClipJob] = []
        for job in self._clip_jobs.jobs:
            if job.status == "pending":
                job.status = "running"
                job.updated_at = _now_iso()
                claimed.append(job.model_copy(deep=True))
        if claimed:
            self._persist_clip_jobs_unlocked()
        return claimed

    @with_state_lock
    def set_clip_job_poster(self, job_id: str, *, poster_path: str) -> None:
        """Publish a job's poster while it's still `running`.

        The poster frame is a cheap single-seek extract, but the sprite
        filmstrip forces a full-clip decode. Reporting the poster early lets
        the gallery swap the loading spinner for the real thumbnail within a
        poll cycle instead of waiting for the whole filmstrip to render.
        """
        job = self._find_clip_job(job_id)
        if job is None:
            return
        job.poster_path = poster_path
        job.updated_at = _now_iso()
        self._persist_clip_jobs_unlocked()

    @with_state_lock
    def complete_sprite_job(
        self, job_id: str, *, poster_path: str, sprite_path: str, sprite_tiles: int
    ) -> None:
        job = self._find_clip_job(job_id)
        if job is None:
            return
        job.status = "completed"
        job.poster_path = poster_path
        job.sprite_path = sprite_path
        job.sprite_tiles = sprite_tiles
        job.error = None
        job.updated_at = _now_iso()
        self._persist_clip_jobs_unlocked()

    @with_state_lock
    def fail_clip_job(self, job_id: str, error: str) -> None:
        job = self._find_clip_job(job_id)
        if job is None:
            return
        job.status = "failed"
        job.error = error
        job.updated_at = _now_iso()
        self._persist_clip_jobs_unlocked()

    def _find_clip_job(self, job_id: str) -> ClipJob | None:
        return next((j for j in self._clip_jobs.jobs if j.id == job_id), None)

    def allocate_thumb_path(self, *, suffix: str) -> Path:
        self._thumbs_dir.mkdir(parents=True, exist_ok=True)
        return self._thumbs_dir / f"{uuid.uuid4().hex}{suffix}"

    # ------------------------------------------------------------------
    # Target/variant derivation jobs (durable; run by LoraDerivationRunner)
    # ------------------------------------------------------------------

    @with_state_lock
    def enqueue_derivation_job(self, req: CreateLoraDerivationJobRequest) -> DerivationJob:
        """Create a background 'generate target/variant' pipeline job."""
        job = DerivationJob(
            id=uuid.uuid4().hex,
            status="pending",
            engine=req.engine,
            direction=req.direction,
            label=req.label,
            driver_path=req.driverPath,
            frame_path=req.framePath,
            reference_path=req.referencePath,
            dataset_id=req.datasetId,
            source_clip_id=req.sourceClipId,
            frame_time_seconds=req.frameTimeSeconds,
            edit_prompt=req.editPrompt,
            nano_banana_model=req.nanoBananaModel,
            edit_engine=req.editEngine,
            scene_prompt=req.scenePrompt,
            conditioning_type=req.conditioningType,
            conditioning_strength=req.conditioningStrength,
            character_orientation=req.characterOrientation,
            keep_audio=req.keepAudio,
            frame_edited=req.frameEdited,
            caption=req.caption,
            require_review=req.requireReview,
            created_at=_now_iso(),
        )
        self._derivation.jobs.append(job)
        self._persist_derivation_unlocked()
        self._derivation_wakeup.set()
        return job.model_copy(deep=True)

    @with_state_lock
    def get_derivation_jobs_state(self) -> DerivationJobsState:
        return self._derivation.model_copy(deep=True)

    @with_state_lock
    def derivation_concurrency(self) -> int:
        """Max simultaneous Fal derivation jobs (user setting, clamped 1..20).

        Read live by the derivation runner so changing it in Settings takes
        effect without a restart. Local GPU drives self-serialize regardless.
        """
        return max(1, min(20, int(self.state.app_settings.lora_fal_concurrency)))

    @with_state_lock
    def claim_next_derivation_job(self) -> DerivationJob | None:
        """Claim the oldest job ready for work, marking it in-flight.

        Two claimable phases (the runner branches on the returned status):
          - `pending`  -> `editing`    : build the anchor (edit phase).
          - `approved` -> `generating` : motion drive only (edit already
            reviewed + approved; reuse `edited_frame_path`).

        `review` jobs are intentionally not claimed — they wait for the user.

        One job per call (atomic under the lock). The derivation runner's
        bounded pool calls this repeatedly to fill up to `lora_fal_concurrency`
        workers; local IC-LoRA drives still self-serialize on the single GPU.

        Local GPU (FLUX.2 [klein] 9B) edits are serialized here to one in-flight
        job at a time, regardless of the Fal concurrency setting. The Klein
        pipeline load is ~32GB and isn't exclusive at the pipeline level — the
        GPU-busy flag (`start_generation`) is only set *after* the load, so
        without this guard the Fal concurrency limit would let many workers
        load Klein simultaneously, blow up CPU RAM, and crash the backend
        (which then can't honor cancel-all). One at a time also means the
        remaining Klein jobs stay `pending` and cancel-all drops them
        immediately instead of them being stuck `editing`.
        """
        klein_in_flight = any(
            j.status in ("editing", "generating")
            and j.edit_engine == "klein"
            and j.edit_prompt.strip()
            for j in self._derivation.jobs
        )
        for job in self._derivation.jobs:
            if job.status == "pending":
                if (
                    klein_in_flight
                    and job.edit_engine == "klein"
                    and job.edit_prompt.strip()
                ):
                    continue
                job.status = "editing"
                job.updated_at = _now_iso()
                self._persist_derivation_unlocked()
                return job.model_copy(deep=True)
            if job.status == "approved":
                job.status = "generating"
                job.updated_at = _now_iso()
                self._persist_derivation_unlocked()
                return job.model_copy(deep=True)
        return None

    @with_state_lock
    def mark_derivation_generating(
        self,
        job_id: str,
        *,
        edited_frame_path: str | None,
        source_frame_path: str | None = None,
    ) -> None:
        job = self._find_derivation_job(job_id)
        if job is None:
            return
        job.status = "generating"
        if edited_frame_path is not None:
            job.edited_frame_path = edited_frame_path
        if source_frame_path is not None:
            job.source_frame_path = source_frame_path
        job.updated_at = _now_iso()
        self._persist_derivation_unlocked()

    @with_state_lock
    def mark_derivation_review(
        self,
        job_id: str,
        *,
        edited_frame_path: str,
        source_frame_path: str | None = None,
    ) -> None:
        """Pause an edited job for user review before the motion drive."""
        job = self._find_derivation_job(job_id)
        if job is None:
            return
        job.status = "review"
        job.edited_frame_path = edited_frame_path
        if source_frame_path is not None:
            job.source_frame_path = source_frame_path
        job.updated_at = _now_iso()
        self._persist_derivation_unlocked()

    @with_state_lock
    def approve_derivation_job(self, job_id: str) -> DerivationJob | None:
        """Approve a reviewed edit; queue it for the motion-only phase."""
        job = self._find_derivation_job(job_id)
        if job is None or job.status != "review":
            return None
        job.status = "approved"
        job.updated_at = _now_iso()
        self._persist_derivation_unlocked()
        self._derivation_wakeup.set()
        return job.model_copy(deep=True)

    @with_state_lock
    def regenerate_derivation_edit(
        self, job_id: str, *, edit_prompt: str | None = None
    ) -> DerivationJob | None:
        """Re-run the edit for a reviewed job (optionally with a new prompt)."""
        job = self._find_derivation_job(job_id)
        if job is None or job.status != "review":
            return None
        if edit_prompt is not None and edit_prompt.strip():
            job.edit_prompt = edit_prompt.strip()
        job.status = "pending"
        job.edited_frame_path = None
        job.updated_at = _now_iso()
        self._persist_derivation_unlocked()
        self._derivation_wakeup.set()
        return job.model_copy(deep=True)

    @with_state_lock
    def complete_derivation_job(
        self, job_id: str, *, derived_path: str, probe: ClipProbeResult
    ) -> None:
        job = self._find_derivation_job(job_id)
        if job is None:
            return
        # Worker callbacks can arrive after a retry/cancel or be replayed after
        # restart. Terminal states are immutable; never revive a cancelled job
        # or replace an already delivered completion.
        if job.status in ("completed", "failed", "cancelled"):
            return
        if job.cancel_requested:
            job.status = "cancelled"
            job.cancel_requested = False
            job.updated_at = _now_iso()
            self._persist_derivation_unlocked()
            return
        job.status = "completed"
        job.derived_path = derived_path
        job.probe = LoraClipProbeApi(
            durationSeconds=probe.duration_seconds,
            width=probe.width,
            height=probe.height,
            fps=probe.fps,
            frameCount=probe.frame_count,
            hasAudio=probe.has_audio,
            videoCodec=probe.video_codec,
        )
        job.error = None
        job.cancel_requested = False
        job.updated_at = _now_iso()
        self._persist_derivation_unlocked()

    @with_state_lock
    def fail_derivation_job(self, job_id: str, error: str) -> None:
        job = self._find_derivation_job(job_id)
        if job is None:
            return
        if job.status in ("completed", "failed", "cancelled"):
            return
        # A cancel that landed mid-flight presents as a failure from the
        # pipeline; record it as cancelled so the UI shows the right state.
        was_cancel_requested = job.cancel_requested
        job.status = "cancelled" if was_cancel_requested else "failed"
        job.error = None if was_cancel_requested else error
        job.cancel_requested = False
        job.updated_at = _now_iso()
        self._persist_derivation_unlocked()

    @with_state_lock
    def cancel_derivation_job(self, job_id: str) -> DerivationJob | None:
        job = self._find_derivation_job(job_id)
        if job is None:
            return None
        if job.status in ("completed", "failed", "cancelled"):
            return job.model_copy(deep=True)
        if job.status in ("pending", "review", "approved"):
            # Not actively in flight (queued or paused) — cancel outright.
            job.status = "cancelled"
            job.cancel_requested = False
        else:
            # In-flight: flag it; the runner checks between stages.
            job.cancel_requested = True
        job.updated_at = _now_iso()
        self._persist_derivation_unlocked()
        return job.model_copy(deep=True)

    @with_state_lock
    def cancel_all_derivation_jobs(self, *, dataset_id: str | None = None) -> int:
        """Cancel every active derivation job (the UI's "Cancel all").

        Mirrors `cancel_derivation_job` per job: queued/paused jobs go
        straight to `cancelled`; in-flight ones (`editing`/`generating`) are
        flagged so the runner stops between stages. Optionally scoped to a
        single dataset so other collections keep running. Returns the number
        of jobs affected.
        """
        active = ("pending", "editing", "review", "approved", "generating")
        count = 0
        for job in self._derivation.jobs:
            if job.status not in active:
                continue
            if dataset_id is not None and job.dataset_id != dataset_id:
                continue
            if job.status in ("pending", "review", "approved"):
                job.status = "cancelled"
                job.cancel_requested = False
            else:
                job.cancel_requested = True
            job.updated_at = _now_iso()
            count += 1
        if count:
            self._persist_derivation_unlocked()
        return count

    @with_state_lock
    def retry_derivation_job(self, job_id: str) -> DerivationJob | None:
        job = self._find_derivation_job(job_id)
        if job is None or job.status not in ("failed", "cancelled"):
            return None
        # A failure after the edit phase must not discard a successfully
        # reviewed/generated anchor. Resume from the motion-only claim state.
        resume_from_anchor = job.edited_frame_path is not None
        job.status = "approved" if resume_from_anchor else "pending"
        job.error = None
        job.cancel_requested = False
        if not resume_from_anchor:
            job.edited_frame_path = None
        job.derived_path = None
        job.probe = None
        job.updated_at = _now_iso()
        self._persist_derivation_unlocked()
        self._derivation_wakeup.set()
        return job.model_copy(deep=True)

    @with_state_lock
    def dismiss_derivation_job(self, job_id: str) -> None:
        """Drop a terminal job from the ledger (UI 'clear')."""
        before = len(self._derivation.jobs)
        self._derivation.jobs = [
            j
            for j in self._derivation.jobs
            if not (j.id == job_id and j.status in ("completed", "failed", "cancelled"))
        ]
        if len(self._derivation.jobs) != before:
            self._persist_derivation_unlocked()

    @with_state_lock
    def is_derivation_cancelled(self, job_id: str) -> bool:
        job = self._find_derivation_job(job_id)
        return job is not None and job.cancel_requested

    def _find_derivation_job(self, job_id: str) -> DerivationJob | None:
        return next((j for j in self._derivation.jobs if j.id == job_id), None)

    # ------------------------------------------------------------------
    # Persistence + crash recovery
    # ------------------------------------------------------------------

    @with_state_lock
    def load_state(self) -> None:
        self._datasets = self._load_file(self._datasets_file, LoraDatasetsState)
        # v1 -> v2: introduced `folders` + `dataset.folder_id`. Both default
        # safely (empty list / None), so this is a no-op backfill — we only
        # advance the recorded version so the ledger persists at v2 going
        # forward. No data transformation needed.
        if self._datasets.schema_version < 3:
            self._datasets.schema_version = 3
            self._persist_datasets_unlocked()
        self._preprocessed = self._load_file(self._preprocessed_file, PreprocessedState)
        self._training = self._load_file(self._training_file, TrainingState)
        if self._training.schema_version < 2:
            self._training.schema_version = 2
            self._persist_training_unlocked()
        self._saved_models = self._load_file(self._saved_models_file, SavedModelState)
        # Seed the built-in profiles the first time the ledger is created
        # (file absent), not whenever it's merely empty — so a user who
        # deletes every profile doesn't get them silently re-added.
        profiles_existed = self._profiles_file.exists()
        self._profiles = self._load_file(self._profiles_file, LoraTrainingProfilesState)
        if not profiles_existed and not self._profiles.profiles:
            self._profiles.profiles = default_training_profiles(_now_iso())
            self._persist_profiles_unlocked()
        else:
            # One-time, version-gated migrations on an existing ledger (each
            # gated on the version bump, not mere absence, so a profile the user
            # deleted/edited post-migration isn't silently re-added/overwritten).
            if self._profiles.schema_version < 5:
                self._migrate_curated_profiles_unlocked()
        self._clip_jobs = self._load_file(self._clip_jobs_file, ClipJobsState)
        self._derivation = self._load_file(self._derivation_file, DerivationJobsState)
        self._recover_unlocked()

    def _load_file(self, path: Path, model: type[_MODEL]) -> _MODEL:
        if not path.exists():
            return model()
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return model.model_validate(payload)
        except Exception as exc:
            backup = path.with_name(f"{path.stem}.corrupt-{int(time.time())}.json")
            try:
                shutil.copy2(path, backup)
                logger.warning(
                    "%s could not be parsed; backed up to %s and starting "
                    "empty: %s",
                    path.name,
                    backup,
                    exc,
                )
            except Exception:
                logger.warning(
                    "%s could not be parsed and backup failed; starting "
                    "empty: %s",
                    path.name,
                    exc,
                )
            return model()

    def _recover_unlocked(self) -> None:
        changed = False
        for dataset in self._datasets.datasets:
            # Upload is idempotent (re-upload overwrites); keep
            # "uploading" so the reconciler simply re-runs it.
            if dataset.status == "uploading" and dataset.target is None:
                # Never got far enough to allocate a workspace — safe to
                # leave; reconciler restarts the upload from scratch.
                continue
        for item in self._preprocessed.items:
            if item.status in ("captioning", "preprocessing") and item.target is None:
                # No live remote handle after restart. If the user had asked to
                # cancel, honor it now (the remote job is gone with the pod);
                # otherwise reset to pending so the runner re-submits.
                if item.cancel_requested:
                    item.status = "cancelled"
                    item.cancel_requested = False
                    item.error = None
                    item.completed_at = _now_iso()
                else:
                    item.status = "pending"
                changed = True
        for job in self._training.items:
            if job.status == "running" and job.target is None:
                job.status = "pending"
                job.started_at = None
                changed = True
        # Clip jobs are idempotent local work: anything left `running`
        # (process died mid-sprite) is reset so the runner re-attempts it.
        clip_jobs_changed = False
        for clip_job in self._clip_jobs.jobs:
            if clip_job.status == "running":
                clip_job.status = "pending"
                clip_jobs_changed = True
        if clip_jobs_changed:
            self._persist_clip_jobs_unlocked()
        # Derivation jobs: a process that died mid-pipeline left a job
        # in-flight. `editing` restarts from the source (`pending`). A
        # `generating` job that already has an approved edited still resumes
        # the motion-only phase (`approved`) so it isn't re-edited; otherwise
        # it restarts from the source. `review` is paused — leave it be.
        derivation_changed = False
        for deriv_job in self._derivation.jobs:
            if deriv_job.status == "editing":
                deriv_job.status = "pending"
                deriv_job.edited_frame_path = None
                deriv_job.cancel_requested = False
                derivation_changed = True
            elif deriv_job.status == "generating":
                deriv_job.cancel_requested = False
                if deriv_job.edited_frame_path:
                    deriv_job.status = "approved"
                else:
                    deriv_job.status = "pending"
                    deriv_job.edited_frame_path = None
                derivation_changed = True
        if derivation_changed:
            self._persist_derivation_unlocked()
        if changed:
            self._persist_all_unlocked()
            logger.info("LoRA trainer: recovered in-flight items after restart")

    def _persist(self, path: Path, payload: BaseModel) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload.model_dump(mode="json"), f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _persist_datasets_unlocked(self) -> None:
        self._persist(self._datasets_file, self._datasets)

    def _persist_preprocessed_unlocked(self) -> None:
        self._persist(self._preprocessed_file, self._preprocessed)

    def _persist_training_unlocked(self) -> None:
        self._persist(self._training_file, self._training)

    def _persist_saved_models_unlocked(self) -> None:
        self._persist(self._saved_models_file, self._saved_models)

    def _persist_profiles_unlocked(self) -> None:
        self._persist(self._profiles_file, self._profiles)

    def _backfill_builtin_profiles_unlocked(self) -> None:
        """Add newly-introduced built-in profiles to an older ledger once.

        Only profiles whose id is absent are appended (never the Standard /
        Low VRAM starters a user may have deleted on purpose). Bumps the
        ledger's `schema_version` so the back-fill runs exactly once.
        """
        existing_ids = {p.id for p in self._profiles.profiles}
        now = _now_iso()
        for builtin in default_training_profiles(now):
            if builtin.id == BUILTIN_DETAILED_RANK64_ID and builtin.id not in existing_ids:
                self._profiles.profiles.append(builtin)
        self._profiles.schema_version = 2
        self._persist_profiles_unlocked()

    def _sync_low_vram_profile_unlocked(self) -> None:
        """Re-sync the built-in "Low VRAM" profile to the current default once.

        It previously shipped at rank 32; it now matches the trainer's official
        `t2v_lora_low_vram.yaml` (rank 16). Only update it when it's still at the
        old rank-32 default — a user who customized it (or changed its rank)
        keeps their version untouched.
        """
        previous_default = TrainingConfig(preset="low_vram")  # the old rank-32 seed
        for profile in self._profiles.profiles:
            if (
                profile.id == BUILTIN_LOW_VRAM_ID
                and profile.builtin
                and profile.config == previous_default
            ):
                profile.config = TrainingConfig(preset="low_vram", rank=16, alpha=16)
                profile.updated_at = _now_iso()
        self._profiles.schema_version = 3
        self._persist_profiles_unlocked()

    def _backfill_int4_profile_unlocked(self) -> None:
        """Add the built-in "Low VRAM (int4)" profile to an older ledger once.

        Appended only when absent (so a user who deleted it post-migration
        doesn't see it reappear). Bumps `schema_version` so this runs once.
        """
        existing_ids = {p.id for p in self._profiles.profiles}
        if BUILTIN_LOW_VRAM_INT4_ID not in existing_ids:
            by_id = {p.id: p for p in default_training_profiles(_now_iso())}
            self._profiles.profiles.append(by_id[BUILTIN_LOW_VRAM_INT4_ID])
        self._profiles.schema_version = 4
        self._persist_profiles_unlocked()

    def _migrate_curated_profiles_unlocked(self) -> None:
        """Replace legacy hardware presets without losing user customizations.

        Old built-ins were editable. If either their name or config differs
        from the shipped value, preserve that work as a normal custom profile
        before removing the legacy entry. Runs are unaffected because they
        snapshot configs rather than profile ids.
        """
        legacy_defaults: dict[str, tuple[str, list[TrainingConfig]]] = {
            BUILTIN_STANDARD_ID: ("Standard", [TrainingConfig(preset="standard")]),
            BUILTIN_LOW_VRAM_ID: (
                "Low VRAM",
                [
                    TrainingConfig(preset="low_vram"),
                    TrainingConfig(preset="low_vram", rank=16, alpha=16),
                ],
            ),
            BUILTIN_LOW_VRAM_INT4_ID: (
                "Low VRAM (int4)",
                [
                    TrainingConfig(
                        preset="low_vram",
                        rank=16,
                        alpha=16,
                        quantization="int4-quanto",
                    )
                ],
            ),
            BUILTIN_DETAILED_RANK64_ID: (
                "Detailed (rank 64)",
                [legacy_detailed_rank64_config()],
            ),
        }
        migrated: list[LoraTrainingProfile] = []
        for profile in self._profiles.profiles:
            legacy = legacy_defaults.get(profile.id)
            if not profile.builtin or legacy is None:
                migrated.append(profile)
                continue
            expected_name, expected_configs = legacy
            if profile.name != expected_name or profile.config not in expected_configs:
                migrated.append(
                    profile.model_copy(
                        update={
                            "id": uuid.uuid4().hex,
                            "builtin": False,
                            "description": "Preserved from a customized legacy built-in.",
                            "dataset_types": ["standard", "ic_lora"],
                            "min_vram_gb": None,
                            "auto_recommended": False,
                            "updated_at": _now_iso(),
                        },
                        deep=True,
                    )
                )

        existing_ids = {profile.id for profile in migrated}
        for builtin in default_training_profiles(_now_iso()):
            if builtin.id not in existing_ids:
                migrated.append(builtin)
        self._profiles.profiles = migrated
        self._profiles.schema_version = 5
        self._persist_profiles_unlocked()

    def _persist_clip_jobs_unlocked(self) -> None:
        self._persist(self._clip_jobs_file, self._clip_jobs)

    def _persist_derivation_unlocked(self) -> None:
        self._persist(self._derivation_file, self._derivation)

    def _persist_all_unlocked(self) -> None:
        self._persist_datasets_unlocked()
        self._persist_preprocessed_unlocked()
        self._persist_training_unlocked()

    # ------------------------------------------------------------------
    # Read API (deep-copied snapshots)
    # ------------------------------------------------------------------

    @with_state_lock
    def get_datasets_state(self) -> LoraDatasetsState:
        return self._datasets.model_copy(deep=True)

    @with_state_lock
    def get_preprocessed_state(self) -> PreprocessedState:
        return self._preprocessed.model_copy(deep=True)

    @with_state_lock
    def get_training_state(self) -> TrainingState:
        return self._training.model_copy(deep=True)

    def get_training_state_by_id(self, training_id: str) -> TrainingJob | None:
        """Read-only lookup of a single training job by id (or None)."""
        with self._lock:
            found = self._find_training(training_id)
            return found.model_copy(deep=True) if found is not None else None

    # ------------------------------------------------------------------
    # Training profiles CRUD
    # ------------------------------------------------------------------

    @with_state_lock
    def get_profiles_state(self) -> LoraTrainingProfilesState:
        return self._profiles.model_copy(deep=True)

    def _find_profile(self, profile_id: str) -> LoraTrainingProfile | None:
        return next((p for p in self._profiles.profiles if p.id == profile_id), None)

    def _require_profile(self, profile_id: str) -> LoraTrainingProfile:
        profile = self._find_profile(profile_id)
        if profile is None:
            raise LoraEntityNotFoundError(f"Training profile not found: {profile_id}")
        return profile

    @with_state_lock
    def get_profile(self, profile_id: str) -> LoraTrainingProfile:
        return self._require_profile(profile_id).model_copy(deep=True)

    @with_state_lock
    def create_profile(
        self,
        *,
        name: str,
        config: TrainingConfig,
        description: str = "",
        dataset_types: list[LoraDatasetType] | None = None,
    ) -> LoraTrainingProfile:
        profile = LoraTrainingProfile(
            id=uuid.uuid4().hex,
            name=name,
            created_at=_now_iso(),
            updated_at=_now_iso(),
            config=config,
            builtin=False,
            description=description,
            dataset_types=dataset_types or ["standard", "ic_lora"],
        )
        self._profiles.profiles.append(profile)
        self._persist_profiles_unlocked()
        return profile.model_copy(deep=True)

    @with_state_lock
    def update_profile(
        self,
        profile_id: str,
        *,
        name: str | None = None,
        config: TrainingConfig | None = None,
        description: str | None = None,
        dataset_types: list[LoraDatasetType] | None = None,
    ) -> LoraTrainingProfile:
        profile = self._require_profile(profile_id)
        if profile.builtin:
            raise LoraTransitionError(
                "Built-in training profiles are read-only. Duplicate this profile to customize it."
            )
        if name is not None:
            profile.name = name
        if config is not None:
            profile.config = config
        if description is not None:
            profile.description = description
        if dataset_types is not None:
            profile.dataset_types = dataset_types
        profile.updated_at = _now_iso()
        self._persist_profiles_unlocked()
        return profile.model_copy(deep=True)

    @with_state_lock
    def delete_profile(self, profile_id: str) -> None:
        profile = self._require_profile(profile_id)
        if profile.builtin:
            raise LoraTransitionError(
                "Built-in training profiles cannot be deleted."
            )
        self._profiles.profiles.remove(profile)
        self._persist_profiles_unlocked()

    # ------------------------------------------------------------------
    # Dataset CRUD
    # ------------------------------------------------------------------

    @with_state_lock
    def create_dataset(
        self,
        *,
        name: str,
        dataset_type: LoraDatasetType = "standard",
        trigger_word: str | None,
        clips: list[LoraClip],
        originating_project_id: str | None = None,
    ) -> LoraDataset:
        dataset = LoraDataset(
            id=uuid.uuid4().hex,
            name=name,
            created_at=_now_iso(),
            status="draft",
            type=dataset_type,
            trigger_word=trigger_word,
            clips=clips,
            updated_at=_now_iso(),
            originating_project_id=originating_project_id,
        )
        self._datasets.datasets.append(dataset)
        self._persist_datasets_unlocked()
        return dataset.model_copy(deep=True)

    @with_state_lock
    def update_dataset(
        self,
        dataset_id: str,
        *,
        name: str | None,
        dataset_type: LoraDatasetType | None = None,
        trigger_word: str | None,
        clips: list[LoraClip] | None,
    ) -> LoraDataset:
        dataset = self._require_dataset(dataset_id)
        if dataset.status not in ("draft", "upload_failed"):
            raise LoraTransitionError(
                f"Cannot edit dataset {dataset_id} in status "
                f"{dataset.status!r}; only draft/upload_failed are editable"
            )
        if name is not None:
            dataset.name = name
        if dataset_type is not None:
            dataset.type = dataset_type
        if trigger_word is not None:
            dataset.trigger_word = trigger_word
        if clips is not None:
            dataset.clips = clips
        dataset.updated_at = _now_iso()
        self._persist_datasets_unlocked()
        return dataset.model_copy(deep=True)

    @with_state_lock
    def rename_dataset(self, dataset_id: str, name: str) -> LoraDataset:
        """Rename a dataset at any status.

        Display-only: the remote dataset dir is recorded in `remote_dataset_dir`
        at upload time and never recomputed from the name, so renaming after
        upload is safe. Unlike `update_dataset` (which also edits clips/type/
        trigger and is therefore locked to draft/upload_failed), this is a
        name-only mutation allowed in every status.
        """
        dataset = self._require_dataset(dataset_id)
        clean = name.strip()
        if not clean:
            raise LoraTransitionError("Dataset name cannot be empty")
        dataset.name = clean
        dataset.updated_at = _now_iso()
        self._persist_datasets_unlocked()
        return dataset.model_copy(deep=True)

    # ----------------------------------------------------------------
    # Collection folders (sidebar organization, datasets-only)
    # ----------------------------------------------------------------

    def _require_folder(self, folder_id: str) -> LoraFolder:
        for f in self._datasets.folders:
            if f.id == folder_id:
                return f
        raise LoraEntityNotFoundError(f"LoraFolder {folder_id} not found")

    def _descendant_folder_ids(self, folder_id: str) -> set[str]:
        """All folder ids nested (transitively) under `folder_id`, including
        itself. Used for cycle checks on reparent and for recursive delete."""
        descendants: set[str] = {folder_id}
        changed = True
        while changed:
            changed = False
            for f in self._datasets.folders:
                if f.parent_id in descendants and f.id not in descendants:
                    descendants.add(f.id)
                    changed = True
        return descendants

    @with_state_lock
    def create_folder(self, name: str, parent_id: str | None) -> LoraFolder:
        clean = name.strip()
        if not clean:
            raise LoraTransitionError("Folder name cannot be empty")
        if parent_id is not None:
            # Validate parent exists — raises LoraEntityNotFoundError if not.
            self._require_folder(parent_id)
        folder = LoraFolder(
            id=uuid.uuid4().hex,
            name=clean,
            parent_id=parent_id,
            created_at=_now_iso(),
        )
        self._datasets.folders.append(folder)
        self._persist_datasets_unlocked()
        return folder.model_copy(deep=True)

    @with_state_lock
    def rename_folder(self, folder_id: str, name: str) -> LoraFolder:
        folder = self._require_folder(folder_id)
        clean = name.strip()
        if not clean:
            raise LoraTransitionError("Folder name cannot be empty")
        folder.name = clean
        self._persist_datasets_unlocked()
        return folder.model_copy(deep=True)

    @with_state_lock
    def move_folder(self, folder_id: str, parent_id: str | None) -> LoraFolder:
        """Reparent a folder. Rejects cycles: `parent_id` cannot be the folder
        itself or one of its descendants. `parent_id=None` moves to root."""
        folder = self._require_folder(folder_id)
        if parent_id == folder_id:
            raise LoraTransitionError("Cannot move a folder into itself")
        if parent_id is not None:
            descendants = self._descendant_folder_ids(folder_id)
            if parent_id in descendants:
                raise LoraTransitionError(
                    "Cannot move a folder into one of its own descendants"
                )
            # Validate target exists.
            self._require_folder(parent_id)
        folder.parent_id = parent_id
        self._persist_datasets_unlocked()
        return folder.model_copy(deep=True)

    @with_state_lock
    def delete_folder(self, folder_id: str, *, recursive: bool) -> None:
        """Delete a folder.

        Non-recursive (default): contained datasets and subfolders are moved up
        to the deleted folder's `parent_id` (or root if it was a top-level
        folder). Recursive: subfolders are deleted recursively and every
        dataset in the subtree is deleted via the existing compute-release
        `delete_dataset` path.
        """
        folder = self._require_folder(folder_id)
        if recursive:
            subtree = self._descendant_folder_ids(folder_id)
            # Delete every dataset that lives in the subtree (own compute-release
            # path so active uploads / preprocessing are guarded against).
            for ds in list(self._datasets.datasets):
                if ds.folder_id in subtree:
                    self.delete_dataset(ds.id)
            # Then drop every folder in the subtree.
            self._datasets.folders = [
                f for f in self._datasets.folders if f.id not in subtree
            ]
        else:
            new_parent = folder.parent_id
            # Re-parent contained datasets up to the deleted folder's parent.
            for ds in self._datasets.datasets:
                if ds.folder_id == folder_id:
                    ds.folder_id = new_parent
            # Re-parent contained subfolders up.
            for f in self._datasets.folders:
                if f.parent_id == folder_id:
                    f.parent_id = new_parent
            self._datasets.folders = [
                f for f in self._datasets.folders if f.id != folder_id
            ]
        self._persist_datasets_unlocked()

    @with_state_lock
    def move_dataset(self, dataset_id: str, folder_id: str | None) -> LoraDataset:
        """Move a dataset into a folder. `folder_id=None` = root. Validates the
        target folder exists when given."""
        dataset = self._require_dataset(dataset_id)
        if folder_id is not None:
            self._require_folder(folder_id)
        dataset.folder_id = folder_id
        dataset.updated_at = _now_iso()
        self._persist_datasets_unlocked()
        return dataset.model_copy(deep=True)

    @with_state_lock
    def delete_dataset(self, dataset_id: str) -> None:
        dataset = self._require_dataset(dataset_id)
        if dataset.status == "uploading":
            raise LoraTransitionError(
                f"Cannot delete dataset {dataset_id} while uploading"
            )
        if any(p.dataset_id == dataset_id and p.status in ("pending", "captioning", "preprocessing")
               for p in self._preprocessed.items):
            raise LoraTransitionError(
                f"Cannot delete dataset {dataset_id}; an active preprocessing "
                "job references it"
            )
        preprocessed_ids = {
            item.id for item in self._preprocessed.items if item.dataset_id == dataset_id
        }
        if any(
            job.preprocessed_id in preprocessed_ids
            and job.status in ("pending", "running")
            for job in self._training.items
        ):
            raise LoraTransitionError(
                f"Cannot delete dataset {dataset_id}; an active training job "
                "references it"
            )
        self._datasets.datasets.remove(dataset)
        self._persist_datasets_unlocked()

    @with_state_lock
    def archive_dataset(self, dataset_id: str) -> LoraDataset:
        dataset = self._require_dataset(dataset_id)
        if dataset.archived_at is not None:
            return dataset.model_copy(deep=True)
        if dataset.status in ("uploading", "gpu_selection_required") or dataset.cancel_requested:
            raise LoraTransitionError(
                f"Cannot archive dataset {dataset_id} while work is active"
            )
        if dataset.auto_pipeline is not None:
            raise LoraTransitionError(
                f"Cannot archive dataset {dataset_id} while pipeline work is queued"
            )
        preprocessed_ids = {
            item.id for item in self._preprocessed.items if item.dataset_id == dataset_id
        }
        if any(
            item.dataset_id == dataset_id
            and (
                item.status in ("pending", "captioning", "preprocessing")
                or item.cancel_requested
            )
            for item in self._preprocessed.items
        ) or any(
            job.preprocessed_id in preprocessed_ids
            and (
                job.status in ("pending", "running", "gpu_selection_required")
                or job.cancel_requested
            )
            for job in self._training.items
        ):
            raise LoraTransitionError(
                f"Cannot archive dataset {dataset_id} while related work is active"
            )
        dataset.archived_at = _now_iso()
        dataset.updated_at = dataset.archived_at
        self._persist_datasets_unlocked()
        return dataset.model_copy(deep=True)

    @with_state_lock
    def unarchive_dataset(self, dataset_id: str) -> LoraDataset:
        dataset = self._require_dataset(dataset_id)
        dataset.archived_at = None
        dataset.updated_at = _now_iso()
        self._persist_datasets_unlocked()
        return dataset.model_copy(deep=True)

    @with_state_lock
    def _snapshot_dataset(self, dataset_id: str) -> LoraDataset:
        """Deep-copied read so export can do heavy file IO without the lock."""
        return self._require_dataset(dataset_id).model_copy(deep=True)

    def export_dataset(
        self,
        dataset_id: str,
        *,
        dest_path: str,
        export_format: str,
        include_rejected: bool,
        profile_id: str | None = None,
        prep_options: "LoraDatasetPrep.PrepOptions | None" = None,
        components: "lora_export.BundleComponents | None" = None,
    ) -> tuple[str, int, list[str]]:
        """Write a portable, trainer-ready bundle to disk (folder or zip).

        Returns ``(export_path, exported_count, drop_lines)`` where
        ``drop_lines`` are "name: reason" strings for pairs excluded by the
        IC-LoRA training-ready pipeline (empty for standard LoRA). Heavy ffmpeg
        + file work runs without the state lock — only the dataset/profile
        reads are locked. When ``profile_id`` is given, ``train_config.yaml`` is
        built from that saved profile (raising ``LoraEntityNotFoundError`` if
        it's gone); otherwise the trainer defaults are used. Raises
        ``LoraTransitionError`` if nothing usable remains to export.
        """
        dataset = self._snapshot_dataset(dataset_id)
        config = self.get_profile(profile_id).config if profile_id else TrainingConfig()
        clips = [
            c
            for c in dataset.clips
            if not c.deleted_at
            and c.triage != "holdout"
            and (include_rejected or c.triage != "reject")
        ]
        if not clips:
            raise LoraTransitionError(
                "Nothing to export: this dataset has no kept clips"
            )
        # IC-LoRA normalizes/validates pairs; default the trigger word from the
        # dataset so target captions are checked against it.
        options = prep_options or LoraDatasetPrep.PrepOptions(trigger_word=dataset.trigger_word)
        if options.trigger_word is None:
            options = replace(options, trigger_word=dataset.trigger_word)
        safe = lora_export.safe_dirname(dataset.name)
        tmp_root = Path(tempfile.mkdtemp(prefix="ltx-export-"))
        try:
            staging = tmp_root / safe
            staging.mkdir(parents=True)
            try:
                report = lora_export.build_bundle(
                    dataset=dataset,
                    clips=clips,
                    staging_dir=staging,
                    config=config,
                    processor=self._clip_processor,
                    options=options,
                    components=components,
                )
            except lora_export.BundleError as exc:
                raise LoraTransitionError(str(exc)) from exc
            if report.exported == 0:
                detail = "Nothing usable to export after validation."
                if report.dropped:
                    detail += " " + report.dropped[0].reason
                raise LoraTransitionError(detail)
            drop_lines = [f"{d.name}: {d.reason}" for d in report.dropped]
            if report.dropped:
                logger.info("lora.export drops dataset=%s\n%s", dataset_id, report.summary())
            if export_format == "zip":
                zip_path = Path(dest_path)
                if zip_path.suffix.lower() != ".zip":
                    zip_path = zip_path.with_suffix(".zip")
                lora_export.zip_dir(staging, zip_path)
                logger.info(
                    "lora.export ok dataset=%s zip=%s exported=%d dropped=%d",
                    dataset_id, zip_path, report.exported, len(report.dropped),
                )
                return str(zip_path), report.exported, drop_lines
            dest_root = Path(dest_path) / safe
            if dest_root.exists():
                shutil.rmtree(dest_root)
            shutil.move(str(staging), str(dest_root))
            logger.info(
                "lora.export ok dataset=%s dir=%s exported=%d dropped=%d",
                dataset_id, dest_root, report.exported, len(report.dropped),
            )
            return str(dest_root), report.exported, drop_lines
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    @with_state_lock
    def _snapshot_training(self, training_id: str) -> TrainingJob:
        """Deep-copied read so publishing can do heavy file IO without the lock."""
        return self._require_training(training_id).model_copy(deep=True)

    def _publication_context(
        self, training_id: str
    ) -> tuple[TrainingJob, PreprocessedDataset, LoraDataset]:
        """Resolve a completed run plus the preprocessed + source dataset it came
        from. Raises a transition error if the run isn't publishable yet."""
        job = self._snapshot_training(training_id)
        if job.status != "completed":
            raise LoraTransitionError("Only a completed run can be published")
        pre = self.get_preprocessed(job.preprocessed_id)
        if pre is None:
            raise LoraEntityNotFoundError("Preprocessed dataset not found for this run")
        dataset = self.get_dataset(pre.dataset_id)
        if dataset is None:
            raise LoraEntityNotFoundError("Source dataset not found for this run")
        return job, pre, dataset

    def publish_preview(
        self,
        training_id: str,
        *,
        platforms: list[lora_publish.PublishPlatform],
        meta: lora_publish.PublicationMeta | None,
        examples: list[lora_publish.PublicationExample],
    ) -> tuple[lora_publish.PublicationMeta, dict[str, str]]:
        """Render (without writing) the card for each requested platform.

        ``meta`` is ``None`` on the first call — we return the suggested fields so
        the wizard can prefill the form, plus the cards rendered from them.
        """
        job, pre, dataset = self._publication_context(training_id)
        resolved = meta or lora_publish.suggest_meta(job, dataset)
        planned = lora_publish.plan_examples(examples)
        card_examples = [card for _, card in planned]
        cards = {
            platform: lora_publish.build_model_card(
                platform=platform,
                job=job,
                preprocessed=pre,
                dataset=dataset,
                examples=card_examples,
                meta=resolved,
            )
            for platform in platforms
        }
        return resolved, cards

    def publish_export(
        self,
        training_id: str,
        *,
        dest_path: str,
        platforms: list[lora_publish.PublishPlatform],
        meta: lora_publish.PublicationMeta,
        examples: list[lora_publish.PublicationExample],
    ) -> tuple[str, dict[str, Any]]:
        """Write the publication bundle under ``dest_path``.

        Returns ``(publication_dir, manifest)``. Heavy file copies run without
        the state lock — only the entity reads are locked.
        """
        job, pre, dataset = self._publication_context(training_id)
        safe = lora_export.safe_dirname(meta.title)
        tmp_root = Path(tempfile.mkdtemp(prefix="ltx-publish-"))
        try:
            staging = tmp_root / safe
            staging.mkdir(parents=True)
            try:
                manifest = lora_publish.build_publication_bundle(
                    platforms=platforms,
                    job=job,
                    preprocessed=pre,
                    dataset=dataset,
                    examples=examples,
                    meta=meta,
                    lora_path=job.local_lora_path,
                    staging_dir=staging,
                )
            except lora_export.BundleError as exc:
                raise LoraTransitionError(str(exc)) from exc
            dest_root = Path(dest_path) / safe
            if dest_root.exists():
                shutil.rmtree(dest_root)
            shutil.move(str(staging), str(dest_root))
            logger.info(
                "lora.publish ok training=%s dir=%s platforms=%s examples=%d",
                training_id,
                dest_root,
                ",".join(platforms),
                manifest.get("exampleCount", 0),
            )
            return str(dest_root), manifest
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    def import_dataset(self, *, source_path: str) -> LoraDataset:
        """Re-create a dataset from a bundle folder or `.zip`.

        Clip files are copied into the app's storage so the imported
        dataset is independent of the source bundle. Captions, trigger
        word, IC-LoRA pairing, triage and origin are restored from the
        bundle manifest.
        """
        src = Path(source_path)
        tmp_root: Path | None = None
        try:
            if src.is_file() and src.suffix.lower() == ".zip":
                tmp_root = Path(tempfile.mkdtemp(prefix="ltx-import-"))
                try:
                    with zipfile.ZipFile(src) as zf:
                        lora_export.safe_extractall(zf, tmp_root)
                except zipfile.BadZipFile as exc:
                    raise LoraTransitionError(f"Not a valid .zip bundle: {exc}") from exc
                except lora_export.BundleError as exc:
                    raise LoraTransitionError(str(exc)) from exc
                root = lora_export.find_manifest_root(tmp_root)
            elif src.is_dir():
                root = lora_export.find_manifest_root(src)
            else:
                raise LoraTransitionError(
                    "Import source must be a bundle folder or a .zip file"
                )
            if root is None:
                raise LoraTransitionError(
                    f"No {lora_export.MANIFEST_NAME} found in the import source"
                )
            try:
                manifest = lora_export.read_manifest(root)
            except lora_export.BundleError as exc:
                raise LoraTransitionError(str(exc)) from exc

            new_id = uuid.uuid4().hex
            dest_dir = self._config.app_data_dir / "lora" / "imported" / new_id
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Pass 1: copy each clip file, mapping its relative bundle path
            # to the new absolute path in app storage.
            file_map: dict[str, str] = {}
            raw_clips = manifest["clips"]
            for entry in raw_clips:
                rel = entry.get("file")
                if not isinstance(rel, str):
                    continue
                try:
                    src_file = lora_export.resolve_bundle_member(root, rel)
                except lora_export.BundleError as exc:
                    raise LoraTransitionError(str(exc)) from exc
                if not src_file.is_file():
                    raise LoraTransitionError(
                        f"Bundle is missing a referenced clip file: {rel}"
                    )
                out = dest_dir / src_file.relative_to(root.resolve())
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, out)
                file_map[rel] = str(out)

            # Pass 2: build clips, remapping references to the new paths.
            clips: list[LoraClip] = []
            for entry in raw_clips:
                rel = entry.get("file")
                local = file_map.get(rel) if isinstance(rel, str) else None
                if local is None:
                    continue
                refs = [
                    file_map[r]
                    for r in entry.get("references", [])
                    if isinstance(r, str) and r in file_map
                ]
                duration = entry.get("durationSeconds")
                clips.append(
                    LoraClip(
                        id=uuid.uuid4().hex,
                        local_path=local,
                        caption=entry.get("caption") or "",
                        duration_seconds=duration
                        if isinstance(duration, (int, float))
                        else None,
                        reference_path=refs[0] if refs else None,
                        reference_paths=refs,
                        origin=lora_export.coerce_origin(entry.get("origin")),
                        triage=lora_export.coerce_triage(entry.get("triage")),
                    )
                )
            if not clips:
                raise LoraTransitionError("Bundle contained no usable clips")
            name = manifest.get("name")
            trigger = manifest.get("triggerWord")
            return self.create_dataset(
                name=name if isinstance(name, str) and name.strip() else "Imported dataset",
                dataset_type=lora_export.manifest_type(manifest),
                trigger_word=trigger if isinstance(trigger, str) else None,
                clips=clips,
            )
        finally:
            if tmp_root is not None:
                shutil.rmtree(tmp_root, ignore_errors=True)

    @with_state_lock
    def request_upload(
        self, dataset_id: str, *, provider: TrainerProvider | None = None
    ) -> LoraDataset:
        dataset = self._require_dataset(dataset_id)
        if dataset.archived_at is not None:
            raise LoraTransitionError("Restore this dataset before uploading it")
        if dataset.status not in ("draft", "upload_failed", "uploaded", "cancelled"):
            raise LoraTransitionError(
                f"Cannot upload dataset {dataset_id} in status {dataset.status!r}"
            )
        if not any(
            c.triage not in ("reject", "holdout") and not c.deleted_at
            for c in dataset.clips
        ):
            raise LoraTransitionError(
                "Cannot upload a dataset with no kept clips (all rejected or empty)"
            )
        dataset.keep_alive_until = None
        dataset.release_status = None
        dataset.release_error = None
        if provider is not None:
            dataset.provider = provider
        dataset.status = "uploading"
        dataset.error = None
        dataset.cancel_requested = False
        dataset.status_detail = None
        dataset.status_percent = None
        dataset.status_eta_seconds = None
        dataset.upload_started_at = _now_iso()
        dataset.upload_completed_at = None
        dataset.updated_at = _now_iso()
        self._persist_datasets_unlocked()
        self._wakeup_event.set()
        return dataset.model_copy(deep=True)

    @with_state_lock
    def start_training_pipeline(
        self,
        *,
        dataset_id: str,
        spec: AutoPipelineSpec,
        provider: TrainerProvider = "runpod",
        workspace_policy: WorkspacePolicy = "primary_cache",
        cache_volume_id: str | None = None,
    ) -> LoraDataset:
        """Kick off the one-click upload → preprocess → train pipeline.

        Stores the full intent on the dataset and flips it to ``uploading``; the
        reconciler auto-advances through preprocess and training from there (so
        the chain survives an app restart — no client-side orchestration).

        ``provider`` is persisted on the dataset (so the upload stage provisions
        the right backend) and carried onto the queued training run; it defaults
        to RunPod so omitting it preserves the original behavior exactly.
        """
        dataset = self._require_dataset(dataset_id)
        if dataset.archived_at is not None:
            raise LoraTransitionError("Restore this dataset before starting work")
        if dataset.status not in (
            "draft",
            "upload_failed",
            "uploaded",
            "cancelled",
            "gpu_selection_required",
        ):
            raise LoraTransitionError(
                f"Cannot start training pipeline for dataset {dataset_id} in "
                f"status {dataset.status!r}"
            )
        if not any(
            c.triage not in ("reject", "holdout") and not c.deleted_at
            for c in dataset.clips
        ):
            raise LoraTransitionError(
                "Cannot train a dataset with no kept clips (all rejected or empty)"
            )
        validate_resolution_buckets(spec.resolution_buckets)
        # Stamp the provider onto both the dataset (read by the upload stage,
        # before any target handle exists) and the carried training intent (read
        # when the run auto-starts after preprocessing), so the whole chain runs
        # on the chosen backend.
        spec = spec.model_copy(
            update={"training": spec.training.model_copy(update={"provider": provider})}
        )
        dataset.provider = provider
        dataset.workspace_policy = workspace_policy
        dataset.cache_volume_id = (
            cache_volume_id if workspace_policy == "primary_cache" else None
        )
        dataset.runpod_selection = spec.runpod_selection
        # If the run carries a trigger-word override, stamp it onto the dataset
        # too — preprocessing reads `dataset.trigger_word` to inject the token
        # into captions, so without this the override would land on the recorded
        # config but never reach the captions / auto-seeded validation prompts.
        trigger = (spec.training.config.trigger_word or "").strip() or None
        if trigger:
            dataset.trigger_word = trigger
        dataset.auto_pipeline = spec
        dataset.keep_alive_until = None
        dataset.release_status = None
        dataset.release_error = None
        if dataset.type == "ic_lora":
            staging = LoraDatasetPrep.options_for_resolution_buckets(
                spec.resolution_buckets, trigger_word=dataset.trigger_word
            )
            dataset.ic_staged_short_side = staging.short_side
            dataset.ic_staged_bucket_frames = staging.bucket_frames
        dataset.status = "uploading"
        dataset.error = None
        dataset.cancel_requested = False
        dataset.status_detail = None
        dataset.status_percent = None
        dataset.status_eta_seconds = None
        dataset.upload_started_at = _now_iso()
        dataset.upload_completed_at = None
        dataset.updated_at = _now_iso()
        self._persist_datasets_unlocked()
        self._wakeup_event.set()
        return dataset.model_copy(deep=True)

    @with_state_lock
    def consume_auto_pipeline(self, dataset_id: str) -> AutoPipelineSpec | None:
        """Atomically read + clear a dataset's pipeline intent (reconciler use)."""
        dataset = self._find_dataset(dataset_id)
        if dataset is None or dataset.auto_pipeline is None:
            return None
        spec = dataset.auto_pipeline
        dataset.auto_pipeline = None
        self._persist_datasets_unlocked()
        return spec

    @with_state_lock
    def consume_preprocess_auto_training(
        self, preprocessed_id: str
    ) -> PendingTraining | None:
        """Atomically read + clear a preprocessed dataset's pending training."""
        item = self._find_preprocessed(preprocessed_id)
        if item is None or item.auto_training is None:
            return None
        pending = item.auto_training
        item.auto_training = None
        self._persist_preprocessed_unlocked()
        return pending

    # ------------------------------------------------------------------
    # Preprocessing CRUD
    # ------------------------------------------------------------------

    @with_state_lock
    def create_preprocessing(
        self,
        *,
        dataset_id: str,
        resolution_buckets: str,
        with_audio: bool,
        auto_caption: bool,
        captioner_type: str,
        auto_training: PendingTraining | None = None,
        preset: TrainingPreset = "standard",
    ) -> PreprocessedDataset:
        dataset = self._require_dataset(dataset_id)
        if dataset.archived_at is not None:
            raise LoraTransitionError("Restore this dataset before preprocessing it")
        if dataset.status != "uploaded":
            raise LoraTransitionError(
                f"Dataset {dataset_id} must be uploaded before preprocessing "
                f"(status is {dataset.status!r})"
            )
        if any(
            item.dataset_id == dataset_id
            and item.status in ("pending", "captioning", "preprocessing")
            for item in self._preprocessed.items
        ):
            raise LoraTransitionError(
                f"Dataset {dataset_id} already has an active preprocessing job"
            )
        validate_resolution_buckets(resolution_buckets)
        dataset.keep_alive_until = None
        dataset.release_status = None
        dataset.release_error = None
        self._persist_datasets_unlocked()
        item = PreprocessedDataset(
            id=uuid.uuid4().hex,
            dataset_id=dataset_id,
            created_at=_now_iso(),
            status="pending",
            resolution_buckets=resolution_buckets,
            with_audio=with_audio,
            # The upstream caption script emits a flat video/caption manifest
            # and would destroy IC-LoRA input/output pairing metadata.
            auto_caption=auto_caption and dataset.type != "ic_lora",
            captioner_type="gemini_flash" if captioner_type == "gemini_flash" else "qwen_omni",
            preset=preset,
            trainer_repo_url=self.state.app_settings.lora_trainer_repo_url,
            trainer_repo_ref=self.state.app_settings.lora_trainer_repo_ref,
            auto_training=auto_training,
        )
        self._preprocessed.items.append(item)
        self._persist_preprocessed_unlocked()
        self._wakeup_event.set()
        return item.model_copy(deep=True)

    @with_state_lock
    def request_cancel_preprocessing(self, preprocessed_id: str) -> PreprocessedDataset:
        item = self._require_preprocessed(preprocessed_id)
        if item.status in ("ready", "failed", "cancelled"):
            raise LoraTransitionError(
                f"Preprocessing {preprocessed_id} already finished "
                f"({item.status!r})"
            )
        # A not-yet-submitted (pending) item has no remote job to kill, so we
        # can cancel immediately. An in-flight item (captioning/preprocessing)
        # keeps its status but sets `cancel_requested`: the reconciler observes
        # that on its next tick, terminates the remote job, and then calls
        # `mark_preprocess_cancelled`. Flipping to `cancelled` here would drop
        # the item out of `list_active_preprocessed` and orphan the remote job.
        if item.status == "pending" and item.target is None:
            item.status = "cancelled"
            item.cancel_requested = False
            item.completed_at = _now_iso()
        else:
            item.cancel_requested = True
            item.error = "Cancellation requested"
        self._persist_preprocessed_unlocked()
        self._wakeup_event.set()
        return item.model_copy(deep=True)

    @with_state_lock
    def mark_preprocess_cancelled(self, preprocessed_id: str) -> None:
        """Reconciler-owned: finalize a cancel after the remote job is gone.

        Sets `cancelled` and clears the cancel flag + transient error. Called
        once the runner has terminated the in-flight remote command (or when
        there was no remote job to terminate — e.g. a pod idle-stopped).
        """
        item = self._require_preprocessed(preprocessed_id)
        item.status = "cancelled"
        item.cancel_requested = False
        item.error = None
        item.completed_at = _now_iso()
        self._persist_preprocessed_unlocked()

    @with_state_lock
    def request_preprocess_resume(self, preprocessed_id: str) -> PreprocessedDataset:
        """Re-run a failed/cancelled preprocess reusing the uploaded workspace.

        Flips the item back to ``pending`` and clears the terminal error, but
        keeps the dataset's uploaded clips + pod handle and the
        ``captioning_completed`` flag, so the reconciler re-runs
        ``process_dataset.py`` (the heavy step that typically OOMs) without
        re-uploading or re-captioning. The runner submits a fresh remote
        command (the old ``remote_job_id`` is cleared).
        """
        item = self._require_preprocessed(preprocessed_id)
        if item.status not in ("failed", "cancelled"):
            raise LoraTransitionError(
                f"Preprocessing {preprocessed_id} is {item.status!r}; only a "
                "failed or cancelled run can be resumed"
            )
        item.status = "pending"
        item.error = None
        item.completed_at = None
        item.cancel_requested = False
        item.consecutive_failures = 0
        item.status_detail = None
        # Drop the spent remote job id so the runner submits a fresh command;
        # keep the pod handle (target.pod_id) so compute is reused.
        if item.target is not None:
            item.target = item.target.model_copy(update={"remote_job_id": None})
        self._persist_preprocessed_unlocked()
        self._wakeup_event.set()
        return item.model_copy(deep=True)

    @with_state_lock
    def request_preprocess_reset(self, preprocessed_id: str) -> PreprocessedDataset:
        """Clear a failed/cancelled preprocess's progress and re-run from scratch.

        Like resume, but also clears ``captioning_completed`` and sets
        ``reset_requested`` so the runner wipes the remote ``.precomputed``
        latent cache (and re-captions, if auto-caption is on) before re-running
        — a true fresh start, not a resume from cached state.
        """
        item = self._require_preprocessed(preprocessed_id)
        if item.status not in ("failed", "cancelled", "ready"):
            raise LoraTransitionError(
                f"Preprocessing {preprocessed_id} is {item.status!r}; only a "
                "finished run can be reset"
            )
        item.status = "pending"
        item.error = None
        item.completed_at = None
        item.cancel_requested = False
        item.consecutive_failures = 0
        item.status_detail = None
        item.captioning_completed = False
        item.effective_resolution_buckets = None
        item.remote_precomputed_dir = None
        item.reset_requested = True
        if item.target is not None:
            item.target = item.target.model_copy(update={"remote_job_id": None})
        self._persist_preprocessed_unlocked()
        self._wakeup_event.set()
        return item.model_copy(deep=True)

    @with_state_lock
    def clear_preprocess_reset_requested(self, preprocessed_id: str) -> None:
        """Reconciler-owned: clear the reset flag once the remote cache is wiped."""
        item = self._find_preprocessed(preprocessed_id)
        if item is not None and item.reset_requested:
            item.reset_requested = False
            self._persist_preprocessed_unlocked()

    @with_state_lock
    def set_preprocess_effective_buckets(
        self, preprocessed_id: str, effective_buckets: str
    ) -> None:
        """Record the bucket string preprocessing actually trained with.

        Called when an IC-LoRA low_vram run collapses a multi-bucket config to
        a single bucket (the trainer rejects multi-bucket + reference
        downscaling). Persisting the effective bucket lets the UI/run-summary
        show the real trained resolution instead of the user's uncollapsed list.
        """
        item = self._require_preprocessed(preprocessed_id)
        if item.effective_resolution_buckets == effective_buckets:
            return
        item.effective_resolution_buckets = effective_buckets
        self._persist_preprocessed_unlocked()

    @with_state_lock
    def delete_preprocessed(self, preprocessed_id: str) -> None:
        item = self._require_preprocessed(preprocessed_id)
        if item.status in ("captioning", "preprocessing"):
            raise LoraTransitionError(
                f"Cannot delete preprocessed {preprocessed_id} while running"
            )
        if any(j.preprocessed_id == preprocessed_id and j.status in ("pending", "running")
               for j in self._training.items):
            raise LoraTransitionError(
                f"Cannot delete preprocessed {preprocessed_id}; an active "
                "training job references it"
            )
        self._preprocessed.items.remove(item)
        self._persist_preprocessed_unlocked()

    # ------------------------------------------------------------------
    # Training CRUD
    # ------------------------------------------------------------------

    @with_state_lock
    def start_training(
        self,
        *,
        preprocessed_id: str,
        name: str,
        config: TrainingConfig,
        provider: TrainerProvider,
        description: str | None = None,
        gpu_type: str = "",
        gpu_vram_gb: int = 0,
        runpod_selection: RunpodSelection | None = None,
        workload_billing_started_at: str | None = None,
        captured_hourly_rate: float | None = None,
    ) -> TrainingJob:
        item = self._require_preprocessed(preprocessed_id)
        if item.status != "ready":
            raise LoraTransitionError(
                f"Preprocessed dataset {preprocessed_id} is not ready "
                f"(status is {item.status!r})"
            )
        if any(
            job.preprocessed_id == preprocessed_id
            and job.status in ("pending", "running")
            for job in self._training.items
        ):
            raise LoraTransitionError(
                f"Preprocessed dataset {preprocessed_id} already has an active "
                "training job"
            )
        dataset = self._find_dataset(item.dataset_id)
        if dataset is None:
            raise LoraTransitionError(
                f"Dataset {item.dataset_id} for preprocessing {preprocessed_id} "
                "no longer exists"
            )
        if dataset.archived_at is not None:
            raise LoraTransitionError("Restore the source dataset before training it")
        workspace_provider = (
            dataset.target.provider if dataset.target is not None else dataset.provider
        )
        if provider != workspace_provider:
            raise LoraTransitionError(
                f"Training provider {provider!r} does not match the "
                f"preprocessed workspace provider {workspace_provider!r}"
            )
        current_selection = dataset.runpod_selection
        if (
            provider == "runpod"
            and runpod_selection is not None
            and current_selection is not None
            and runpod_selection != current_selection
        ):
            same_persistent_workspace = (
                runpod_selection.workspace_policy == "primary_cache"
                and current_selection.workspace_policy == "primary_cache"
                and runpod_selection.volume_id is not None
                and runpod_selection.volume_id == current_selection.volume_id
                and runpod_selection.datacenter == current_selection.datacenter
            )
            if not same_persistent_workspace:
                raise LoraTransitionError(
                    "This preprocessed dataset can only train on its original "
                    "cache volume and region. Start a new full pipeline to use "
                    "another region."
                )
        if config.preset != item.preset:
            raise LoraTransitionError(
                f"Training preset {config.preset!r} does not match preprocessing "
                f"preset {item.preset!r}; preprocess again with the selected preset"
            )
        # `with_audio` is preprocessing-driven, never a profile knob: a run
        # can only train audio if its latents were precomputed with audio.
        resolved = config.model_copy(deep=True, update={"with_audio": item.with_audio})
        # `trigger_word` is dataset-driven (set in the New Dataset modal, used by
        # preprocessing to inject the token into captions). Snapshot it onto the
        # run config — like `with_audio` — so the Run summary records it and the
        # trained LoRA's registry entry uses the explicit token instead of the
        # name-derived fallback. An explicit per-run override (already in
        # `config.trigger_word` from `triggerWordOverride`) wins.
        if not resolved.trigger_word and dataset.trigger_word:
            resolved = resolved.model_copy(
                deep=True, update={"trigger_word": dataset.trigger_word}
            )
        dataset.keep_alive_until = None
        dataset.release_status = None
        dataset.release_error = None
        self._persist_datasets_unlocked()
        job = TrainingJob(
            id=uuid.uuid4().hex,
            preprocessed_id=preprocessed_id,
            name=name,
            description=(description or "").strip() or None,
            created_at=_now_iso(),
            status="pending",
            config=resolved,
            provider=provider,
            trainer_repo_url=item.trainer_repo_url,
            trainer_repo_ref=item.trainer_repo_ref,
            total_steps=resolved.steps,
            gpu_type=gpu_type,
            gpu_vram_gb=gpu_vram_gb,
            runpod_selection=runpod_selection,
            workload_billing_started_at=workload_billing_started_at,
            captured_hourly_rate=captured_hourly_rate,
            compute_rate_per_hr=captured_hourly_rate,
            pod_preparation_started_at=workload_billing_started_at,
        )
        self._training.items.append(job)
        self._persist_training_unlocked()
        self._wakeup_event.set()
        return job.model_copy(deep=True)

    @with_state_lock
    def request_cancel_training(self, training_id: str) -> TrainingJob:
        job = self._require_training(training_id)
        if job.status in ("completed", "failed", "cancelled"):
            raise LoraTransitionError(
                f"Training {training_id} already finished ({job.status!r})"
            )
        if job.status == "pending" and job.target is None:
            job.status = "cancelled"
            job.completed_at = _now_iso()
            self._close_training_billing_unlocked(job, job.completed_at)
        else:
            job.cancel_requested = True
        self._persist_training_unlocked()
        self._wakeup_event.set()
        return job.model_copy(deep=True)

    @with_state_lock
    def delete_training(self, training_id: str) -> None:
        job = self._require_training(training_id)
        if job.status in ("pending", "running"):
            raise LoraTransitionError(
                f"Cannot delete training {training_id} while active; cancel "
                "it first"
            )
        self._training.items.remove(job)
        self._persist_training_unlocked()
        # Tear down any attached library example media so it doesn't orphan.
        if job.example_path:
            try:
                Path(job.example_path).unlink(missing_ok=True)
            except OSError:
                pass

    @with_state_lock
    def archive_training(self, training_id: str) -> TrainingJob:
        job = self._require_training(training_id)
        if job.archived_at is not None:
            return job.model_copy(deep=True)
        if (
            job.status not in ("completed", "failed", "cancelled")
            or job.cancel_requested
            or job.redownload_requested
            or job.reset_requested
        ):
            raise LoraTransitionError(
                f"Cannot archive training {training_id} while work is active"
            )
        job.archived_at = _now_iso()
        self._persist_training_unlocked()
        return job.model_copy(deep=True)

    @with_state_lock
    def unarchive_training(self, training_id: str) -> TrainingJob:
        job = self._require_training(training_id)
        job.archived_at = None
        self._persist_training_unlocked()
        return job.model_copy(deep=True)

    @with_state_lock
    def update_training_meta(
        self,
        training_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> TrainingJob:
        """Patch editable library metadata on a completed training job.

        ``name`` (non-blank when provided) and ``description`` are display-only
        metadata surfaced in the LoRA Library; the weights file and run folder
        are untouched. At least one field must be supplied.
        """
        job = self._require_training(training_id)
        if name is None and description is None:
            raise LoraTransitionError("Provide at least one field to update")
        if name is not None and not name.strip():
            raise LoraTransitionError("LoRA name is required")
        updates: dict[str, object] = {}
        if name is not None:
            updates["name"] = name.strip()
        if description is not None:
            updates["description"] = description.strip() or None
        updated = job.model_copy(update=updates)
        self._training.items = [updated if j.id == job.id else j for j in self._training.items]
        self._persist_training_unlocked()
        return updated.model_copy(deep=True)

    @with_state_lock
    def set_training_example_path(self, training_id: str, *, example_path: str) -> TrainingJob:
        """Record the on-disk path of an example media file attached to a job.

        State-only: the caller (inference handler) owns copying the file into
        place and tearing down the previous one. Persists and returns the job.
        """
        job = self._require_training(training_id)
        updated = job.model_copy(update={"example_path": example_path})
        self._training.items = [updated if j.id == job.id else j for j in self._training.items]
        self._persist_training_unlocked()
        return updated.model_copy(deep=True)

    @with_state_lock
    def clear_training_example_path(self, training_id: str) -> TrainingJob | None:
        """Unset the example media path on a job (state-only). Returns the
        updated job, or None if the job no longer exists (idempotent clear)."""
        job = self._require_training(training_id)
        if job.example_path is None:
            return job.model_copy(deep=True)
        updated = job.model_copy(update={"example_path": None})
        self._training.items = [updated if j.id == job.id else j for j in self._training.items]
        self._persist_training_unlocked()
        return updated.model_copy(deep=True)

    # ------------------------------------------------------------------
    # Reconciler-facing reads + transitions
    # ------------------------------------------------------------------

    @with_state_lock
    def list_datasets_to_upload(self) -> list[LoraDataset]:
        return [
            d.model_copy(deep=True)
            for d in self._datasets.datasets
            if d.status == "uploading" and d.archived_at is None
        ]

    @with_state_lock
    def list_active_preprocessed(self) -> list[PreprocessedDataset]:
        return [
            p.model_copy(deep=True)
            for p in self._preprocessed.items
            if p.status in ("pending", "captioning", "preprocessing")
            and (
                (dataset := self._find_dataset(p.dataset_id)) is not None
                and dataset.archived_at is None
            )
        ]

    @with_state_lock
    def list_active_training(self) -> list[TrainingJob]:
        return [
            j.model_copy(deep=True)
            for j in self._training.items
            if j.status in ("pending", "running", "gpu_selection_required")
            and j.archived_at is None
        ]

    @with_state_lock
    def list_active_datasets(self) -> list[LoraDataset]:
        return [
            dataset.model_copy(deep=True)
            for dataset in self._datasets.datasets
            if dataset.archived_at is None
            and (dataset.target is not None
            or dataset.status in ("uploading", "gpu_selection_required")
            )
        ]

    @with_state_lock
    def mark_saved_model_ready(
        self,
        *,
        volume_id: str,
        fingerprint: str,
        estimated_download_bytes: int | None,
    ) -> None:
        if not volume_id:
            return
        existing = next(
            (item for item in self._saved_models.volumes if item.volume_id == volume_id),
            None,
        )
        record = SavedModelVolumeMetadata(
            volume_id=volume_id,
            fingerprint=fingerprint,
            status="ready",
            estimated_download_bytes=estimated_download_bytes,
            updated_at=_now_iso(),
        )
        if existing is None:
            self._saved_models.volumes.append(record)
        else:
            self._saved_models.volumes[self._saved_models.volumes.index(existing)] = record
        self._persist_saved_models_unlocked()

    @with_state_lock
    def saved_model_readiness(
        self,
        *,
        volume_id: str | None,
        fingerprint: str,
        estimated_download_bytes: int | None,
    ) -> tuple[Literal["ready", "missing", "unknown"], int | None]:
        if not volume_id:
            return "missing", estimated_download_bytes
        existing = next(
            (item for item in self._saved_models.volumes if item.volume_id == volume_id),
            None,
        )
        if existing is None:
            return "unknown", estimated_download_bytes
        if existing.fingerprint != fingerprint:
            return "missing", estimated_download_bytes
        return existing.status, (
            0 if existing.status == "ready" else existing.estimated_download_bytes
        )

    @with_state_lock
    def reconcile_saved_model_volumes(self, volume_ids: set[str]) -> None:
        kept = [
            item for item in self._saved_models.volumes if item.volume_id in volume_ids
        ]
        if len(kept) != len(self._saved_models.volumes):
            self._saved_models.volumes = kept
            self._persist_saved_models_unlocked()

    @with_state_lock
    def remove_saved_model_volume(self, volume_id: str) -> None:
        kept = [
            item for item in self._saved_models.volumes if item.volume_id != volume_id
        ]
        if len(kept) != len(self._saved_models.volumes):
            self._saved_models.volumes = kept
            self._persist_saved_models_unlocked()

    @with_state_lock
    def require_dataset_gpu_selection(self, dataset_id: str, error: str) -> None:
        dataset = self._require_dataset(dataset_id)
        dataset.status = "gpu_selection_required"
        dataset.error = error
        dataset.status_detail = "Select another GPU"
        dataset.updated_at = _now_iso()
        self._persist_datasets_unlocked()

    @with_state_lock
    def reselect_dataset(
        self, dataset_id: str, selection: RunpodSelection
    ) -> LoraDataset:
        dataset = self._require_dataset(dataset_id)
        if dataset.status != "gpu_selection_required" or dataset.auto_pipeline is None:
            raise LoraTransitionError("Dataset pipeline is not waiting for GPU selection")
        dataset.auto_pipeline.runpod_selection = selection
        dataset.auto_pipeline.training.runpod_selection = selection
        dataset.auto_pipeline.training.gpu_type = selection.gpu_type
        dataset.auto_pipeline.training.gpu_vram_gb = selection.gpu_vram_gb
        dataset.runpod_selection = selection
        dataset.workspace_policy = selection.workspace_policy
        dataset.cache_volume_id = selection.volume_id
        dataset.status = "uploading"
        dataset.error = None
        dataset.status_detail = None
        dataset.upload_started_at = _now_iso()
        dataset.upload_completed_at = None
        self._persist_datasets_unlocked()
        self._wakeup_event.set()
        return dataset.model_copy(deep=True)

    @with_state_lock
    def require_training_gpu_selection(self, training_id: str, error: str) -> None:
        job = self._require_training(training_id)
        job.status = "gpu_selection_required"
        job.error = error
        job.target = None
        self._persist_training_unlocked()

    @with_state_lock
    def reselect_training(
        self, training_id: str, selection: RunpodSelection
    ) -> TrainingJob:
        job = self._require_training(training_id)
        if job.status != "gpu_selection_required":
            raise LoraTransitionError("Training is not waiting for GPU selection")
        preprocessed = self._find_preprocessed(job.preprocessed_id)
        dataset = (
            self._find_dataset(preprocessed.dataset_id)
            if preprocessed is not None
            else None
        )
        current = dataset.runpod_selection if dataset is not None else None
        if current is not None and selection != current:
            same_persistent_workspace = (
                selection.workspace_policy == "primary_cache"
                and current.workspace_policy == "primary_cache"
                and selection.volume_id is not None
                and selection.volume_id == current.volume_id
                and selection.datacenter == current.datacenter
            )
            if not same_persistent_workspace:
                raise LoraTransitionError(
                    "This preprocessed dataset can only continue on its original "
                    "cache volume and region."
                )
        job.runpod_selection = selection
        job.gpu_type = selection.gpu_type
        job.gpu_vram_gb = selection.gpu_vram_gb
        job.status = "pending"
        job.error = None
        job.target = None
        self._persist_training_unlocked()
        self._wakeup_event.set()
        return job.model_copy(deep=True)

    @with_state_lock
    def cache_volume_dependencies(self, volume_id: str) -> tuple[list[str], list[str]]:
        """Return (active work, recovery artifacts) tied to a cache volume.

        Volume relocation/deletion is a control-plane operation.  Keeping this
        dependency check beside the durable ledgers makes the guard atomic and
        prevents provider code from guessing which remote paths remain useful.
        """
        dataset_ids = {
            d.id
            for d in self._datasets.datasets
            if d.provider == "runpod" and d.cache_volume_id == volume_id
        }
        preprocessed_ids = {
            p.id for p in self._preprocessed.items if p.dataset_id in dataset_ids
        }
        active: list[str] = []
        recovery: list[str] = []
        for dataset in self._datasets.datasets:
            if dataset.id not in dataset_ids:
                continue
            if dataset.status == "uploading":
                active.append(f"dataset:{dataset.id}")
            if dataset.remote_dataset_dir or dataset.target is not None:
                recovery.append(f"dataset:{dataset.id}")
        for item in self._preprocessed.items:
            if item.id not in preprocessed_ids:
                continue
            if item.status in ("pending", "captioning", "preprocessing"):
                active.append(f"preprocess:{item.id}")
            if item.remote_precomputed_dir or item.target is not None:
                recovery.append(f"preprocess:{item.id}")
        for job in self._training.items:
            if job.preprocessed_id not in preprocessed_ids:
                continue
            if job.status in ("pending", "running"):
                active.append(f"training:{job.id}")
            if (
                job.remote_output_dir
                or job.target is not None
                or job.redownload_requested
                or job.reset_requested
            ):
                recovery.append(f"training:{job.id}")
        return active, recovery

    @with_state_lock
    def current_hf_token(self) -> str | None:
        """Best-effort HuggingFace token for remote model downloads.

        Returns the current OAuth access token when authenticated and
        unexpired, else ``None`` so provisioning downloads proceed
        anonymously (sufficient for public weights). Never raises — a
        missing token must not block the trainer auto-provision path.
        """
        match self.state.hf_auth_state:
            case HfAuthenticated(access_token=token, expires_at=exp):
                return token if time.time() <= exp else None
            case _:
                return None

    @with_state_lock
    def get_dataset(self, dataset_id: str) -> LoraDataset | None:
        found = self._find_dataset(dataset_id)
        return None if found is None else found.model_copy(deep=True)

    @with_state_lock
    def get_preprocessed(self, preprocessed_id: str) -> PreprocessedDataset | None:
        found = self._find_preprocessed(preprocessed_id)
        return None if found is None else found.model_copy(deep=True)

    @with_state_lock
    def get_training(self, training_id: str) -> TrainingJob | None:
        found = self._find_training(training_id)
        return None if found is None else found.model_copy(deep=True)

    @with_state_lock
    def mark_dataset_uploaded(
        self, dataset_id: str, *, remote_dataset_dir: str, handle: TargetHandle
    ) -> None:
        dataset = self._require_dataset(dataset_id)
        dataset.status = "uploaded"
        dataset.remote_dataset_dir = remote_dataset_dir
        dataset.target = handle
        dataset.last_pod_id = handle.pod_id or dataset.last_pod_id
        dataset.error = None
        dataset.status_detail = None
        dataset.status_percent = None
        dataset.status_eta_seconds = None
        dataset.upload_completed_at = _now_iso()
        dataset.updated_at = _now_iso()
        dataset.last_active_at = _now_iso()
        self._persist_datasets_unlocked()

    @with_state_lock
    def set_ic_staging_envelope(
        self, dataset_id: str, *, short_side: int, bucket_frames: int
    ) -> None:
        dataset = self._require_dataset(dataset_id)
        dataset.ic_staged_short_side = short_side
        dataset.ic_staged_bucket_frames = bucket_frames
        dataset.updated_at = _now_iso()
        self._persist_datasets_unlocked()

    @with_state_lock
    def set_dataset_status_detail(
        self,
        dataset_id: str,
        detail: str | None,
        *,
        percent: int | None = None,
        eta_seconds: int | None = None,
    ) -> None:
        """Update the upload sub-stage + optional structured progress shown on
        the card (percent/ETA come from the remote setup log's download bar)."""
        dataset = self._find_dataset(dataset_id)
        if dataset is None:
            return
        dataset.status_detail = detail
        dataset.status_percent = percent
        dataset.status_eta_seconds = eta_seconds
        self._persist_datasets_unlocked()

    @with_state_lock
    def touch_dataset_activity(self, dataset_id: str) -> None:
        """Mark the dataset's pod as used now (resets the idle-stop clock)."""
        dataset = self._find_dataset(dataset_id)
        if dataset is None:
            return
        dataset.last_active_at = _now_iso()
        dataset.keep_alive_until = None
        dataset.release_status = None
        dataset.release_error = None
        self._persist_datasets_unlocked()

    @with_state_lock
    def set_pipeline_billing_start(
        self, dataset_id: str, *, started_at: str, hourly_rate: float | None
    ) -> None:
        """Persist one-click attribution before its pending run exists."""
        dataset = self._find_dataset(dataset_id)
        if dataset is None or dataset.auto_pipeline is None:
            return
        pending = dataset.auto_pipeline.training
        if pending.workload_billing_started_at is None:
            pending.workload_billing_started_at = started_at
        if pending.captured_hourly_rate is None and hourly_rate is not None:
            pending.captured_hourly_rate = hourly_rate
        self._persist_datasets_unlocked()

    @with_state_lock
    def begin_training_billing(
        self, training_id: str, *, started_at: str, hourly_rate: float | None
    ) -> None:
        job = self._find_training(training_id)
        if job is None:
            return
        if job.workload_billing_started_at is None:
            job.workload_billing_started_at = started_at
            job.pod_preparation_started_at = started_at
        if job.captured_hourly_rate is None and hourly_rate is not None:
            job.captured_hourly_rate = hourly_rate
            job.compute_rate_per_hr = hourly_rate
        self._persist_training_unlocked()

    @with_state_lock
    def mark_training_setup_started(self, training_id: str) -> None:
        job = self._find_training(training_id)
        if job is None:
            return
        now = _now_iso()
        job.training_setup_started_at = job.training_setup_started_at or now
        job.pod_preparation_ended_at = job.pod_preparation_ended_at or now
        self._persist_training_unlocked()

    @with_state_lock
    def extend_workspace_keep_alive(
        self, pod_id: str, *, minutes: int
    ) -> LoraDataset:
        datasets = [
            dataset
            for dataset in self._datasets.datasets
            if dataset.target is not None and dataset.target.pod_id == pod_id
        ]
        if not datasets:
            raise LoraEntityNotFoundError(f"RunPod workspace not found: {pod_id}")
        if any(
            dataset.status in ("uploading", "gpu_selection_required")
            for dataset in datasets
        ):
            raise LoraTransitionError("This workspace is currently in use")
        until = (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()
        for dataset in datasets:
            dataset.keep_alive_until = until
            dataset.release_status = "scheduled"
            dataset.release_error = None
        self._persist_datasets_unlocked()
        return datasets[0].model_copy(deep=True)

    @with_state_lock
    def mark_workspace_release_attempt(
        self, pod_id: str, *, error: str | None
    ) -> None:
        now = _now_iso()
        changed = False
        for dataset in self._datasets.datasets:
            if dataset.target is None or dataset.target.pod_id != pod_id:
                continue
            dataset.release_attempted_at = now
            dataset.release_status = "failed" if error else "releasing"
            dataset.release_error = error
            changed = True
        if changed:
            self._persist_datasets_unlocked()

    @with_state_lock
    def set_dataset_runpod_selection(
        self, dataset_id: str, selection: RunpodSelection
    ) -> None:
        dataset = self._require_dataset(dataset_id)
        dataset.runpod_selection = selection
        dataset.workspace_policy = selection.workspace_policy
        dataset.cache_volume_id = selection.volume_id
        self._persist_datasets_unlocked()

    @with_state_lock
    def set_dataset_target(self, dataset_id: str, handle: TargetHandle) -> None:
        """Persist a (re-)acquired pod handle, e.g. after idle auto-stop.

        Also refreshes the activity clock since acquiring a pod is work.
        """
        dataset = self._find_dataset(dataset_id)
        if dataset is None:
            return
        dataset.target = handle
        dataset.last_pod_id = handle.pod_id or dataset.last_pod_id
        dataset.last_active_at = _now_iso()
        self._persist_datasets_unlocked()

    @with_state_lock
    def mark_dataset_pod_stopped(self, dataset_id: str) -> None:
        """Clear the pod id after an idle auto-stop teardown.

        Keeps the provider + remote dir + uploaded status so the next
        preprocess/training tick re-acquires a pod (re-mounting the
        network volume that preserves the install + cached latents).
        """
        dataset = self._find_dataset(dataset_id)
        if dataset is None or dataset.target is None:
            return
        dataset.target = dataset.target.model_copy(update={"pod_id": None})
        self._persist_datasets_unlocked()

    @with_state_lock
    def mark_pod_stopped(self, pod_id: str) -> None:
        """Clear every ledger handle that references a released shared pod."""
        datasets_changed = False
        preprocessed_changed = False
        training_changed = False
        for dataset in self._datasets.datasets:
            if dataset.target is None or dataset.target.pod_id != pod_id:
                continue
            dataset.target = dataset.target.model_copy(update={"pod_id": None})
            dataset.keep_alive_until = None
            dataset.release_status = "released"
            dataset.release_error = None
            datasets_changed = True
        for item in self._preprocessed.items:
            if item.target is None or item.target.pod_id != pod_id:
                continue
            item.target = item.target.model_copy(update={"pod_id": None})
            preprocessed_changed = True
        for job in self._training.items:
            if job.target is None or job.target.pod_id != pod_id:
                continue
            job.target = job.target.model_copy(update={"pod_id": None})
            training_changed = True
        if datasets_changed:
            self._persist_datasets_unlocked()
        if preprocessed_changed:
            self._persist_preprocessed_unlocked()
        if training_changed:
            self._persist_training_unlocked()

    @with_state_lock
    def list_datasets_with_pod(self) -> list[LoraDataset]:
        """Datasets that hold a live pod — idle-stop candidates.

        Includes `upload_failed` so a pod left running by a failed upload
        (e.g. a staging error after the pod was created) still gets reclaimed
        rather than billing forever.
        """
        return [
            d.model_copy(deep=True)
            for d in self._datasets.datasets
            if d.status in ("uploaded", "upload_failed")
            and d.target is not None
            and d.target.pod_id is not None
        ]

    @with_state_lock
    def fail_dataset_upload(self, dataset_id: str, error: str) -> None:
        dataset = self._require_dataset(dataset_id)
        dataset.status = "upload_failed"
        dataset.error = error
        dataset.status_detail = None
        dataset.status_percent = None
        dataset.status_eta_seconds = None
        dataset.updated_at = _now_iso()
        self._persist_datasets_unlocked()

    @with_state_lock
    def request_cancel_upload(self, dataset_id: str) -> LoraDataset:
        """Request cancellation of an in-progress upload.

        Mirrors `request_cancel_preprocessing`: if no pod was acquired yet
        (target is None) there's nothing to reclaim, so finalize immediately;
        otherwise set `cancel_requested` and let the reconciler observe it at
        the next upload sub-phase boundary, release the pod, then call
        `mark_dataset_upload_cancelled`. Flipping to `cancelled` here would drop
        the dataset out of `list_datasets_to_upload` mid-blocking-call and orphan
        the pod the runner is still inside of.
        """
        dataset = self._require_dataset(dataset_id)
        if dataset.status != "uploading":
            raise LoraTransitionError(
                f"Cannot cancel upload for dataset {dataset_id} in status "
                f"{dataset.status!r}"
            )
        if dataset.target is None or dataset.target.pod_id is None:
            dataset.status = "cancelled"
            dataset.cancel_requested = False
            dataset.error = None
            dataset.status_detail = None
            dataset.status_percent = None
            dataset.status_eta_seconds = None
            dataset.updated_at = _now_iso()
        else:
            dataset.cancel_requested = True
            dataset.error = "Cancellation requested"
        self._persist_datasets_unlocked()
        self._wakeup_event.set()
        return dataset.model_copy(deep=True)

    @with_state_lock
    def mark_dataset_upload_cancelled(self, dataset_id: str) -> None:
        """Reconciler-owned: finalize an upload cancel after the pod is released.

        Sets `cancelled`, clears the cancel flag + transient error + progress,
        and drops the pod id so the released pod isn't treated as live compute.
        """
        dataset = self._find_dataset(dataset_id)
        if dataset is None:
            return
        dataset.status = "cancelled"
        dataset.cancel_requested = False
        dataset.error = None
        dataset.status_detail = None
        dataset.status_percent = None
        dataset.status_eta_seconds = None
        if dataset.target is not None:
            dataset.target = dataset.target.model_copy(update={"pod_id": None})
        dataset.updated_at = _now_iso()
        self._persist_datasets_unlocked()

    @with_state_lock
    def is_dataset_cancel_requested(self, dataset_id: str) -> bool:
        """Snapshot read the reconciler uses between upload sub-phases.

        A deep-copy-style read under the lock keeps this safe against a
        concurrent `request_cancel_upload` landing mid-blocking-upload.
        """
        dataset = self._find_dataset(dataset_id)
        return dataset is not None and dataset.cancel_requested

    # ------------------------------------------------------------------
    # Reconciler outer-guard: transient-failure accounting
    # ------------------------------------------------------------------

    @with_state_lock
    def record_reconcile_success(
        self, kind: ReconcileEntityKind, entity_id: str
    ) -> None:
        """Reset transient-failure bookkeeping after a clean reconcile tick.

        For preprocess/training we also clear `status_detail` — it's only ever
        set by the retry path below, so a clean tick means the retry message is
        stale. For datasets we leave `status_detail` alone: the upload sub-stage
        ("Creating GPU pod…", "Uploading clips…", …) owns it and overwrites it
        each phase. No-op for entities that are terminal or already clean.
        """
        if kind == "dataset":
            ds = self._find_dataset(entity_id)
            if ds is not None and ds.consecutive_failures:
                ds.consecutive_failures = 0
                self._persist_datasets_unlocked()
            return
        if kind == "preprocess":
            item = self._find_preprocessed(entity_id)
            if item is None or item.status in ("ready", "failed", "cancelled"):
                return
            if item.consecutive_failures or item.status_detail:
                item.consecutive_failures = 0
                item.status_detail = None
                self._persist_preprocessed_unlocked()
            return
        job = self._find_training(entity_id)
        if job is None or job.status in ("completed", "failed", "cancelled"):
            return
        if job.consecutive_failures or (
            job.status_detail is not None
            and job.status_detail.startswith("Retrying after error:")
        ):
            job.consecutive_failures = 0
            job.status_detail = None
            self._persist_training_unlocked()

    @with_state_lock
    def record_transient_failure(
        self, kind: ReconcileEntityKind, entity_id: str, detail: str
    ) -> bool:
        """Record a retryable reconcile failure on an entity.

        Surfaces ``Retrying after error: <detail>`` on the card via
        ``status_detail`` and bumps ``consecutive_failures``; once the budget is
        exhausted, escalates to a terminal ``failed``/``upload_failed`` (so a
        stuck entity can't retry forever). Returns ``True`` once escalated.
        """
        message = f"Retrying after error: {detail}"
        if kind == "dataset":
            ds = self._require_dataset(entity_id)
            ds.consecutive_failures += 1
            if ds.consecutive_failures >= _TRANSIENT_FAILURE_BUDGET:
                ds.status = "upload_failed"
                ds.error = f"Repeated failures: {detail}"
                ds.status_detail = None
                ds.status_percent = None
                ds.status_eta_seconds = None
                ds.updated_at = _now_iso()
                self._persist_datasets_unlocked()
                return True
            ds.status_detail = message
            ds.updated_at = _now_iso()
            self._persist_datasets_unlocked()
            return False
        if kind == "preprocess":
            item = self._require_preprocessed(entity_id)
            item.consecutive_failures += 1
            if item.consecutive_failures >= _TRANSIENT_FAILURE_BUDGET:
                item.status = "failed"
                item.error = f"Repeated failures: {detail}"
                item.status_detail = None
                item.completed_at = _now_iso()
                self._persist_preprocessed_unlocked()
                return True
            item.status_detail = message
            self._persist_preprocessed_unlocked()
            return False
        job = self._require_training(entity_id)
        job.consecutive_failures += 1
        if job.consecutive_failures >= _TRANSIENT_FAILURE_BUDGET:
            job.status = "failed"
            job.error = f"Repeated failures: {detail}"
            job.status_detail = None
            job.completed_at = _now_iso()
            job.redownload_requested = False
            self._persist_training_unlocked()
            return True
        job.status_detail = message
        self._persist_training_unlocked()
        return False

    @with_state_lock
    def set_preprocess_captioning(
        self, preprocessed_id: str, *, handle: TargetHandle, remote_job_id: str
    ) -> None:
        item = self._require_preprocessed(preprocessed_id)
        item.status = "captioning"
        item.target = handle.model_copy(update={"remote_job_id": remote_job_id})
        item.started_at = item.started_at or _now_iso()
        self._persist_preprocessed_unlocked()

    @with_state_lock
    def set_preprocess_processing(
        self, preprocessed_id: str, *, handle: TargetHandle, remote_job_id: str
    ) -> None:
        item = self._require_preprocessed(preprocessed_id)
        item.status = "preprocessing"
        item.target = handle.model_copy(update={"remote_job_id": remote_job_id})
        item.started_at = item.started_at or _now_iso()
        # Captioning (if it ran) is now done — record it so a later resume
        # skips re-captioning and goes straight back to re-running
        # `process_dataset.py`.
        item.captioning_completed = True
        self._persist_preprocessed_unlocked()

    @with_state_lock
    def mark_preprocess_ready(
        self, preprocessed_id: str, *, remote_precomputed_dir: str
    ) -> None:
        item = self._require_preprocessed(preprocessed_id)
        item.status = "ready"
        item.remote_precomputed_dir = remote_precomputed_dir
        item.completed_at = _now_iso()
        item.error = None
        self._persist_preprocessed_unlocked()

    @with_state_lock
    def fail_preprocess(self, preprocessed_id: str, error: str) -> None:
        item = self._require_preprocessed(preprocessed_id)
        item.status = "failed"
        item.error = error
        item.completed_at = _now_iso()
        self._persist_preprocessed_unlocked()

    @with_state_lock
    def set_training_running(
        self,
        training_id: str,
        *,
        handle: TargetHandle,
        remote_job_id: str,
        remote_output_dir: str,
    ) -> None:
        job = self._require_training(training_id)
        job.status = "running"
        job.target = handle.model_copy(update={"remote_job_id": remote_job_id})
        job.last_pod_id = handle.pod_id or job.last_pod_id
        job.remote_output_dir = remote_output_dir
        job.started_at = job.started_at or _now_iso()
        job.training_setup_ended_at = job.training_setup_ended_at or job.started_at
        self._persist_training_unlocked()

    @with_state_lock
    def update_training_progress(
        self,
        training_id: str,
        *,
        current_step: int | None,
        total_steps: int | None,
        eta_seconds: int | None = None,
    ) -> None:
        job = self._find_training(training_id)
        if job is None:
            return
        if current_step is not None:
            job.current_step = current_step
            # First real step → end of the silent setup phase. Stamp once so the
            # run summary can split setup time from training time.
            if current_step > 0 and job.first_step_at is None:
                job.first_step_at = _now_iso()
                job.training_steps_started_at = job.first_step_at
        if total_steps is not None:
            job.total_steps = total_steps
        if eta_seconds is not None:
            job.eta_seconds = eta_seconds
        if current_step is not None and current_step > 0:
            job.status_detail = "Training"
        self._persist_training_unlocked()

    @with_state_lock
    def set_training_status_detail(self, training_id: str, detail: str | None) -> None:
        """Surface the current setup/download phase before step progress exists."""
        job = self._find_training(training_id)
        if job is None or job.status in ("completed", "failed", "cancelled"):
            return
        if job.status_detail == detail:
            return
        job.status_detail = detail
        self._persist_training_unlocked()

    @with_state_lock
    def set_training_compute_rate(
        self, training_id: str, rate_per_hr: float
    ) -> None:
        job = self._find_training(training_id)
        if job is None or rate_per_hr < 0:
            return
        if job.compute_rate_per_hr == rate_per_hr:
            return
        job.compute_rate_per_hr = rate_per_hr
        if job.captured_hourly_rate is None:
            job.captured_hourly_rate = rate_per_hr
        self._persist_training_unlocked()

    @with_state_lock
    def update_training_checkpoint_step(
        self, training_id: str, step: int
    ) -> None:
        """Record the highest adapter checkpoint step seen so far.

        Persisted during polling so it survives a reconciler restart; the
        download/redownload path falls back to it when the remote checkpoint
        listing is unavailable. Only writes when the step advances, to avoid a
        disk write on every poll tick.
        """
        job = self._find_training(training_id)
        if job is None:
            return
        if step > (job.latest_checkpoint_step or -1):
            job.latest_checkpoint_step = step
            self._persist_training_unlocked()

    @with_state_lock
    def set_validation_sample_refs(
        self, training_id: str, refs: list[ValidationSampleRef]
    ) -> None:
        """Record the validation samples configured for a run (for the feed).

        Set once at training start so later polls can map a downloaded
        validation artifact (by 1-based sample index) back to its prompt/source
        without recomputing the sample list.
        """
        job = self._find_training(training_id)
        if job is None:
            return
        job.validation_sample_refs = list(refs)
        self._persist_training_unlocked()

    @with_state_lock
    def append_validation_feed_items(
        self, training_id: str, items: list[ValidationFeedItem]
    ) -> None:
        """Append newly downloaded validation samples to the feed (bounded).

        Dedupes by ``(step, sample_index, extension)`` so a reconciler re-poll that
        re-lists an already-downloaded sample doesn't double-add, and trims to
        the last `VALIDATION_FEED_MAX_ITEMS` so a long run's job record stays
        bounded.
        """
        if not items:
            return
        job = self._find_training(training_id)
        if job is None:
            return
        existing = {
            (i.step, i.sample_index, i.extension) for i in job.validation_feed
        }
        for item in items:
            key = (item.step, item.sample_index, item.extension)
            if key in existing:
                continue
            existing.add(key)
            job.validation_feed.append(item)
        # Keep the most recent items (sort by step then sample_index, take tail).
        job.validation_feed.sort(
            key=lambda i: (i.step, i.sample_index, i.extension)
        )
        if len(job.validation_feed) > VALIDATION_FEED_MAX_ITEMS:
            job.validation_feed = job.validation_feed[-VALIDATION_FEED_MAX_ITEMS:]
        self._persist_training_unlocked()

    @with_state_lock
    def append_checkpoint_artifacts(
        self, training_id: str, items: list[CheckpointArtifact]
    ) -> None:
        """Append newly downloaded adapter checkpoints to the run (bounded).

        Dedupes by `step` so a reconciler re-poll can't double-add, and trims
        to the run's configured `checkpoint_keep_last_n` — mirroring the
        remote retention the user set — by deleting the pruned local files so
        disk doesn't accumulate stale adapters. The newest checkpoints are
        kept (highest steps).
        """
        if not items:
            return
        job = self._find_training(training_id)
        if job is None:
            return
        existing = {c.step for c in job.checkpoints}
        for item in items:
            if item.step in existing:
                continue
            existing.add(item.step)
            job.checkpoints.append(item)
        job.checkpoints.sort(key=lambda c: c.step)
        keep = max(1, job.config.checkpoint_keep_last_n)
        if len(job.checkpoints) > keep:
            pruned, job.checkpoints = (
                job.checkpoints[:-keep],
                job.checkpoints[-keep:],
            )
            for c in pruned:
                try:
                    Path(c.local_path).unlink(missing_ok=True)
                except OSError:
                    pass
        self._persist_training_unlocked()

    @with_state_lock
    def set_training_gpu_status(
        self, training_id: str, status: GpuStatus
    ) -> None:
        """Update the live GPU telemetry snapshot for a run (GPU-status panel)."""
        job = self._find_training(training_id)
        if job is None:
            return
        job.gpu_status = status
        self._persist_training_unlocked()

    @with_state_lock
    def mark_training_completed(self, training_id: str, *, local_lora_path: str) -> None:
        job = self._require_training(training_id)
        job.status = "completed"
        job.local_lora_path = local_lora_path
        job.completed_at = _now_iso()
        self._close_training_billing_unlocked(job, job.completed_at)
        job.error = None
        job.redownload_requested = False
        self._persist_training_unlocked()

    @with_state_lock
    def fail_training(self, training_id: str, error: str) -> None:
        job = self._require_training(training_id)
        job.status = "failed"
        job.error = error
        job.completed_at = _now_iso()
        self._close_training_billing_unlocked(job, job.completed_at)
        job.redownload_requested = False
        self._persist_training_unlocked()

    @with_state_lock
    def request_training_redownload(self, training_id: str) -> TrainingJob:
        """Re-fetch the trained adapter for a run that failed at the download
        step (the weights persist on the network volume, so no re-training).

        Only valid for a failed job that recorded a remote output dir. Flips the
        job back to ``running`` with ``redownload_requested`` so the reconciler
        re-acquires a pod and downloads the existing artifact.
        """
        job = self._require_training(training_id)
        if job.archived_at is not None:
            raise LoraTransitionError("Restore this training run before retrying it")
        if job.status != "failed":
            raise LoraTransitionError(
                f"Training {training_id} is {job.status!r}; only a failed run "
                "can retry its download"
            )
        if job.remote_output_dir is None:
            raise LoraTransitionError(
                f"Training {training_id} never reached the remote, so there is "
                "no trained adapter to re-download"
            )
        job.status = "running"
        job.error = None
        job.completed_at = None
        job.redownload_requested = True
        self._persist_training_unlocked()
        self._wakeup_event.set()
        return job.model_copy(deep=True)

    @with_state_lock
    def request_training_resume(self, training_id: str) -> TrainingJob:
        """Resume a failed/cancelled training run from its last checkpoint.

        Re-runs ``train.py`` with ``config.load_checkpoint`` set to the highest
        checkpoint the run reached (persisted during polling), so training
        continues from where it stopped instead of restarting at step 0. If no
        checkpoint was ever saved (the run died before the first checkpoint
        interval), it resumes from step 0. The remote output dir is preserved
        so new checkpoints land alongside the existing ones; the spent remote
        job id is cleared so the runner submits a fresh command.
        """
        job = self._require_training(training_id)
        if job.archived_at is not None:
            raise LoraTransitionError("Restore this training run before resuming it")
        if job.status not in ("failed", "cancelled"):
            raise LoraTransitionError(
                f"Training {training_id} is {job.status!r}; only a failed or "
                "cancelled run can be resumed"
            )
        load_checkpoint = None
        if job.latest_checkpoint_step is not None and job.remote_output_dir:
            load_checkpoint = paths.lora_checkpoint_path_in(
                job.remote_output_dir, job.latest_checkpoint_step
            )
        job.config = job.config.model_copy(update={"load_checkpoint": load_checkpoint})
        job.status = "pending"
        job.error = None
        job.completed_at = None
        job.cancel_requested = False
        job.consecutive_failures = 0
        job.status_detail = None
        job.redownload_requested = False
        # Drop the spent remote job id so the runner submits a fresh command.
        # Keep remote_output_dir (checkpoints persist there) + total_steps.
        if job.target is not None:
            job.target = job.target.model_copy(update={"remote_job_id": None})
        self._persist_training_unlocked()
        self._wakeup_event.set()
        return job.model_copy(deep=True)

    @with_state_lock
    def request_training_reset(self, training_id: str) -> TrainingJob:
        """Clear a finished training run's progress and re-train from scratch.

        Flips the job back to ``pending`` with ``load_checkpoint=None`` and
        sets ``reset_requested`` so the reconciler wipes the remote output dir
        (checkpoints + samples) and the local run folder before re-running
        ``train.py`` from step 0. Progress counters and the validation feed are
        cleared. The remote output dir path is preserved so the wipe targets
        the same location and new outputs land there again.
        """
        job = self._require_training(training_id)
        if job.archived_at is not None:
            raise LoraTransitionError("Restore this training run before resetting it")
        if job.status not in ("failed", "cancelled", "completed"):
            raise LoraTransitionError(
                f"Training {training_id} is {job.status!r}; only a finished "
                "run can be reset"
            )
        job.config = job.config.model_copy(update={"load_checkpoint": None})
        job.status = "pending"
        job.error = None
        job.completed_at = None
        job.cancel_requested = False
        job.consecutive_failures = 0
        job.status_detail = None
        job.redownload_requested = False
        job.reset_requested = True
        job.current_step = None
        job.latest_checkpoint_step = None
        job.eta_seconds = None
        job.first_step_at = None
        job.started_at = None
        job.local_lora_path = None
        job.validation_feed = []
        job.validation_sample_refs = []
        job.gpu_status = None
        if job.target is not None:
            job.target = job.target.model_copy(update={"remote_job_id": None})
        self._persist_training_unlocked()
        self._wakeup_event.set()
        return job.model_copy(deep=True)

    @with_state_lock
    def clear_training_reset_requested(self, training_id: str) -> None:
        """Reconciler-owned: clear the reset flag once the remote output is wiped."""
        job = self._find_training(training_id)
        if job is not None and job.reset_requested:
            job.reset_requested = False
            self._persist_training_unlocked()

    @with_state_lock
    def mark_training_cancelled(self, training_id: str) -> None:
        job = self._require_training(training_id)
        job.status = "cancelled"
        job.completed_at = _now_iso()
        self._close_training_billing_unlocked(job, job.completed_at)
        self._persist_training_unlocked()

    def _close_training_billing_unlocked(
        self, job: TrainingJob, ended_at: str
    ) -> None:
        """Close attribution exactly once and schedule workspace release."""
        if job.workload_billing_started_at is not None:
            job.workload_billing_ended_at = ended_at
            try:
                seconds = max(
                    0.0,
                    (
                        datetime.fromisoformat(ended_at)
                        - datetime.fromisoformat(job.workload_billing_started_at)
                    ).total_seconds(),
                )
            except ValueError:
                seconds = 0.0
            job.attributed_seconds = seconds
            rate = job.captured_hourly_rate or job.compute_rate_per_hr
            job.attributed_cost = (
                seconds * rate / 3600.0 if rate is not None else None
            )
        job.training_steps_ended_at = ended_at
        preprocessed = self._find_preprocessed(job.preprocessed_id)
        dataset = (
            self._find_dataset(preprocessed.dataset_id)
            if preprocessed is not None
            else None
        )
        if dataset is not None:
            dataset.final_activity_at = ended_at
            dataset.last_active_at = ended_at
            dataset.keep_alive_until = None
            dataset.release_status = "scheduled"
            dataset.release_error = None
            self._persist_datasets_unlocked()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _find_dataset(self, dataset_id: str) -> LoraDataset | None:
        for d in self._datasets.datasets:
            if d.id == dataset_id:
                return d
        return None

    def _require_dataset(self, dataset_id: str) -> LoraDataset:
        found = self._find_dataset(dataset_id)
        if found is None:
            raise LoraEntityNotFoundError(f"Dataset not found: {dataset_id}")
        return found

    def _find_preprocessed(self, preprocessed_id: str) -> PreprocessedDataset | None:
        for p in self._preprocessed.items:
            if p.id == preprocessed_id:
                return p
        return None

    def _require_preprocessed(self, preprocessed_id: str) -> PreprocessedDataset:
        found = self._find_preprocessed(preprocessed_id)
        if found is None:
            raise LoraEntityNotFoundError(f"Preprocessed not found: {preprocessed_id}")
        return found

    def _find_training(self, training_id: str) -> TrainingJob | None:
        for j in self._training.items:
            if j.id == training_id:
                return j
        return None

    def _require_training(self, training_id: str) -> TrainingJob:
        found = self._find_training(training_id)
        if found is None:
            raise LoraEntityNotFoundError(f"Training not found: {training_id}")
        return found

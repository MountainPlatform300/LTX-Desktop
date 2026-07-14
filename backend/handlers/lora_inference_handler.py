"""In-app LoRA inference orchestration handler (Gen Space "Apply LoRA" → generate).

The handler is the single entry point for generating with a LoRA picked from
the inference registry. It resolves the registry entry by `loraId` and routes
to the right pipeline by the entry's `variant`:

  - ``standard``            → `VideoGenerationHandler.generate` with the
                              adapter built into the fast DistilledPipeline
                              (lora_path / lora_scale). A standard LoRA is
                              "Generate Video + adapter".
  - ``union_control``       → `IcLoraHandler.generate` (official LTX-2 union
                              IC-LoRA, control-signal conditioned: canny /
                              depth / pose). The union checkpoint is resolved
                              inside the IC-LoRA handler from
                              `conditioning_type`, so only the id is checked.
  - ``video_input_ic_lora`` → `IcLoraHandler.generate_video_input` with the
                              user-trained adapter and a raw reference video
                              (no control-signal preprocessing).

The handler owns no GPU state — it delegates to the existing single-flight
generation handlers, so the queue's cooperative slot gating and the
generation progress UI work unchanged.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from api_types import (
    ImportedLoraVariantApi,
    LoraGenerateCompleteResponse,
    LoraGenerateCancelledResponse,
    LoraGenerateRequest,
    LoraGenerateResponse,
    LoraInferenceEntryApi,
    LoraStandardGenerateRequest,
    LoraUnionControlGenerateRequest,
    LoraVideoInputIcLoraGenerateRequest,
)
from _routes._errors import HTTPError
from handlers.base import StateHandlerBase
from handlers.ic_lora_handler import IcLoraHandler
from handlers.imported_lora_library import ImportedLoraLibrary, example_media_type_for
from handlers.lora_inference_registry import LoraInferenceRegistry
from handlers.lora_prompt_template import LoraPromptTemplateStore
from handlers.lora_training_handler import LoraTrainingHandler
from handlers.video_generation_handler import VideoGenerationHandler
from services.clip_processor.caption_proxy import build_caption_proxy_if_oversized
from services.interfaces import (
    ClipProcessor,
    LoraPromptProfileStatus,
    LoraPromptProfiler,
    VideoCaptioner,
)
from services.video_captioner.video_captioner import VideoCaptionerError
from state.app_state_types import AppState

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig


class LoraInferenceHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        config: RuntimeConfig,
        registry: LoraInferenceRegistry,
        video_generation: VideoGenerationHandler,
        ic_lora: IcLoraHandler,
        video_captioner: VideoCaptioner,
        template_store: LoraPromptTemplateStore,
        clip_processor: ClipProcessor,
        imported_library: ImportedLoraLibrary,
        prompt_profiler: LoraPromptProfiler,
        training_handler: LoraTrainingHandler,
    ) -> None:
        super().__init__(state, lock, config)
        self._registry = registry
        self._video_generation = video_generation
        self._ic_lora = ic_lora
        self._captioner = video_captioner
        self._templates = template_store
        self._clip_processor = clip_processor
        self._imported_library = imported_library
        self._profiler = prompt_profiler
        self._training = training_handler
        # Trained-LoRA example media lives here (imported examples live in the
        # per-LoRA library dir, owned by `ImportedLoraLibrary`). One file per
        # job: `user-<jobId>.<ext>`.
        self._trained_examples_dir: Path = config.app_data_dir / "lora" / "examples"

    def generate(self, req: LoraGenerateRequest) -> LoraGenerateResponse:
        entry = self._resolve_entry(req.loraId)
        match req:
            case LoraStandardGenerateRequest():
                return self._generate_standard(req, entry)
            case LoraUnionControlGenerateRequest():
                return self._generate_union_control(req)
            case LoraVideoInputIcLoraGenerateRequest():
                return self._generate_video_input(req, entry)
            case _:
                # Exhaustiveness: pydantic's discriminated union makes this
                # unreachable at runtime; the assert gives pyright a terminal.
                raise HTTPError(400, f"Unsupported LoRA variant: {req.variant!r}")

    # ------------------------------------------------------------------
    # Variant dispatch
    # ------------------------------------------------------------------

    def _generate_standard(
        self, req: LoraStandardGenerateRequest, entry: LoraInferenceEntryApi
    ) -> LoraGenerateResponse:
        lora_path = self._require_local_path(entry)
        request = req.request.model_copy(
            update={
                "prompt": self._prompt_with_trigger(
                    req.request.prompt, entry.triggerWord
                )
            }
        )
        resp = self._video_generation.generate(
            request, lora_path=lora_path, lora_scale=req.loraScale
        )
        if resp.status == "complete":
            return LoraGenerateCompleteResponse(videoPath=resp.video_path)
        return LoraGenerateCancelledResponse()

    def _generate_union_control(self, req: LoraUnionControlGenerateRequest) -> LoraGenerateResponse:
        # The union checkpoint is resolved inside the IC-LoRA handler from
        # conditioning_type, so we don't touch entry.localPath here — but the
        # id must still resolve (catches stale / unknown ids at the boundary).
        resp = self._ic_lora.generate(req.request)
        if resp.status == "complete":
            return LoraGenerateCompleteResponse(videoPath=resp.video_path)
        return LoraGenerateCancelledResponse()

    def _generate_video_input(
        self, req: LoraVideoInputIcLoraGenerateRequest, entry: LoraInferenceEntryApi
    ) -> LoraGenerateResponse:
        lora_path = self._require_local_path(entry)
        resp = self._ic_lora.generate_video_input(
            lora_path=lora_path,
            lora_scale=req.loraScale,
            prompt=self._prompt_with_trigger(req.prompt, entry.triggerWord),
            video_path=req.videoPath,
            conditioning_strength=req.conditioningStrength,
            negative_prompt=req.negativePrompt,
            target_duration=req.duration,
            preserve_audio=req.preserveAudio,
            refine=req.refine,
            resolution=req.resolution,
        )
        if resp.status == "complete":
            return LoraGenerateCompleteResponse(videoPath=resp.video_path)
        return LoraGenerateCancelledResponse()

    @staticmethod
    def _prompt_with_trigger(prompt: str, trigger_word: str | None) -> str:
        """Put the verified training trigger first without duplicating it."""
        cleaned = prompt.strip()
        trigger = trigger_word.strip() if trigger_word else ""
        if not trigger or cleaned == trigger or cleaned.startswith(f"{trigger} "):
            return cleaned
        return f"{trigger} {cleaned}"

    # ------------------------------------------------------------------
    # Import (with per-LoRA prompt profiling)
    # ------------------------------------------------------------------

    def import_lora_with_profile(
        self,
        *,
        source_path: str,
        name: str,
        variant: ImportedLoraVariantApi,
        description: str | None,
        trigger_word: str | None,
        huggingface_url: str | None,
        example_prompt: str | None,
    ) -> tuple[LoraInferenceEntryApi, LoraPromptProfileStatus, str | None]:
        """Import a LoRA, then derive + persist an accurate per-LoRA prompt profile.

        Copies the weights via the imported-library ledger, then runs the prompt
        profiler (built-in official profile → HuggingFace card → example prompt)
        to obtain the trigger word + system prompt the adapter was trained on.
        When a profile is found it's persisted as a template override so the
        registry serves the configured prompt immediately and the auto-prompt
        assistant activates the LoRA instead of silently no-op'ing. Profiling is
        best-effort: on any failure the import still succeeds with the
        name-derived default, and the profiling outcome (status + user-facing
        message) is returned so the caller can surface it — previously this was
        entirely silent. When no exact trigger can be discovered, the entry
        remains triggerless rather than guessing from its filename. Returns
        ``(entry, profile_status, profile_message)``.
        """
        entry = self._imported_library.import_lora(
            source_path=source_path,
            name=name,
            variant=variant,
            description=description,
            trigger_word=trigger_word,
            huggingface_url=huggingface_url,
        )
        api_key = self.state.app_settings.gemini_api_key or ""
        profile_status: LoraPromptProfileStatus = "skipped"
        profile_message: str | None = None
        try:
            result = self._profiler.profile(
                name=name,
                filename=Path(source_path).name,
                variant=variant,
                huggingface_url=huggingface_url,
                example_prompt=example_prompt,
                api_key=api_key,
            )
            profile_status = result.status
            profile_message = result.message
            profile = result.profile
        except Exception:  # noqa: BLE001 — profiling must never break the import
            profile = None
            profile_status = "failed"
            profile_message = "Profiling raised an unexpected error — using the default prompt."

        if profile is not None:
            # The profile's trigger word wins (it's the exact token the adapter
            # was trained on); fall back to the user-supplied trigger only when
            # the profile has no trigger (some LoRAs use a descriptive prompt).
            resolved_trigger = profile.trigger_word or trigger_word
            self._templates.set_override(
                entry.id,
                prompt_template=profile.system_prompt,
                trigger_word=resolved_trigger,
            )
        # Re-resolve through the registry so the returned entry always carries
        # the overlaid template (configured profile or safe generic default).
        return self._resolve_entry(entry.id), profile_status, profile_message

    # ------------------------------------------------------------------
    # Registry resolution
    # ------------------------------------------------------------------

    def _resolve_entry(self, lora_id: str) -> LoraInferenceEntryApi:
        for entry in self._registry.list_entries():
            if entry.id == lora_id:
                return entry
        raise HTTPError(404, f"Unknown LoRA id: {lora_id}", code="LORA_NOT_FOUND")

    @staticmethod
    def _require_local_path(entry: LoraInferenceEntryApi) -> str:
        if not entry.available or entry.localPath is None:
            raise HTTPError(
                409,
                f"LoRA '{entry.name}' is not available on disk",
                code="LORA_UNAVAILABLE",
            )
        if Path(entry.localPath).suffix.lower() != ".safetensors":
            raise HTTPError(
                409,
                "This LoRA uses a legacy executable weight format. Convert it "
                "to .safetensors before applying it.",
                code="LORA_UNSAFE_WEIGHT_FORMAT",
            )
        return entry.localPath

    # ------------------------------------------------------------------
    # Per-LoRA prompt-writing assistant
    # ------------------------------------------------------------------

    def auto_prompt(self, *, lora_id: str, video_path: str) -> str:
        """Have Gemini Flash write a tailored prompt for `video_path` using the
        LoRA's per-LoRA system prompt.

        Gated on a configured Gemini API key. The entry must carry a
        `promptTemplate` (auto-generated or user-edited); `standard` style LoRAs
        have none and are rejected here — the UI hides the action for them.
        """
        entry = self._resolve_entry(lora_id)
        if not entry.promptTemplate:
            raise HTTPError(
                409,
                f"LoRA '{entry.name}' has no prompt template",
                code="LORA_NO_PROMPT_TEMPLATE",
            )
        api_key = self.state.app_settings.gemini_api_key
        if not api_key:
            raise HTTPError(
                400,
                "A Gemini API key is required for auto-prompt. Add one in Settings.",
                code="GEMINI_API_KEY_MISSING",
            )
        # The reference video can be far larger than Gemini's ~14MB inline
        # captioning ceiling (a full imported clip). Transcode a small
        # caption-only proxy when needed so auto-prompt works on big references
        # instead of failing with a 413. Auto-prompt is video-only (the LoRA
        # template describes motion/subject), so the proxy is muted.
        proxy = build_caption_proxy_if_oversized(
            self._clip_processor, video_path, with_audio=False
        )
        caption_path = video_path
        proxy_dir: Path | None = None
        if proxy is not None:
            proxy_dir, caption_path = proxy
        try:
            return self._captioner.caption(
                video_path=caption_path,
                api_key=api_key,
                with_audio=False,
                instructions=entry.promptTemplate,
            )
        except VideoCaptionerError as exc:
            raise HTTPError(exc.status_code, exc.detail, code="AUTO_PROMPT_FAILED") from exc
        finally:
            if proxy_dir is not None:
                shutil.rmtree(proxy_dir, ignore_errors=True)

    def update_prompt_template(
        self,
        *,
        lora_id: str,
        prompt_template: str | None,
        trigger_word: str | None,
    ) -> LoraInferenceEntryApi:
        """Persist a user edit of a LoRA's prompt template / trigger word.

        Validates the LoRA exists before writing so a stale id doesn't create a
        dangling override. Returns the re-derived entry (with the override
        applied) so the UI updates immediately.
        """
        self._resolve_entry(lora_id)  # raises LORA_NOT_FOUND for unknown ids
        self._templates.set_override(
            lora_id,
            prompt_template=prompt_template,
            trigger_word=trigger_word,
        )
        return self._resolve_entry(lora_id)

    # ------------------------------------------------------------------
    # LoRA Library management (imported + trained metadata, reprofile, delete)
    # ------------------------------------------------------------------

    def update_imported(
        self,
        *,
        lora_id: str,
        name: str | None = None,
        description: str | None = None,
        huggingface_url: str | None = None,
    ) -> LoraInferenceEntryApi:
        """Patch editable metadata on an imported LoRA, then re-resolve."""
        self._imported_library.update_imported(
            lora_id,
            name=name,
            description=description,
            huggingface_url=huggingface_url,
        )
        return self._resolve_entry(lora_id)

    def reprofile_imported(
        self,
        *,
        lora_id: str,
        huggingface_url: str | None,
        example_prompt: str | None,
    ) -> tuple[LoraInferenceEntryApi, LoraPromptProfileStatus, str | None]:
        """Re-run the prompt profiler for an already-imported LoRA.

        Falls back to the stored HuggingFace URL when the caller omits one, so
        the user can re-profile with a single click. Persists the resulting
        template + trigger as an override and returns the re-resolved entry with
        the profiling outcome.
        """
        item = self._imported_library.find_imported(lora_id)
        if item is None:
            raise HTTPError(404, f"Unknown imported LoRA: {lora_id}", code="IMPORT_LORA_NOT_FOUND")
        effective_hf = huggingface_url if huggingface_url is not None else item.huggingface_url
        api_key = self.state.app_settings.gemini_api_key or ""
        profile_status: LoraPromptProfileStatus = "skipped"
        profile_message: str | None = None
        profile = None
        try:
            result = self._profiler.profile(
                name=item.name,
                filename=Path(item.local_path).name,
                variant=item.variant,
                huggingface_url=effective_hf,
                example_prompt=example_prompt,
                api_key=api_key,
            )
            profile_status = result.status
            profile_message = result.message
            profile = result.profile
        except Exception:  # noqa: BLE001 — reprofile must never 500
            profile_status = "failed"
            profile_message = "Profiling raised an unexpected error — using the default prompt."
        if profile is not None:
            resolved_trigger = profile.trigger_word or item.trigger_word
            self._templates.set_override(
                lora_id,
                prompt_template=profile.system_prompt,
                trigger_word=resolved_trigger,
            )
        return self._resolve_entry(lora_id), profile_status, profile_message

    def update_trained(
        self,
        *,
        lora_id: str,
        name: str | None = None,
        description: str | None = None,
    ) -> LoraInferenceEntryApi:
        """Patch editable metadata on a user-trained LoRA (backed by the job)."""
        entry = self._resolve_entry(lora_id)
        if entry.kind != "user_trained" or not entry.sourceTrainingId:
            raise HTTPError(
                400, f"{lora_id} is not a user-trained LoRA", code="LORA_NOT_TRAINED"
            )
        self._training.update_training_meta(
            entry.sourceTrainingId, name=name, description=description
        )
        return self._resolve_entry(lora_id)

    def delete_trained(self, lora_id: str) -> None:
        """Delete a user-trained LoRA by registry entry id (``user-<jobId>``)."""
        entry = self._resolve_entry(lora_id)
        if entry.kind != "user_trained" or not entry.sourceTrainingId:
            raise HTTPError(
                400, f"{lora_id} is not a user-trained LoRA", code="LORA_NOT_TRAINED"
            )
        self._training.delete_training(entry.sourceTrainingId)

    # ------------------------------------------------------------------
    # Example media (CivitAI-style "what does this LoRA do?" preview)
    # ------------------------------------------------------------------

    def set_example(self, *, lora_id: str, source_path: str) -> LoraInferenceEntryApi:
        """Attach (or replace) an example image/video to any user-managed LoRA.

        Delegates to the imported library for imported LoRAs (file lives in the
        per-LoRA dir); for trained LoRAs copies the file into the shared
        examples dir and persists the path on the training job. Official union
        LoRAs can't have an example. Returns the re-resolved entry.
        """
        entry = self._resolve_entry(lora_id)
        src = Path(source_path)
        if not src.is_file():
            raise HTTPError(
                400, f"Example file not found: {source_path}", code="LORA_EXAMPLE_NOT_FOUND"
            )
        if example_media_type_for(source_path) is None:
            raise HTTPError(
                400,
                f"Unsupported example media type: {src.suffix}",
                code="LORA_EXAMPLE_UNSUPPORTED_TYPE",
            )
        if entry.kind == "imported":
            self._imported_library.set_example(lora_id, source_path=source_path)
            return self._resolve_entry(lora_id)
        if entry.kind == "user_trained" and entry.sourceTrainingId:
            job = self._training.get_training_state_by_id(entry.sourceTrainingId)
            if job is None:
                raise HTTPError(404, f"Unknown training job: {entry.sourceTrainingId}", code="LORA_NOT_FOUND")
            # Tear down any prior example file before writing the new one.
            if job.example_path:
                try:
                    Path(job.example_path).unlink(missing_ok=True)
                except OSError:
                    pass
            self._trained_examples_dir.mkdir(parents=True, exist_ok=True)
            dest = self._trained_examples_dir / f"user-{job.id}{src.suffix.lower()}"
            shutil.copy2(src, dest)
            self._training.set_training_example_path(job.id, example_path=str(dest))
            return self._resolve_entry(lora_id)
        raise HTTPError(
            400, "Examples can only be attached to imported or trained LoRAs", code="LORA_EXAMPLE_NOT_SUPPORTED"
        )

    def clear_example(self, lora_id: str) -> None:
        """Remove a LoRA's example media (file + stored path). Idempotent."""
        entry = self._resolve_entry(lora_id)
        if entry.kind == "imported":
            self._imported_library.clear_example(lora_id)
            return
        if entry.kind == "user_trained" and entry.sourceTrainingId:
            job = self._training.get_training_state_by_id(entry.sourceTrainingId)
            if job is None:
                return
            example_path = job.example_path
            self._training.clear_training_example_path(entry.sourceTrainingId)
            if example_path:
                try:
                    Path(example_path).unlink(missing_ok=True)
                except OSError:
                    pass
            return
        raise HTTPError(
            400, "Examples can only be attached to imported or trained LoRAs", code="LORA_EXAMPLE_NOT_SUPPORTED"
        )

    def get_example_local_path(self, lora_id: str) -> str | None:
        """Resolve the on-disk path of a LoRA's example media for the FileResponse.

        Returns None when the entry has no example or the file is missing, so
        the route can 404 cleanly. Imported examples are read from the ledger;
        trained examples from the training job.
        """
        entry = self._resolve_entry(lora_id)
        if entry.kind == "imported":
            item = self._imported_library.find_imported(lora_id)
            if item is None or not item.example_path:
                return None
            return item.example_path if Path(item.example_path).is_file() else None
        if entry.kind == "user_trained" and entry.sourceTrainingId:
            job = self._training.get_training_state_by_id(entry.sourceTrainingId)
            if job is None or not job.example_path:
                return None
            return job.example_path if Path(job.example_path).is_file() else None
        return None

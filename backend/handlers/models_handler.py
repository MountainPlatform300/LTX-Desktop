"""Checkpoint recommendation and filesystem model state helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from _routes._errors import HTTPError
from api_types import (
    CheckpointPathResponse,
    ImageGenRecommendationResponse,
    LoadModelFromPathResponse,
    LtxDownloadRecommendationResponse,
    LtxIcLoraRecommendationResponse,
    LtxOkRecommendationResponse,
    LtxRecommendationResponse,
    LtxUpgradeRecommendationResponse,
    LTXLocalModelId,
    ModelCheckpointID,
    TextEncoderRecommendationResponse,
)
from handlers.base import StateHandlerBase
from runtime_config.model_download_specs import (
    ALL_MODEL_CP_IDS,
    DEPTH_PROCESSOR_CP_ID,
    LTXLocalModelRelevant,
    PERSON_DETECTOR_CP_ID,
    POSE_PROCESSOR_CP_ID,
    get_downloaded_ltx_model_id,
    get_ic_loras_cp_ids,
    get_latest_ltx_model_id,
    get_ltx_cps,
    get_ltx_model_cp_ids,
    get_ltx_model_id_for_cp,
    get_ltx_model_spec,
    get_ltx_overlay_cp_ids,
    get_model_cp_spec,
    is_cp_downloaded,
    delete_cp_path,
    resolve_model_path,
)

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig
    from state.app_state_types import AppState


@dataclass(frozen=True, slots=True)
class ResolvedUpgradeDownload:
    current_model_id: LTXLocalModelId
    target_model_id: LTXLocalModelId
    cp_ids: tuple[ModelCheckpointID, ...]


class ModelsHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        config: RuntimeConfig,
    ) -> None:
        super().__init__(state, lock, config)

    def _ordered_cp_ids(self, cp_ids: set[ModelCheckpointID]) -> list[ModelCheckpointID]:
        return [cp_id for cp_id in ALL_MODEL_CP_IDS if cp_id in cp_ids]

    def _ensure_local_model_mode(self) -> None:
        if self.config.force_api_generations:
            raise HTTPError(409, "LOCAL_MODEL_RECOMMENDATIONS_DISABLED_IN_FORCE_API_MODE")

    def _current_downloaded_ltx_model_id(self) -> LTXLocalModelId | None:
        return get_downloaded_ltx_model_id(self.models_dir)

    def _has_api_key(self) -> bool:
        return bool(self.state.app_settings.ltx_api_key.strip())

    def is_cp_downloaded(self, cp_id: ModelCheckpointID) -> bool:
        return is_cp_downloaded(self.models_dir, cp_id)

    def get_downloaded_checkpoints(self) -> set[ModelCheckpointID]:
        return {cp_id for cp_id in ALL_MODEL_CP_IDS if self.is_cp_downloaded(cp_id)}

    def _get_required_ltx_cp_ids(self, model_id: LTXLocalModelId) -> set[ModelCheckpointID]:
        spec = get_ltx_model_spec(model_id)
        required: set[ModelCheckpointID] = {spec.model_cp, spec.upscale_cp}
        if not self._has_api_key():
            required.add(spec.text_encoder_cp)
        return required

    def _get_missing_cp_ids(self, cp_ids: set[ModelCheckpointID]) -> set[ModelCheckpointID]:
        return {cp_id for cp_id in cp_ids if not self.is_cp_downloaded(cp_id)}

    def _get_upgrade_message(self, current_model_id: LTXLocalModelId, target_model_id: LTXLocalModelId) -> str | None:
        relevance = get_ltx_model_spec(target_model_id).relevance
        if not isinstance(relevance, LTXLocalModelRelevant):
            return None
        return relevance.upgrade_messages.get(current_model_id)

    def _get_upgrade_dependency_downloads(
        self,
        current_model_id: LTXLocalModelId,
        target_model_id: LTXLocalModelId,
    ) -> set[ModelCheckpointID]:
        current_spec = get_ltx_model_spec(current_model_id)
        target_spec = get_ltx_model_spec(target_model_id)
        cp_ids: set[ModelCheckpointID] = {target_spec.model_cp}

        if (
            current_spec.upscale_cp != target_spec.upscale_cp
            and self.is_cp_downloaded(current_spec.upscale_cp)
            and not self.is_cp_downloaded(target_spec.upscale_cp)
        ):
            cp_ids.add(target_spec.upscale_cp)

        if (
            current_spec.text_encoder_cp != target_spec.text_encoder_cp
            and self.is_cp_downloaded(current_spec.text_encoder_cp)
            and not self.is_cp_downloaded(target_spec.text_encoder_cp)
        ):
            cp_ids.add(target_spec.text_encoder_cp)

        current_ic_loras_spec = current_spec.ic_loras_spec
        target_ic_loras_spec = target_spec.ic_loras_spec
        ic_lora_pairs: tuple[tuple[ModelCheckpointID, ModelCheckpointID], ...] = (
            (current_ic_loras_spec.depth_cp, target_ic_loras_spec.depth_cp),
            (current_ic_loras_spec.canny_cp, target_ic_loras_spec.canny_cp),
            (current_ic_loras_spec.pose_cp, target_ic_loras_spec.pose_cp),
        )
        for current_cp_id, target_cp_id in ic_lora_pairs:
            if (
                current_cp_id != target_cp_id
                and self.is_cp_downloaded(current_cp_id)
                and not self.is_cp_downloaded(target_cp_id)
            ):
                cp_ids.add(target_cp_id)

        return cp_ids

    def _get_upgrade_delete_cp_ids(
        self,
        current_model_id: LTXLocalModelId,
        target_model_id: LTXLocalModelId,
    ) -> set[ModelCheckpointID]:
        current_cp_ids = set(get_ltx_model_cp_ids(current_model_id))
        target_cp_ids = set(get_ltx_model_cp_ids(target_model_id))
        return {
            cp_id
            for cp_id in current_cp_ids - target_cp_ids
            if self.is_cp_downloaded(cp_id)
        }

    def get_ltx_recommendation(self) -> LtxRecommendationResponse:
        self._ensure_local_model_mode()

        current_model_id = self._current_downloaded_ltx_model_id()
        latest_model_id = get_latest_ltx_model_id()

        if current_model_id is None:
            cps_to_download = self._ordered_cp_ids(
                self._get_missing_cp_ids(self._get_required_ltx_cp_ids(latest_model_id))
            )
            return LtxDownloadRecommendationResponse(status="download", cps_to_download=cps_to_download)

        if current_model_id == latest_model_id:
            missing_required = self._ordered_cp_ids(
                self._get_missing_cp_ids(self._get_required_ltx_cp_ids(latest_model_id))
            )
            if missing_required:
                return LtxDownloadRecommendationResponse(status="download", cps_to_download=missing_required)
            # Opt-in dev quality base: surface any missing overlay checkpoints
            # (dev + distilled v1.1 LoRA) so the user can fetch them. IC-LoRA
            # generation transparently falls back to the distilled checkpoint
            # until both are present, so this is a non-blocking prompt.
            if self.state.app_settings.use_dev_quality_base:
                overlay_missing = self._ordered_cp_ids(
                    self._get_missing_cp_ids(set(get_ltx_overlay_cp_ids(latest_model_id)))
                )
                if overlay_missing:
                    return LtxDownloadRecommendationResponse(status="download", cps_to_download=overlay_missing)
            return LtxOkRecommendationResponse(status="ok")

        cps_to_download = self._ordered_cp_ids(
            self._get_upgrade_dependency_downloads(current_model_id, latest_model_id)
        )
        cps_to_delete = self._ordered_cp_ids(
            self._get_upgrade_delete_cp_ids(current_model_id, latest_model_id)
        )
        return LtxUpgradeRecommendationResponse(
            status="upgrade",
            ltx_model_id=latest_model_id,
            upgrade_message=self._get_upgrade_message(current_model_id, latest_model_id),
            cps_to_download=cps_to_download,
            cps_to_delete=cps_to_delete,
        )

    def get_img_gen_recommendation(self) -> ImageGenRecommendationResponse:
        self._ensure_local_model_mode()
        # First-run gating: recommend the first *available* (inference-wired)
        # image model that isn't downloaded yet. Coming-soon catalog entries are
        # intentionally skipped — they're downloadable from the picker but
        # aren't required to use the image mode. Edit-only models (Klein) are
        # also skipped: they're optional and served by the edit endpoint, not a
        # prerequisite for text-to-image.
        cp_to_download: ModelCheckpointID | None = None
        from runtime_config.image_model_specs import IMAGE_MODELS

        for spec in IMAGE_MODELS:
            if spec.inference_status != "available":
                continue
            if spec.is_edit_model:
                continue
            if not self.is_cp_downloaded(spec.checkpoint_id):
                cp_to_download = spec.checkpoint_id
                break
        return ImageGenRecommendationResponse(cp_to_download=cp_to_download)

    def _require_downloaded_ltx_model_id(self) -> LTXLocalModelId:
        model_id = self._current_downloaded_ltx_model_id()
        if model_id is None:
            raise HTTPError(409, "NO_DOWNLOADED_LTX_MODEL")
        return model_id

    def get_ltx_ic_lora_recommendation(self) -> LtxIcLoraRecommendationResponse:
        self._ensure_local_model_mode()
        model_id = self._require_downloaded_ltx_model_id()
        spec = get_ltx_model_spec(model_id)
        # The union IC-LoRA advertises canny / depth / pose. Canny needs no
        # extra checkpoint; depth needs the MiDaS DPT model; pose needs both
        # the DW pose processor and the YOLOX person detector it runs on top
        # of. Bundle all of them so a user who picks pose isn't sent straight
        # into a "Checkpoint not found" at inference time — the download gate
        # surfaces them up front (matching how depth is already handled).
        required_cp_ids: set[ModelCheckpointID] = set(get_ic_loras_cp_ids(spec.ic_loras_spec))
        required_cp_ids.add(DEPTH_PROCESSOR_CP_ID)
        required_cp_ids.add(POSE_PROCESSOR_CP_ID)
        required_cp_ids.add(PERSON_DETECTOR_CP_ID)
        cp_ids = self._get_missing_cp_ids(required_cp_ids)
        return LtxIcLoraRecommendationResponse(cps_to_download=self._ordered_cp_ids(cp_ids))

    def get_text_encoder_recommendation(self) -> TextEncoderRecommendationResponse:
        self._ensure_local_model_mode()
        model_id = self._require_downloaded_ltx_model_id()
        cp_id = get_ltx_model_spec(model_id).text_encoder_cp
        spec = get_model_cp_spec(cp_id)
        return TextEncoderRecommendationResponse(
            cp_to_download=None if self.is_cp_downloaded(cp_id) else cp_id,
            expected_size_bytes=spec.expected_size_bytes,
            expected_size_gb=round(spec.expected_size_bytes / (1024**3), 1),
        )

    def resolve_upgrade_download(self, requested_cp_ids: set[ModelCheckpointID]) -> ResolvedUpgradeDownload:
        self._ensure_local_model_mode()

        current_model_id = self._current_downloaded_ltx_model_id()
        if current_model_id is None:
            raise HTTPError(409, "NO_DOWNLOADED_LTX_MODEL")

        latest_model_id = get_latest_ltx_model_id()
        if current_model_id == latest_model_id:
            raise HTTPError(409, "ALREADY_ON_LATEST_LTX_MODEL")

        requested_ltx_cp_ids = requested_cp_ids & get_ltx_cps()
        if len(requested_ltx_cp_ids) != 1:
            raise HTTPError(409, "INVALID_UPGRADE_REQUEST")

        target_model_cp_id = next(iter(requested_ltx_cp_ids))
        target_model_id = get_ltx_model_id_for_cp(target_model_cp_id)
        if target_model_id is None:
            raise HTTPError(500, "INVALID_LTX_MODEL_CONFIG")

        if target_model_id != latest_model_id:
            raise HTTPError(409, "INVALID_UPGRADE_REQUEST")
        target_relevance = get_ltx_model_spec(target_model_id).relevance
        if not isinstance(target_relevance, LTXLocalModelRelevant):
            raise HTTPError(500, "INVALID_LTX_MODEL_CONFIG")

        recommendation = self.get_ltx_recommendation()
        if not isinstance(recommendation, LtxUpgradeRecommendationResponse):
            raise HTTPError(409, "INVALID_UPGRADE_REQUEST")

        expected_cp_ids = set(recommendation.cps_to_download)
        if requested_cp_ids != expected_cp_ids:
            raise HTTPError(409, "INVALID_UPGRADE_REQUEST")

        return ResolvedUpgradeDownload(
            current_model_id=current_model_id,
            target_model_id=target_model_id,
            cp_ids=tuple(self._ordered_cp_ids(expected_cp_ids)),
        )

    def get_protected_cp_ids(self) -> set[ModelCheckpointID]:
        current_model_id = self._current_downloaded_ltx_model_id()
        if current_model_id is None:
            return set()
        protected: set[ModelCheckpointID] = set(get_ltx_model_cp_ids(current_model_id))
        # When the dev quality base is enabled, the overlay checkpoints are in
        # active use by IC-LoRA — protect them from the model-manager delete
        # action. Toggling the setting off unprotects them again.
        if self.state.app_settings.use_dev_quality_base:
            protected.update(get_ltx_overlay_cp_ids(current_model_id))
        return protected

    def delete_checkpoints(self, cp_ids: set[ModelCheckpointID]) -> None:
        protected = self.get_protected_cp_ids()
        if cp_ids & protected:
            raise HTTPError(409, "DELETE_PROTECTED_CHECKPOINT")
        for cp_id in cp_ids:
            delete_cp_path(self.models_dir, cp_id)

    def get_checkpoint_path(self, cp_id: ModelCheckpointID) -> CheckpointPathResponse:
        """Resolved on-disk path for a checkpoint (for "Reveal in Explorer")."""
        path = resolve_model_path(self.models_dir, cp_id)
        return CheckpointPathResponse(
            cp_id=cp_id,
            path=str(path),
            exists=is_cp_downloaded(self.models_dir, cp_id),
        )

    def load_from_path(self, cp_id: ModelCheckpointID, source_path: str) -> LoadModelFromPathResponse:
        """Link or copy an already-downloaded model into the models dir.

        The user picks a folder/file on disk containing the checkpoint; this
        links it (symlink, or a Windows junction for folders) into the expected
        location so the app uses it in place without re-downloading. If linking
        fails (e.g. missing privilege), falls back to a copy. Raises 400 on a
        bad/missing source or type mismatch, 409         if the target already exists.
        """
        spec = get_model_cp_spec(cp_id)
        src = Path(source_path).expanduser()
        if not src.exists():
            raise HTTPError(400, f"Source not found: {src}", code="LOAD_SOURCE_NOT_FOUND")
        if spec.is_folder and not src.is_dir():
            raise HTTPError(
                400,
                f"{cp_id} expects a folder, but {src} is not a directory.",
                code="LOAD_SOURCE_TYPE_MISMATCH",
            )
        if not spec.is_folder and not src.is_file():
            raise HTTPError(
                400,
                f"{cp_id} expects a file, but {src} is not a file.",
                code="LOAD_SOURCE_TYPE_MISMATCH",
            )

        dst = resolve_model_path(self.models_dir, cp_id)
        if is_cp_downloaded(self.models_dir, cp_id):
            raise HTTPError(
                409,
                f"{cp_id} is already present at {dst}. Remove it first to re-link.",
                code="LOAD_TARGET_EXISTS",
            )

        dst.parent.mkdir(parents=True, exist_ok=True)

        # 1) Symlink — keeps the model in place, no duplication.
        try:
            os.symlink(src, dst, target_is_directory=spec.is_folder)
            return LoadModelFromPathResponse(cp_id=cp_id, path=str(dst), method="linked")
        except OSError:
            pass

        # 2) Windows directory junction — no admin/developer-mode needed.
        if sys.platform == "win32" and spec.is_folder:
            try:
                result = subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(dst), str(src)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0 and dst.exists():
                    return LoadModelFromPathResponse(cp_id=cp_id, path=str(dst), method="linked")
            except OSError:
                pass

        # 3) Copy fallback — correct but slow for large models.
        if spec.is_folder:
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        return LoadModelFromPathResponse(cp_id=cp_id, path=str(dst), method="copied")

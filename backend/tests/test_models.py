"""Integration-style tests for checkpoint recommendation and download endpoints."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from _routes._errors import HTTPError
import handlers.models_handler as models_handler_module
from runtime_config.model_download_specs import (
    DEPTH_PROCESSOR_CP_ID,
    IMG_GEN_MODEL_CP_ID,
    LTXLocalModelDeprecated,
    PERSON_DETECTOR_CP_ID,
    POSE_PROCESSOR_CP_ID,
    get_ic_loras_cp_ids,
    get_latest_ltx_model_id,
    get_ltx_model_spec,
    get_ltx_overlay_cp_ids,
    resolve_downloading_dir,
    resolve_model_path,
)
from state.app_state_types import DownloadSessionComplete, DownloadSessionError, DownloadingSession, FileDownloadRunning
from tests.http_error_assertions import assert_http_error


def _current_ltx_spec():
    return get_ltx_model_spec(get_latest_ltx_model_id())


def _cp_path(test_state, cp_id: str) -> Path:
    return resolve_model_path(test_state.config.default_models_dir, cp_id)


class TestRecommendations:
    def test_ltx_recommendation_requires_primary_local_bundle(self, client):
        spec = _current_ltx_spec()
        response = client.get("/api/models/ltx-recommendation")
        assert response.status_code == 200
        assert response.json() == {
            "status": "download",
            "cps_to_download": [
                spec.model_cp,
                spec.upscale_cp,
                spec.text_encoder_cp,
            ],
        }

    def test_ltx_recommendation_skips_text_encoder_when_api_key_exists(self, client, test_state):
        test_state.state.app_settings.ltx_api_key = "test-key"
        spec = _current_ltx_spec()
        response = client.get("/api/models/ltx-recommendation")
        assert response.status_code == 200
        assert response.json() == {
            "status": "download",
            "cps_to_download": [
                spec.model_cp,
                spec.upscale_cp,
            ],
        }

    def test_ltx_recommendation_ok_when_required_bundle_is_downloaded(self, client, create_fake_model_files):
        create_fake_model_files()
        response = client.get("/api/models/ltx-recommendation")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_ltx_recommendation_surfaces_dev_quality_base_overlay_when_enabled(
        self, client, test_state, create_fake_model_files
    ):
        create_fake_model_files()
        spec = _current_ltx_spec()
        test_state.state.app_settings.use_dev_quality_base = True

        # Required bundle is downloaded but the opt-in overlay (dev + distilled
        # v1.1 LoRA) is not -> recommendation prompts the overlay downloads in
        # order. IC-LoRA generation falls back to distilled until then.
        response = client.get("/api/models/ltx-recommendation")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "download"
        assert body["cps_to_download"] == [spec.quality_base_cp, spec.distilled_lora_cp]

        # Once the overlay is downloaded, recommendation returns ok again.
        for cp_id in get_ltx_overlay_cp_ids(get_latest_ltx_model_id()):
            path = _cp_path(test_state, cp_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x00" * 1024)
        response = client.get("/api/models/ltx-recommendation")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_ltx_recommendation_ignores_dev_overlay_when_setting_off(
        self, client, test_state, create_fake_model_files
    ):
        create_fake_model_files()
        # Overlay checkpoints present but the setting is off -> not required,
        # recommendation stays ok (overlay is opt-in, never forced).
        for cp_id in get_ltx_overlay_cp_ids(get_latest_ltx_model_id()):
            path = _cp_path(test_state, cp_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x00" * 1024)
        response = client.get("/api/models/ltx-recommendation")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_ltx_recommendation_reports_missing_text_encoder_for_current_model(self, client, test_state, create_fake_model_files):
        create_fake_model_files()
        text_encoder_path = _cp_path(test_state, _current_ltx_spec().text_encoder_cp)
        for child in text_encoder_path.iterdir():
            child.unlink()
        text_encoder_path.rmdir()

        response = client.get("/api/models/ltx-recommendation")
        assert response.status_code == 200
        assert response.json() == {
            "status": "download",
            "cps_to_download": [_current_ltx_spec().text_encoder_cp],
        }

    def test_img_gen_recommendation(self, client, create_fake_model_files):
        response = client.get("/api/models/img-gen-recommendation")
        assert response.status_code == 200
        assert response.json()["cp_to_download"] == IMG_GEN_MODEL_CP_ID

        create_fake_model_files(include_zit=True)
        response = client.get("/api/models/img-gen-recommendation")
        assert response.status_code == 200
        assert response.json()["cp_to_download"] is None

    def test_text_encoder_recommendation(self, client, create_fake_model_files, test_state):
        create_fake_model_files()
        text_encoder_path = _cp_path(test_state, _current_ltx_spec().text_encoder_cp)
        for child in text_encoder_path.iterdir():
            child.unlink()
        text_encoder_path.rmdir()

        response = client.get("/api/models/text-encoder-recommendation")
        assert response.status_code == 200
        assert response.json()["cp_to_download"] == _current_ltx_spec().text_encoder_cp
        assert response.json()["expected_size_bytes"] > 0

    def test_ic_lora_recommendation(self, client, create_fake_model_files, create_fake_ic_lora_files):
        create_fake_model_files()
        response = client.get("/api/models/ltx-ic-lora-recommendation")
        assert response.status_code == 200
        assert response.json()["cps_to_download"] == [
            *get_ic_loras_cp_ids(_current_ltx_spec().ic_loras_spec),
            DEPTH_PROCESSOR_CP_ID,
            PERSON_DETECTOR_CP_ID,
            POSE_PROCESSOR_CP_ID,
        ]

        create_fake_ic_lora_files()
        response = client.get("/api/models/ltx-ic-lora-recommendation")
        assert response.status_code == 200
        assert response.json()["cps_to_download"] == []


class TestDownloadProgress:
    def test_unknown_session_returns_404(self, client):
        response = client.get("/api/models/download/progress", params={"sessionId": "nonexistent"})
        assert_http_error(response, status_code=404, code="UNKNOWN_DOWNLOAD_SESSION")

    def test_active_progress(self, client, test_state):
        test_state.state.downloading_session = DownloadingSession(
            id="test-session",
            current_running_file=FileDownloadRunning(
                file_type="ltx-2.3-22b-distilled",
                target_path="ltx-2.3-22b-distilled.safetensors",
                downloaded_bytes=5_000_000_000,
                speed_bytes_per_sec=50_000_000.0,
            ),
            files_to_download={"ltx-2.3-22b-distilled"},
            completed_files=set(),
            completed_bytes=0,
        )
        response = client.get("/api/models/download/progress", params={"sessionId": "test-session"})
        assert response.status_code == 200
        assert response.json()["status"] == "downloading"
        assert response.json()["current_downloading_file"] == "ltx-2.3-22b-distilled"

    def test_completed_and_error_sessions(self, client, test_state):
        test_state.state.completed_download_sessions["done-session"] = DownloadSessionComplete()
        test_state.state.completed_download_sessions["err-session"] = DownloadSessionError(error_message="network error")

        complete = client.get("/api/models/download/progress", params={"sessionId": "done-session"})
        assert complete.status_code == 200
        assert complete.json()["status"] == "complete"

        failed = client.get("/api/models/download/progress", params={"sessionId": "err-session"})
        assert failed.status_code == 200
        assert failed.json()["status"] == "error"
        assert failed.json()["error"] == "network error"


class TestModelDownloads:
    def test_download_start_success(self, client, test_state):
        response = client.post(
            "/api/models/download",
            json={"type": "download", "cp_ids": [IMG_GEN_MODEL_CP_ID]},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "started"
        assert _cp_path(test_state, IMG_GEN_MODEL_CP_ID).exists()

    def test_download_conflicts_when_another_session_is_running(self, client, test_state):
        test_state.downloads.start_download({"ltx-2.3-22b-distilled"})
        response = client.post(
            "/api/models/download",
            json={"type": "download", "cp_ids": [IMG_GEN_MODEL_CP_ID]},
        )
        assert_http_error(response, status_code=409, code="DOWNLOAD_ALREADY_RUNNING")

    def test_upgrade_without_downloaded_model_is_rejected(self, client):
        response = client.post(
            "/api/models/download",
            json={"type": "upgrade", "cp_ids": [_current_ltx_spec().model_cp]},
        )
        assert_http_error(response, status_code=409, code="NO_DOWNLOADED_LTX_MODEL")

    def test_upgrade_raises_500_for_internal_ltx_mapping_inconsistency(self, test_state, monkeypatch):
        monkeypatch.setattr(test_state.models, "_current_downloaded_ltx_model_id", lambda: "ltx-legacy")
        monkeypatch.setattr(models_handler_module, "get_ltx_model_id_for_cp", lambda cp_id: None)

        with pytest.raises(HTTPError) as exc_info:
            test_state.models.resolve_upgrade_download({_current_ltx_spec().model_cp})

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "INVALID_LTX_MODEL_CONFIG"

    def test_upgrade_raises_500_when_latest_ltx_model_is_not_relevant(self, test_state, monkeypatch):
        monkeypatch.setattr(test_state.models, "_current_downloaded_ltx_model_id", lambda: "ltx-legacy")
        monkeypatch.setattr(models_handler_module, "get_latest_ltx_model_id", lambda: "ltx-2.3-22b-distilled")
        monkeypatch.setattr(models_handler_module, "get_ltx_model_id_for_cp", lambda cp_id: "ltx-2.3-22b-distilled")

        original_get_ltx_model_spec = models_handler_module.get_ltx_model_spec

        def _get_ltx_model_spec(model_id):
            spec = original_get_ltx_model_spec(model_id)
            if model_id == "ltx-2.3-22b-distilled":
                return replace(spec, relevance=LTXLocalModelDeprecated())
            return spec

        monkeypatch.setattr(models_handler_module, "get_ltx_model_spec", _get_ltx_model_spec)

        with pytest.raises(HTTPError) as exc_info:
            test_state.models.resolve_upgrade_download({_current_ltx_spec().model_cp})

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "INVALID_LTX_MODEL_CONFIG"

    def test_download_error_is_reported(self, client, test_state):
        test_state.model_downloader.fail_next = RuntimeError("Connection refused")

        response = client.post(
            "/api/models/download",
            json={"type": "download", "cp_ids": [IMG_GEN_MODEL_CP_ID]},
        )
        assert response.status_code == 200
        session_id = response.json()["sessionId"]

        progress = client.get("/api/models/download/progress", params={"sessionId": session_id})
        assert progress.status_code == 200
        assert progress.json()["status"] == "error"

    def test_download_uses_progress_callback(self, client, test_state):
        response = client.post(
            "/api/models/download",
            json={"type": "download", "cp_ids": [IMG_GEN_MODEL_CP_ID]},
        )
        assert response.status_code == 200
        assert test_state.model_downloader.calls
        assert all(call["on_progress"] is not None for call in test_state.model_downloader.calls)

    def test_failed_download_cleans_staging_dir(self, test_state):
        test_state.model_downloader.fail_next = RuntimeError("network error")
        test_state.downloads.start_model_download(download_type="download", cp_ids={IMG_GEN_MODEL_CP_ID})
        assert len(test_state.task_runner.errors) == 1
        assert not resolve_downloading_dir(test_state.config.default_models_dir).exists()


class TestCheckpointDeletion:
    def test_delete_missing_checkpoint_is_noop(self, client):
        response = client.request(
            "DELETE",
            "/api/models/delete",
            json={"cp_ids": [IMG_GEN_MODEL_CP_ID]},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_delete_rejects_current_ltx_bundle(self, client, create_fake_model_files):
        create_fake_model_files()
        response = client.request(
            "DELETE",
            "/api/models/delete",
            json={"cp_ids": [_current_ltx_spec().model_cp]},
        )
        assert_http_error(response, status_code=409, code="DELETE_PROTECTED_CHECKPOINT")

    def test_delete_removes_non_protected_checkpoint(self, client, test_state):
        img_gen_path = _cp_path(test_state, IMG_GEN_MODEL_CP_ID)
        img_gen_path.mkdir(parents=True, exist_ok=True)
        (img_gen_path / "model.safetensors").write_bytes(b"\x00" * 1024)

        response = client.request(
            "DELETE",
            "/api/models/delete",
            json={"cp_ids": [IMG_GEN_MODEL_CP_ID]},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert not img_gen_path.exists()


class TestCheckpointPathAndLoadFromPath:
    def test_checkpoint_path_reports_exists(self, client, test_state):
        # Not present yet.
        r = client.get("/api/models/checkpoint-path", params={"cp_id": IMG_GEN_MODEL_CP_ID})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["cp_id"] == IMG_GEN_MODEL_CP_ID
        assert body["exists"] is False
        assert body["path"].endswith("Z-Image-Turbo")

        # Materialize the folder with a file → exists flips to True.
        dst = _cp_path(test_state, IMG_GEN_MODEL_CP_ID)
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "model.safetensors").write_bytes(b"\x00")
        r = client.get("/api/models/checkpoint-path", params={"cp_id": IMG_GEN_MODEL_CP_ID})
        assert r.status_code == 200
        assert r.json()["exists"] is True

    def test_load_from_path_links_or_copies_folder(self, client, test_state, tmp_path):
        # A user-supplied folder containing the Z-Image checkpoint.
        src = tmp_path / "my-z-image"
        src.mkdir()
        (src / "model.safetensors").write_bytes(b"\x00" * 1024)
        (src / "config.json").write_bytes(b"{}")

        r = client.post(
            "/api/models/load-from-path",
            json={"cp_id": IMG_GEN_MODEL_CP_ID, "sourcePath": str(src)},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["cp_id"] == IMG_GEN_MODEL_CP_ID
        assert body["method"] in ("linked", "copied")
        # The checkpoint is now considered downloaded at the expected path.
        dst = _cp_path(test_state, IMG_GEN_MODEL_CP_ID)
        assert dst.exists()
        assert (dst / "model.safetensors").exists()

    def test_load_from_path_rejects_missing_source(self, client):
        r = client.post(
            "/api/models/load-from-path",
            json={"cp_id": IMG_GEN_MODEL_CP_ID, "sourcePath": "C:/does/not/exist/Z-Image-Turbo"},
        )
        assert r.status_code == 400
        assert r.json()["code"] == "LOAD_SOURCE_NOT_FOUND"

    def test_load_from_path_rejects_already_present(self, client, test_state, tmp_path):
        # Target already exists (non-empty folder) → 409.
        dst = _cp_path(test_state, IMG_GEN_MODEL_CP_ID)
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "model.safetensors").write_bytes(b"\x00")
        src = tmp_path / "my-z-image"
        src.mkdir()
        (src / "model.safetensors").write_bytes(b"\x00")
        r = client.post(
            "/api/models/load-from-path",
            json={"cp_id": IMG_GEN_MODEL_CP_ID, "sourcePath": str(src)},
        )
        assert r.status_code == 409
        assert r.json()["code"] == "LOAD_TARGET_EXISTS"

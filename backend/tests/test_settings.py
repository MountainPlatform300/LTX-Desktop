"""Tests for GET /api/settings and POST /api/settings."""

from __future__ import annotations

import json
from pathlib import Path

from state.app_settings import AppSettings, UpdateSettingsRequest
from state import build_initial_state
from app_handler import ServiceBundle
from tests.conftest import TEST_ADMIN_TOKEN
from tests.fakes.services import FakeServices


class TestGetSettings:
    def test_default_settings(self, client, default_app_settings, test_state):
        r = client.get("/api/settings")
        assert r.status_code == 200
        data = r.json()
        assert data["useTorchCompile"] is False
        assert data["hasLtxApiKey"] is False
        assert data["userPrefersLtxApiVideoGenerations"] is False
        assert data["hasFalApiKey"] is False
        assert data["useLocalTextEncoder"] is False
        assert data["promptCacheSize"] == 100
        assert data["promptEnhancerEnabledT2V"] is True
        assert data["promptEnhancerEnabledI2V"] is False
        assert data["runpodVolumeSizeGb"] == 250
        assert data["hasGeminiApiKey"] is False
        assert data["seedLocked"] is False
        assert data["lockedSeed"] == 42
        # When no custom path is set, the response surfaces the runtime default
        # so the first-run UI can show the install location.
        assert data["modelsDir"] == str(test_state.config.default_models_dir)
        assert "fastModel" not in data
        assert "proModel" not in data
        assert "ltxApiKey" not in data
        assert "falApiKey" not in data
        assert "geminiApiKey" not in data

    def test_reflects_changed_settings(self, client, test_state):
        test_state.state.app_settings.use_torch_compile = True
        r = client.get("/api/settings")
        assert r.json()["useTorchCompile"] is True

    def test_has_api_key_true_when_set(self, client, test_state):
        test_state.state.app_settings.ltx_api_key = "test-key-123"
        r = client.get("/api/settings")
        data = r.json()
        assert data["hasLtxApiKey"] is True
        assert "ltxApiKey" not in data


class TestPostSettings:
    def test_update_single_field(self, client, test_state):
        r = client.post("/api/settings", json={"useTorchCompile": True})
        assert r.status_code == 200
        assert test_state.state.app_settings.use_torch_compile is True

    def test_update_multiple_fields(self, client, test_state):
        r = client.post("/api/settings", json={"useTorchCompile": True, "promptCacheSize": 42})
        assert r.status_code == 200
        assert test_state.state.app_settings.use_torch_compile is True
        assert test_state.state.app_settings.prompt_cache_size == 42

    def test_prompt_cache_size_clamped_max(self, client, test_state):
        r = client.post("/api/settings", json={"promptCacheSize": 5000})
        assert r.status_code == 200
        assert test_state.state.app_settings.prompt_cache_size <= 1000

    def test_prompt_cache_size_clamped_min(self, client, test_state):
        r = client.post("/api/settings", json={"promptCacheSize": -10})
        assert r.status_code == 200
        assert test_state.state.app_settings.prompt_cache_size >= 0

    def test_locked_seed_clamped_range(self, client, test_state):
        r = client.post("/api/settings", json={"lockedSeed": 9_999_999_999})
        assert r.status_code == 200
        assert test_state.state.app_settings.locked_seed == 2_147_483_647

    def test_prompt_cache_shrinks_cache(self, client, test_state):
        te = test_state.state.text_encoder
        assert te is not None
        for i in range(5):
            te.prompt_cache[(f"key_{i}", False)] = f"value_{i}"  # type: ignore[assignment]

        r = client.post("/api/settings", json={"promptCacheSize": 2})
        assert r.status_code == 200
        assert len(te.prompt_cache) <= 2

    def test_update_api_keys(self, client, test_state):
        r = client.post(
            "/api/settings",
            json={
                "ltxApiKey": "ltx-key-abc",
                "geminiApiKey": "gemini-key-xyz",
                "falApiKey": "fal-key-123",
            },
        )
        assert r.status_code == 200
        assert test_state.state.app_settings.ltx_api_key == "ltx-key-abc"
        assert test_state.state.app_settings.gemini_api_key == "gemini-key-xyz"
        assert test_state.state.app_settings.fal_api_key == "fal-key-123"

    def test_update_user_prefers_api_video_generations(self, client, test_state):
        r = client.post("/api/settings", json={"userPrefersLtxApiVideoGenerations": True})
        assert r.status_code == 200
        assert test_state.state.app_settings.user_prefers_ltx_api_video_generations is True

    def test_empty_string_does_not_erase_key(self, client, test_state):
        test_state.state.app_settings.ltx_api_key = "real-key"
        test_state.state.app_settings.fal_api_key = "fal-key"
        r = client.post("/api/settings", json={"ltxApiKey": "", "falApiKey": ""})
        assert r.status_code == 200
        assert test_state.state.app_settings.ltx_api_key == "real-key"
        assert test_state.state.app_settings.fal_api_key == "fal-key"

    def test_omitted_key_does_not_erase_key(self, client, test_state):
        test_state.state.app_settings.ltx_api_key = "real-key"
        r = client.post("/api/settings", json={"useTorchCompile": True})
        assert r.status_code == 200
        assert test_state.state.app_settings.ltx_api_key == "real-key"

    def test_unknown_field_rejected(self, client):
        r = client.post("/api/settings", json={"unknownSetting": True})
        assert r.status_code == 422


class TestModelsDirAdminGuard:
    def test_models_dir_requires_admin_token(self, client, test_state):
        r = client.post("/api/settings", json={"modelsDir": "/tmp/new-models"})
        assert r.status_code == 403

    def test_models_dir_with_wrong_admin_token(self, client, test_state):
        r = client.post(
            "/api/settings",
            json={"modelsDir": "/tmp/new-models"},
            headers={"X-Admin-Token": "wrong-token"},
        )
        assert r.status_code == 403

    def test_models_dir_with_valid_admin_token(self, client, test_state):
        r = client.post(
            "/api/settings",
            json={"modelsDir": "/tmp/new-models"},
            headers={"X-Admin-Token": TEST_ADMIN_TOKEN},
        )
        assert r.status_code == 200
        assert test_state.state.app_settings.models_dir == "/tmp/new-models"

    def test_non_admin_fields_without_admin_token(self, client, test_state):
        r = client.post("/api/settings", json={"useTorchCompile": True})
        assert r.status_code == 200
        assert test_state.state.app_settings.use_torch_compile is True

    def test_effective_models_dir_uses_custom(self, client, test_state):
        test_state.state.app_settings.models_dir = "/custom/models"
        assert test_state.models.models_dir == Path("/custom/models")

    def test_effective_models_dir_fallback(self, client, test_state):
        assert test_state.state.app_settings.models_dir == ""
        assert test_state.models.models_dir == test_state.config.default_models_dir

    def test_models_dir_persists_and_loads(self, client, test_state, default_app_settings):
        r = client.post(
            "/api/settings",
            json={"modelsDir": "/tmp/persisted-models"},
            headers={"X-Admin-Token": TEST_ADMIN_TOKEN},
        )
        assert r.status_code == 200

        fake_services = FakeServices()
        bundle = ServiceBundle(
            http=fake_services.http,
            gpu_cleaner=fake_services.gpu_cleaner,
            model_downloader=fake_services.model_downloader,
            gpu_info=fake_services.gpu_info,
            video_processor=fake_services.video_processor,
            text_encoder=fake_services.text_encoder,
            task_runner=fake_services.task_runner,
            ltx_api_client=fake_services.ltx_api_client,
            zit_api_client=fake_services.zit_api_client,
            fast_video_pipeline_class=type(fake_services.fast_video_pipeline),
            image_generation_pipeline_class=type(fake_services.image_generation_pipeline),
            ic_lora_pipeline_class=type(fake_services.ic_lora_pipeline),
            depth_processor_pipeline_class=type(fake_services.depth_processor_pipeline),
            pose_processor_pipeline_class=type(fake_services.pose_processor_pipeline),
            a2v_pipeline_class=type(fake_services.a2v_pipeline),
            retake_pipeline_class=type(fake_services.retake_pipeline),
            image_edit_pipeline_class=type(fake_services.image_edit_pipeline),
            trainer_target=fake_services.trainer_target,
            video_captioner=fake_services.video_captioner,
            clip_processor=fake_services.clip_processor,
            image_editor=fake_services.image_editor,
            video_restyler=fake_services.video_restyler,
            pexels_client=fake_services.pexels_client,
        )
        loaded = build_initial_state(test_state.config, default_app_settings.model_copy(deep=True), service_bundle=bundle)
        assert loaded.state.app_settings.models_dir == "/tmp/persisted-models"
        assert loaded.models.models_dir == Path("/tmp/persisted-models")


class TestSettingsPersistence:
    def _new_state(self, test_state, default_app_settings):
        fake_services = FakeServices()
        bundle = ServiceBundle(
            http=fake_services.http,
            gpu_cleaner=fake_services.gpu_cleaner,
            model_downloader=fake_services.model_downloader,
            gpu_info=fake_services.gpu_info,
            video_processor=fake_services.video_processor,
            text_encoder=fake_services.text_encoder,
            task_runner=fake_services.task_runner,
            ltx_api_client=fake_services.ltx_api_client,
            zit_api_client=fake_services.zit_api_client,
            fast_video_pipeline_class=type(fake_services.fast_video_pipeline),
            image_generation_pipeline_class=type(fake_services.image_generation_pipeline),
            ic_lora_pipeline_class=type(fake_services.ic_lora_pipeline),
            depth_processor_pipeline_class=type(fake_services.depth_processor_pipeline),
            pose_processor_pipeline_class=type(fake_services.pose_processor_pipeline),
            a2v_pipeline_class=type(fake_services.a2v_pipeline),
            retake_pipeline_class=type(fake_services.retake_pipeline),
            image_edit_pipeline_class=type(fake_services.image_edit_pipeline),
            trainer_target=fake_services.trainer_target,
            video_captioner=fake_services.video_captioner,
            clip_processor=fake_services.clip_processor,
            image_editor=fake_services.image_editor,
            video_restyler=fake_services.video_restyler,
            pexels_client=fake_services.pexels_client,
        )
        return build_initial_state(test_state.config, default_app_settings.model_copy(deep=True), service_bundle=bundle)

    def test_load_settings_clamps_from_disk_and_ignores_removed_fields(self, test_state, default_app_settings):
        test_state.config.settings_file.write_text(
            json.dumps(
                {
                    "prompt_cache_size": 5000,
                    "locked_seed": -55,
                    "fast_model": {"use_upscaler": False},
                    "pro_model": {"steps": 999},
                }
            ),
            encoding="utf-8",
        )

        loaded = self._new_state(test_state, default_app_settings)
        assert loaded.state.app_settings.prompt_cache_size == 1000
        assert loaded.state.app_settings.locked_seed == 0
        assert "fast_model" not in loaded.state.app_settings.model_dump(by_alias=False)
        assert "pro_model" not in loaded.state.app_settings.model_dump(by_alias=False)

    def test_legacy_prompt_enhancer_key_migrates(self, test_state, default_app_settings):
        test_state.config.settings_file.write_text(
            json.dumps({"prompt_enhancer_enabled": False}),
            encoding="utf-8",
        )

        loaded = self._new_state(test_state, default_app_settings)
        assert loaded.state.app_settings.prompt_enhancer_enabled_t2v is False
        assert loaded.state.app_settings.prompt_enhancer_enabled_i2v is False

    def test_user_prefers_api_video_generations_persists(self, client, test_state, default_app_settings):
        r = client.post("/api/settings", json={"userPrefersLtxApiVideoGenerations": True})
        assert r.status_code == 200
        assert test_state.state.app_settings.user_prefers_ltx_api_video_generations is True

        loaded = self._new_state(test_state, default_app_settings)
        assert loaded.state.app_settings.user_prefers_ltx_api_video_generations is True

    def test_credentials_are_encrypted_outside_settings_json(
        self, client, test_state, default_app_settings
    ):
        response = client.post(
            "/api/settings",
            json={"runpodApiKey": "runpod-secret-value"},
        )
        assert response.status_code == 200

        settings_text = test_state.config.settings_file.read_text(encoding="utf-8")
        vault_path = test_state.config.app_data_dir / "credentials.vault"
        vault_bytes = vault_path.read_bytes()
        assert "runpod-secret-value" not in settings_text
        assert "runpod_api_key" not in settings_text
        assert b"runpod-secret-value" not in vault_bytes

        loaded = self._new_state(test_state, default_app_settings)
        assert loaded.state.app_settings.runpod_api_key == "runpod-secret-value"

    def test_plaintext_credentials_migrate_to_encrypted_vault(
        self, test_state, default_app_settings
    ):
        test_state.config.settings_file.write_text(
            json.dumps(
                {
                    "runpod_api_key": "legacy-plaintext-secret",
                    "prompt_cache_size": 12,
                }
            ),
            encoding="utf-8",
        )

        loaded = self._new_state(test_state, default_app_settings)

        assert loaded.state.app_settings.runpod_api_key == "legacy-plaintext-secret"
        assert loaded.state.app_settings.prompt_cache_size == 12
        assert "legacy-plaintext-secret" not in test_state.config.settings_file.read_text(
            encoding="utf-8"
        )
        assert b"legacy-plaintext-secret" not in (
            test_state.config.app_data_dir / "credentials.vault"
        ).read_bytes()


class TestLegacyTrainerRepoMigration:
    def test_legacy_ltx_video_trainer_url_is_migrated(self):
        s = AppSettings.model_validate(
            {"lora_trainer_repo_url": "https://github.com/Lightricks/LTX-Video-Trainer.git"}
        )
        assert s.lora_trainer_repo_url == "https://github.com/Lightricks/LTX-2.git"

    def test_valid_ltx2_url_is_preserved(self):
        url = "https://github.com/Lightricks/LTX-2.git"
        s = AppSettings.model_validate({"lora_trainer_repo_url": url})
        assert s.lora_trainer_repo_url == url

    def test_custom_fork_url_is_rejected_in_favor_of_audited_source(self):
        url = "https://github.com/acme/LTX-2-fork.git"
        s = AppSettings.model_validate({"lora_trainer_repo_url": url})
        assert s.lora_trainer_repo_url == "https://github.com/Lightricks/LTX-2.git"

    def test_official_main_ref_is_migrated_to_verified_revision(self):
        s = AppSettings.model_validate(
            {
                "lora_trainer_repo_url": "https://github.com/Lightricks/LTX-2.git",
                "lora_trainer_repo_ref": "main",
            }
        )
        assert s.lora_trainer_repo_ref == "9377758131b1ffde4b7f766804590a6617bf2ab9"

    def test_custom_fork_main_ref_is_replaced_with_pinned_revision(self):
        s = AppSettings.model_validate(
            {
                "lora_trainer_repo_url": "https://github.com/acme/LTX-2-fork.git",
                "lora_trainer_repo_ref": "main",
            }
        )
        assert s.lora_trainer_repo_ref != "main"

    def test_stale_ltx2_model_repo_is_migrated_to_ltx23(self):
        # The 22B checkpoint lives in LTX-2.3; the old LTX-2 (19B) repo value
        # self-heals so the default checkpoint file resolves.
        s = AppSettings.model_validate({"lora_model_hf_repo": "Lightricks/LTX-2"})
        assert s.lora_model_hf_repo == "Lightricks/LTX-2.3"

    def test_explicit_ltx23_model_repo_is_preserved(self):
        s = AppSettings.model_validate({"lora_model_hf_repo": "Lightricks/LTX-2.3"})
        assert s.lora_model_hf_repo == "Lightricks/LTX-2.3"


class TestSettingsSchemaDrift:
    def test_update_request_tracks_app_settings_fields(self):
        assert set(AppSettings.model_fields) == set(UpdateSettingsRequest.model_fields)

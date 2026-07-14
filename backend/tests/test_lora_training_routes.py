"""Integration tests for the LoRA-trainer HTTP routes.

These assert CRUD + validation + typed-error mapping using immediate
responses. They deliberately avoid the reconciler-driven remote flow
(covered in `test_lora_training_runner.py`): the background reconciler
only acts on datasets in `uploading`, and these tests never upload, so
created entities stay in their initial state regardless of the runner
thread the `client` lifespan starts.
"""

from __future__ import annotations

import pytest


def _wsl_installed(client) -> bool:
    """True if the real eligibility probe reports WSL2 as installed.

    The local-eligibility endpoint hits the real `LocalTrainerTarget` (no
    mocks), so this reflects the actual host. Used to skip tests whose premise
    is "no WSL2" on machines that have it — the no-mock policy means we can't
    fake WSL's absence.
    """
    r = client.get("/api/lora/local-eligibility")
    assert r.status_code == 200, r.text
    return bool(r.json()["wslInstalled"])


def _create_dataset(client, name: str = "ds", trigger: str | None = "TOK") -> dict:
    r = client.post(
        "/api/lora/datasets",
        json={
            "name": name,
            "triggerWord": trigger,
            "clips": [{"localPath": "/tmp/a.mp4", "caption": "a cat"}],
        },
    )
    assert r.status_code == 200, r.text
    return r.json()


class TestDatasets:
    def test_create_and_list(self, client) -> None:
        created = _create_dataset(client)
        assert created["status"] == "draft"
        assert created["triggerWord"] == "TOK"
        assert len(created["clips"]) == 1
        assert created["clips"][0]["id"]  # id minted server-side

        r = client.get("/api/lora/datasets")
        assert r.status_code == 200
        body = r.json()
        assert [d["id"] for d in body["datasets"]] == [created["id"]]

    def test_create_with_originating_project(self, client) -> None:
        r = client.post(
            "/api/lora/datasets",
            json={
                "name": "from-genspace",
                "clips": [{"localPath": "/tmp/a.mp4", "caption": "a cat"}],
                "originatingProjectId": "proj-123",
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["originatingProjectId"] == "proj-123"

    def test_upload_stamps_selected_provider(self, client, test_state) -> None:
        created = _create_dataset(client)

        response = client.post(
            f"/api/lora/datasets/{created['id']}/upload",
            json={"provider": "local"},
        )

        assert response.status_code == 200, response.text
        stored = test_state.lora_training.get_dataset(created["id"])
        assert stored is not None
        assert stored.provider == "local"

    def test_update_draft(self, client) -> None:
        created = _create_dataset(client)
        r = client.patch(
            f"/api/lora/datasets/{created['id']}",
            json={"name": "renamed", "triggerWord": "NEW"},
        )
        assert r.status_code == 200
        assert r.json()["name"] == "renamed"
        assert r.json()["triggerWord"] == "NEW"

    def test_update_clips_preserves_existing_ids(self, client) -> None:
        created = _create_dataset(client)
        clip_id = created["clips"][0]["id"]

        response = client.patch(
            f"/api/lora/datasets/{created['id']}",
            json={
                "clips": [
                    {
                        "id": clip_id,
                        "localPath": "/tmp/a.mp4",
                        "caption": "updated caption",
                    }
                ]
            },
        )

        assert response.status_code == 200, response.text
        assert response.json()["clips"][0]["id"] == clip_id
        assert response.json()["clips"][0]["caption"] == "updated caption"

    def test_create_dataset_mints_ids_instead_of_accepting_foreign_ids(self, client) -> None:
        response = client.post(
            "/api/lora/datasets",
            json={
                "name": "copy",
                "clips": [
                    {
                        "id": "id-from-another-dataset",
                        "localPath": "/tmp/copy.mp4",
                    }
                ],
            },
        )

        assert response.status_code == 200, response.text
        assert response.json()["clips"][0]["id"] != "id-from-another-dataset"

    def test_update_missing_404(self, client) -> None:
        r = client.patch("/api/lora/datasets/nope", json={"name": "x"})
        assert r.status_code == 404

    def test_delete_draft(self, client) -> None:
        created = _create_dataset(client)
        r = client.delete(f"/api/lora/datasets/{created['id']}")
        assert r.status_code == 204
        assert client.get("/api/lora/datasets").json()["datasets"] == []

    def test_delete_active_training_does_not_release_workspace(
        self, client, test_state, fake_services
    ) -> None:
        from state.lora_training_state import LoraClip, TargetHandle, TrainingConfig

        handler = test_state.lora_training
        dataset = handler.create_dataset(
            name="active",
            trigger_word=None,
            clips=[LoraClip(id="clip", local_path="/tmp/a.mp4", caption="a cat")],
        )
        handler.request_upload(dataset.id)
        handler.mark_dataset_uploaded(
            dataset.id,
            remote_dataset_dir="/workspace/datasets/active",
            handle=TargetHandle(provider="runpod", pod_id="pod-1"),
        )
        preprocessed = handler.create_preprocessing(
            dataset_id=dataset.id,
            resolution_buckets="768x448x89",
            with_audio=False,
            auto_caption=False,
            captioner_type="qwen_omni",
        )
        handler.mark_preprocess_ready(
            preprocessed.id,
            remote_precomputed_dir="/workspace/.precomputed/active",
        )
        handler.start_training(
            preprocessed_id=preprocessed.id,
            name="run",
            config=TrainingConfig(),
            provider="runpod",
        )
        releases_before = fake_services.trainer_target.released

        response = client.delete(f"/api/lora/datasets/{dataset.id}")

        assert response.status_code == 409
        assert fake_services.trainer_target.released == releases_before
        assert handler.get_dataset(dataset.id) is not None

    def test_cancel_upload_missing_404(self, client) -> None:
        r = client.post("/api/lora/datasets/nope/cancel")
        assert r.status_code == 404

    def test_cancel_upload_non_uploading_409(self, client) -> None:
        # A freshly-created dataset is `draft`, not `uploading` -> conflict.
        created = _create_dataset(client)
        r = client.post(f"/api/lora/datasets/{created['id']}/cancel")
        assert r.status_code == 409

    def test_rename_dataset(self, client) -> None:
        created = _create_dataset(client)
        r = client.post(
            f"/api/lora/datasets/{created['id']}/rename", json={"name": "  Zeev  "}
        )
        assert r.status_code == 200, r.text
        assert r.json()["name"] == "Zeev"

    def test_rename_dataset_empty_409(self, client) -> None:
        created = _create_dataset(client)
        r = client.post(
            f"/api/lora/datasets/{created['id']}/rename", json={"name": "   "}
        )
        assert r.status_code == 409

    def test_rename_dataset_missing_404(self, client) -> None:
        r = client.post("/api/lora/datasets/nope/rename", json={"name": "x"})
        assert r.status_code == 404


class TestFolders:
    def test_create_list_move_dataset(self, client) -> None:
        # Folders surface in the datasets listing.
        f = client.post("/api/lora/folders", json={"name": "People", "parentId": None})
        assert f.status_code == 200, f.text
        folder = f.json()
        assert folder["parentId"] is None

        listed = client.get("/api/lora/datasets").json()
        assert [fo["id"] for fo in listed["folders"]] == [folder["id"]]

        ds = _create_dataset(client)
        moved = client.post(
            f"/api/lora/datasets/{ds['id']}/move", json={"folderId": folder["id"]}
        )
        assert moved.status_code == 200, moved.text
        assert moved.json()["folderId"] == folder["id"]

        back = client.post(
            f"/api/lora/datasets/{ds['id']}/move", json={"folderId": None}
        )
        assert back.status_code == 200
        assert back.json()["folderId"] is None

    def test_create_folder_missing_parent_404(self, client) -> None:
        r = client.post("/api/lora/folders", json={"name": "x", "parentId": "ghost"})
        assert r.status_code == 404

    def test_rename_folder(self, client) -> None:
        folder = client.post("/api/lora/folders", json={"name": "A"}).json()
        r = client.patch(f"/api/lora/folders/{folder['id']}", json={"name": "B"})
        assert r.status_code == 200
        assert r.json()["name"] == "B"

    def test_move_folder_rejects_cycle(self, client) -> None:
        root = client.post("/api/lora/folders", json={"name": "Root"}).json()
        child = client.post(
            "/api/lora/folders", json={"name": "Child", "parentId": root["id"]}
        ).json()
        grand = client.post(
            "/api/lora/folders", json={"name": "Grand", "parentId": child["id"]}
        ).json()
        # Moving root under its own descendant must be rejected.
        r = client.post(
            f"/api/lora/folders/{root['id']}/move", json={"parentId": grand["id"]}
        )
        assert r.status_code == 409

    def test_delete_folder_non_recursive_moves_up(self, client) -> None:
        root = client.post("/api/lora/folders", json={"name": "Root"}).json()
        inner = client.post(
            "/api/lora/folders", json={"name": "Inner", "parentId": root["id"]}
        ).json()
        ds = _create_dataset(client)
        client.post(
            f"/api/lora/datasets/{ds['id']}/move", json={"folderId": inner["id"]}
        )

        r = client.delete(f"/api/lora/folders/{inner['id']}")
        assert r.status_code == 204

        folders = client.get("/api/lora/datasets").json()["folders"]
        assert [f["id"] for f in folders] == [root["id"]]
        # Dataset reparented up to root.
        listed = client.get("/api/lora/datasets").json()["datasets"]
        assert next(d for d in listed if d["id"] == ds["id"])["folderId"] == root["id"]

    def test_delete_folder_recursive_deletes_contents(self, client) -> None:
        root = client.post("/api/lora/folders", json={"name": "Root"}).json()
        inner = client.post(
            "/api/lora/folders", json={"name": "Inner", "parentId": root["id"]}
        ).json()
        ds = _create_dataset(client)
        client.post(
            f"/api/lora/datasets/{ds['id']}/move", json={"folderId": inner["id"]}
        )

        r = client.delete(f"/api/lora/folders/{inner['id']}?recursive=true")
        assert r.status_code == 204

        listed = client.get("/api/lora/datasets").json()
        assert [f["id"] for f in listed["folders"]] == [root["id"]]
        assert all(d["id"] != ds["id"] for d in listed["datasets"])

    def test_move_dataset_missing_folder_404(self, client) -> None:
        ds = _create_dataset(client)
        r = client.post(
            f"/api/lora/datasets/{ds['id']}/move", json={"folderId": "ghost"}
        )
        assert r.status_code == 404


class TestPreprocessing:
    def test_rejects_non_uploaded_dataset(self, client) -> None:
        created = _create_dataset(client)
        r = client.post(
            "/api/lora/preprocessed",
            json={"datasetId": created["id"], "resolutionBuckets": "768x448x89"},
        )
        assert r.status_code == 409

    def test_rejects_invalid_resolution(self, client) -> None:
        # Missing dataset is checked before resolution, so use a bogus id
        # to confirm the 404 path, then a separate validation path is
        # covered by the handler unit test.
        r = client.post(
            "/api/lora/preprocessed",
            json={"datasetId": "missing", "resolutionBuckets": "768x448x89"},
        )
        assert r.status_code == 404

    def test_list_empty(self, client) -> None:
        r = client.get("/api/lora/preprocessed")
        assert r.status_code == 200
        assert r.json()["items"] == []

    def test_manual_preprocess_auto_matches_low_vram_preset(self, client, test_state) -> None:
        # A sub-80 GB RunPod GPU -> the manual preprocess route auto-picks the
        # low_vram preset (8-bit text encoder), matching the one-click pipeline,
        # so process_dataset.py won't OOM loading Gemma3 12B in bf16.
        from state.lora_training_state import TargetHandle

        client.post("/api/settings", json={"runpodGpuVramGb": 32})
        created = _create_dataset(client)
        # Bypass the real upload (mark uploaded directly); the runner only acts
        # on `uploading` datasets, so this `uploaded` dataset is left alone.
        test_state.lora_training.mark_dataset_uploaded(
            created["id"],
            remote_dataset_dir="/workspace/datasets/x",
            handle=TargetHandle(provider="runpod", pod_id="p1"),
        )
        r = client.post(
            "/api/lora/preprocessed",
            json={"datasetId": created["id"], "resolutionBuckets": "768x448x89"},
        )
        assert r.status_code == 200, r.text
        pre = test_state.lora_training.get_preprocessed(r.json()["id"])
        assert pre is not None
        assert pre.preset == "low_vram"

    def test_manual_preprocess_keeps_standard_on_big_gpu(self, client, test_state) -> None:
        # An 80 GB+ GPU keeps the standard (bf16) preset.
        from state.lora_training_state import TargetHandle

        client.post("/api/settings", json={"runpodGpuVramGb": 80})
        created = _create_dataset(client)
        test_state.lora_training.mark_dataset_uploaded(
            created["id"],
            remote_dataset_dir="/workspace/datasets/x",
            handle=TargetHandle(provider="runpod", pod_id="p1"),
        )
        r = client.post(
            "/api/lora/preprocessed",
            json={"datasetId": created["id"], "resolutionBuckets": "768x448x89"},
        )
        assert r.status_code == 200, r.text
        pre = test_state.lora_training.get_preprocessed(r.json()["id"])
        assert pre is not None
        assert pre.preset == "standard"


class TestTraining:
    def test_start_missing_preprocessed_404(self, client) -> None:
        r = client.post(
            "/api/lora/training",
            json={"preprocessedId": "nope", "name": "run1"},
        )
        assert r.status_code == 404

    def test_list_empty(self, client) -> None:
        r = client.get("/api/lora/training")
        assert r.status_code == 200
        assert r.json()["items"] == []

    def test_cancel_missing_404(self, client) -> None:
        r = client.post("/api/lora/training/nope/cancel")
        assert r.status_code == 404

    def test_retry_download_missing_404(self, client) -> None:
        r = client.post("/api/lora/training/nope/retry-download")
        assert r.status_code == 404

    def test_start_blocks_gpu_below_min_vram(self, client) -> None:
        # A sub-32 GB GPU can't train at all — the start route rejects it up
        # front (before the preprocessed lookup) with a clear, actionable error.
        client.post("/api/settings", json={"runpodGpuVramGb": 24})
        r = client.post(
            "/api/lora/training",
            json={"preprocessedId": "nope", "name": "run"},
        )
        assert r.status_code == 422, r.text
        assert "VRAM" in r.json()["message"]

    def test_start_defaults_to_runpod_when_provider_omitted(self, client) -> None:
        # Omitting `provider` keeps the original RunPod behavior: the local
        # eligibility gate is skipped entirely, so the request flows straight
        # to the (missing) preprocessed lookup and 404s exactly as before.
        r = client.post(
            "/api/lora/training",
            json={"preprocessedId": "nope", "name": "run"},
        )
        assert r.status_code == 404, r.text

    def test_start_local_rejected_without_wsl(self, client) -> None:
        # Only meaningful where WSL2 is absent; skip on machines that have it
        # (the no-mock policy means we can't fake WSL's absence).
        if _wsl_installed(client):
            pytest.skip("WSL2 is installed on this machine; the local-rejected path can't be exercised without mocking.")
        # A `provider: "local"` run is impossible without WSL2 — the route
        # probes eligibility and 422s with the reason BEFORE touching the
        # preprocessed ledger (so it's not a 404).
        r = client.post(
            "/api/lora/training",
            json={"preprocessedId": "nope", "name": "run", "provider": "local"},
        )
        assert r.status_code == 422, r.text
        body = r.json()
        assert body["code"] == "LOCAL_TRAINER_UNAVAILABLE"
        assert body["message"].strip() != ""


class TestTrainingPipeline:
    def _create_dataset(self, client) -> dict:
        return _create_dataset(client, name="pipeline-ds")

    def test_pipeline_defaults_to_runpod_when_provider_omitted(self, client) -> None:
        # No provider => RunPod => no eligibility probe; the dataset flips to
        # `uploading` and records the default provider, exactly as before.
        ds = self._create_dataset(client)
        r = client.post(
            "/api/lora/training-pipeline",
            json={"datasetId": ds["id"], "name": "run1"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "uploading"
        assert body["workspacePolicy"] == "ephemeral_any_region"
        assert body["cacheVolumeId"] is None

    def test_gpu_recovery_exposes_and_updates_pending_pipeline(
        self, client, test_state
    ) -> None:
        # This test drives the recovery transition directly. Stop the lifespan
        # worker so it cannot concurrently stage the intentionally minimal
        # fixture and overwrite gpu_selection_required with upload_failed.
        test_state.lora_training_runner.stop()
        ds = self._create_dataset(client)
        started = client.post(
            "/api/lora/training-pipeline",
            json={
                "datasetId": ds["id"],
                "name": "original run",
                "description": "Removes foreground subjects",
                "resolutionBuckets": "512x512x49",
            },
        )
        assert started.status_code == 200, started.text
        test_state.lora_training.require_dataset_gpu_selection(
            ds["id"], "Selected GPU is no longer available"
        )

        waiting = client.get("/api/lora/datasets").json()["datasets"][0]
        assert waiting["status"] == "gpu_selection_required"
        pending = waiting["pendingPipeline"]
        assert pending["name"] == "original run"
        assert pending["description"] == "Removes foreground subjects"
        assert pending["resolutionBuckets"] == "512x512x49"

        edited_config = {**pending["config"], "steps": 777}
        resumed = client.post(
            "/api/lora/training-pipeline",
            json={
                "datasetId": ds["id"],
                "name": "edited run",
                "description": "Reconstructs a clean background",
                "resolutionBuckets": "512x512x49",
                "autoCaption": False,
                "config": edited_config,
            },
        )
        assert resumed.status_code == 200, resumed.text
        stored = test_state.lora_training.get_dataset(ds["id"])
        assert stored is not None and stored.auto_pipeline is not None
        assert stored.auto_pipeline.training.name == "edited run"
        assert stored.auto_pipeline.training.description == "Reconstructs a clean background"
        assert stored.auto_pipeline.resolution_buckets == "512x512x49"
        assert stored.auto_pipeline.auto_caption is False
        assert stored.auto_pipeline.training.config.steps == 777

    def test_pipeline_persists_one_time_ephemeral_policy(
        self, client, test_state
    ) -> None:
        client.post(
            "/api/settings",
            json={
                "runpodKeepModelCached": True,
                "runpodNetworkVolumeId": "primary-vol",
            },
        )
        ds = self._create_dataset(client)
        r = client.post(
            "/api/lora/training-pipeline",
            json={
                "datasetId": ds["id"],
                "name": "uncached-once",
                "workspacePolicy": "ephemeral_any_region",
            },
        )
        assert r.status_code == 200, r.text
        stored = test_state.lora_training.get_dataset(ds["id"])
        assert stored is not None
        assert stored.workspace_policy == "ephemeral_any_region"
        assert stored.cache_volume_id is None
        settings = test_state.settings.get_settings_snapshot()
        creds = test_state.lora_training_runner._credentials(
            settings,
            "runpod",
            workspace_policy=stored.workspace_policy,
            cache_volume_id=stored.cache_volume_id,
        )
        assert creds.runpod_network_volume_id == ""

    def test_explicit_primary_cache_requires_selected_volume(self, client) -> None:
        ds = self._create_dataset(client)
        r = client.post(
            "/api/lora/training-pipeline",
            json={
                "datasetId": ds["id"],
                "name": "cached",
                "workspacePolicy": "primary_cache",
            },
        )
        assert r.status_code == 409
        assert r.json()["code"] == "LORA_CACHE_NOT_CONFIGURED"

    def test_pipeline_auto_on_32gb_uses_rank_16(self, client, test_state) -> None:
        client.post("/api/settings", json={"runpodGpuVramGb": 32})
        ds = self._create_dataset(client)
        r = client.post(
            "/api/lora/training-pipeline",
            json={"datasetId": ds["id"], "name": "safe-auto"},
        )
        assert r.status_code == 200, r.text
        stored = test_state.lora_training.get_dataset(ds["id"])
        assert stored is not None
        assert stored.auto_pipeline is not None
        assert stored.auto_pipeline.resolution_buckets == "512x512x49"
        config = stored.auto_pipeline.training.config
        assert config.preset == "low_vram"
        assert config.rank == 16
        assert config.alpha == 16
        assert config.optimizer_type == "adamw8bit"
        assert config.quantization == "int8-quanto"
        assert config.load_text_encoder_in_8bit is True
        assert config.offload_optimizer_during_validation is True
        assert config.skip_initial_validation is True

    def test_pipeline_unknown_runpod_vram_uses_safe_defaults(
        self, client, test_state
    ) -> None:
        client.post("/api/settings", json={"runpodGpuVramGb": 0})
        ds = self._create_dataset(client)
        r = client.post(
            "/api/lora/training-pipeline",
            json={"datasetId": ds["id"], "name": "unknown-vram"},
        )
        assert r.status_code == 200, r.text
        stored = test_state.lora_training.get_dataset(ds["id"])
        assert stored is not None
        assert stored.auto_pipeline is not None
        config = stored.auto_pipeline.training.config
        assert config.preset == "low_vram"
        assert config.rank == 16

    def test_pipeline_risky_profile_requires_expert_override(
        self, client, test_state
    ) -> None:
        client.post("/api/settings", json={"runpodGpuVramGb": 32})
        profiles = client.get("/api/lora/profiles").json()["profiles"]
        standard_id = next(p["id"] for p in profiles if p["name"] == "Standard LoRA")
        ds = self._create_dataset(client)

        blocked = client.post(
            "/api/lora/training-pipeline",
            json={
                "datasetId": ds["id"],
                "name": "risky",
                "profileId": standard_id,
                "resolutionBuckets": "512x512x49",
            },
        )
        assert blocked.status_code == 422, blocked.text
        assert blocked.json()["code"] == "LORA_UNSAFE_TRAINING_OVERRIDE"

        allowed = client.post(
            "/api/lora/training-pipeline",
            json={
                "datasetId": ds["id"],
                "name": "expert",
                "profileId": standard_id,
                "resolutionBuckets": "768x448x49",
                "allowUnsafeOverride": True,
            },
        )
        assert allowed.status_code == 200, allowed.text
        stored = test_state.lora_training.get_dataset(ds["id"])
        assert stored is not None and stored.auto_pipeline is not None
        assert stored.auto_pipeline.training.config.preset == "standard"
        assert stored.auto_pipeline.training.config.rank == 32

    def test_pipeline_local_rejected_without_wsl(self, client) -> None:
        # Only meaningful where WSL2 is absent; skip on machines that have it
        # (the no-mock policy means we can't fake WSL's absence).
        if _wsl_installed(client):
            pytest.skip("WSL2 is installed on this machine; the local-rejected path can't be exercised without mocking.")
        ds = self._create_dataset(client)
        # `provider: "local"` on a WSL-less machine is rejected up front, so the
        # dataset is never moved to `uploading`.
        r = client.post(
            "/api/lora/training-pipeline",
            json={"datasetId": ds["id"], "name": "run1", "provider": "local"},
        )
        assert r.status_code == 422, r.text
        assert r.json()["code"] == "LOCAL_TRAINER_UNAVAILABLE"
        # Dataset stayed a draft — the impossible run never started.
        assert client.get("/api/lora/datasets").json()["datasets"][0]["status"] == "draft"


class TestProfiles:
    def test_builtins_seeded_on_first_load(self, client) -> None:
        r = client.get("/api/lora/profiles")
        assert r.status_code == 200, r.text
        names = [p["name"] for p in r.json()["profiles"]]
        assert names == ["Standard LoRA", "Low VRAM", "IC-LoRA"]
        assert all(p["builtin"] for p in r.json()["profiles"])
        assert next(p for p in r.json()["profiles"] if p["name"] == "IC-LoRA")[
            "datasetTypes"
        ] == ["ic_lora"]

    def test_create_update_delete(self, client) -> None:
        created = client.post(
            "/api/lora/profiles",
            json={
                "name": "My Profile",
                "description": "My custom settings",
                "datasetTypes": ["standard"],
                "config": {"rank": 64, "steps": 3000},
            },
        )
        assert created.status_code == 200, created.text
        pid = created.json()["id"]
        assert created.json()["config"]["rank"] == 64
        assert created.json()["config"]["steps"] == 3000
        assert created.json()["builtin"] is False
        assert created.json()["description"] == "My custom settings"
        assert created.json()["datasetTypes"] == ["standard"]

        updated = client.patch(
            f"/api/lora/profiles/{pid}",
            json={"name": "Renamed", "config": {"rank": 16}},
        )
        assert updated.status_code == 200, updated.text
        assert updated.json()["name"] == "Renamed"
        assert updated.json()["config"]["rank"] == 16

        deleted = client.delete(f"/api/lora/profiles/{pid}")
        assert deleted.status_code == 204
        remaining = [p["id"] for p in client.get("/api/lora/profiles").json()["profiles"]]
        assert pid not in remaining

    def test_builtins_are_read_only(self, client) -> None:
        builtin_id = client.get("/api/lora/profiles").json()["profiles"][0]["id"]
        assert client.patch(
            f"/api/lora/profiles/{builtin_id}", json={"name": "Changed"}
        ).status_code == 409
        assert client.delete(f"/api/lora/profiles/{builtin_id}").status_code == 409

    def test_update_missing_404(self, client) -> None:
        r = client.patch("/api/lora/profiles/nope", json={"name": "x"})
        assert r.status_code == 404

    def test_delete_missing_404(self, client) -> None:
        r = client.delete("/api/lora/profiles/nope")
        assert r.status_code == 404

    def test_start_with_unknown_profile_404(self, client) -> None:
        r = client.post(
            "/api/lora/training",
            json={"preprocessedId": "x", "name": "run", "profileId": "missing"},
        )
        assert r.status_code == 404


def test_apply_validation_prompts_override() -> None:
    from _routes.lora_training import _apply_validation_prompts_override
    from state.lora_training_state import TrainingConfig

    base = TrainingConfig()
    # None -> no change (the runner auto-seeds from captions later).
    assert _apply_validation_prompts_override(base, None) is base
    # A list replaces the resolved config's prompts verbatim.
    overridden = _apply_validation_prompts_override(base, ["p1", "p2"])
    assert overridden.validation_prompts == ["p1", "p2"]
    # An empty list is honored (no prompt-only samples).
    empty = _apply_validation_prompts_override(base, [])
    assert empty.validation_prompts == []
    # Original is untouched.
    assert base.validation_prompts != []



class TestCaptioning:
    def test_requires_gemini_key(self, client) -> None:
        r = client.post(
            "/api/lora/caption-clip",
            json={"videoPath": "/tmp/a.mp4"},
        )
        assert r.status_code == 400
        assert "Gemini" in r.json()["message"]

    def test_returns_caption(self, client, test_state, fake_services) -> None:
        test_state.state.app_settings.gemini_api_key = "key"
        fake_services.video_captioner.caption_text = "a cat on a sofa"
        r = client.post(
            "/api/lora/caption-clip",
            json={"videoPath": "/tmp/a.mp4", "withAudio": True},
        )
        assert r.status_code == 200, r.text
        assert r.json()["caption"] == "a cat on a sofa"
        assert fake_services.video_captioner.calls[-1]["with_audio"] is True

    def test_maps_captioner_error_status(self, client, test_state, fake_services) -> None:
        from services.video_captioner.video_captioner import VideoCaptionerError

        test_state.state.app_settings.gemini_api_key = "key"
        fake_services.video_captioner.error = VideoCaptionerError(
            "Clip is too large", status_code=413
        )
        r = client.post(
            "/api/lora/caption-clip",
            json={"videoPath": "/tmp/huge.mp4"},
        )
        assert r.status_code == 413
        assert "too large" in r.json()["message"]

    def test_oversized_clip_is_downscaled_to_a_proxy(
        self, client, test_state, fake_services, tmp_path
    ) -> None:
        test_state.state.app_settings.gemini_api_key = "key"
        big = tmp_path / "huge.mp4"
        big.write_bytes(b"\x00" * (13 * 1024 * 1024))  # over the ~12MB budget

        r = client.post(
            "/api/lora/caption-clip",
            json={"videoPath": str(big), "withAudio": False},
        )
        assert r.status_code == 200, r.text
        assert r.json()["caption"]

        # A downscaled proxy (768 long-side from the 1280x720 fake probe, fps
        # capped at 16, audio muted) was rendered and captioned instead.
        render = fake_services.clip_processor.render_calls[-1]
        plan = render["plan"]
        assert (plan.scale.width, plan.scale.height) == (768, 432)
        assert plan.fps == 16.0 and plan.mute is True
        assert str(fake_services.video_captioner.calls[-1]["video_path"]).endswith("proxy.mp4")

    def test_small_clip_skips_proxy(self, client, test_state, fake_services, tmp_path) -> None:
        test_state.state.app_settings.gemini_api_key = "key"
        small = tmp_path / "small.mp4"
        small.write_bytes(b"\x00" * 1024)  # well under budget

        r = client.post(
            "/api/lora/caption-clip",
            json={"videoPath": str(small), "withAudio": False},
        )
        assert r.status_code == 200, r.text
        assert fake_services.clip_processor.render_calls == []
        assert fake_services.video_captioner.calls[-1]["video_path"] == str(small)


class TestProbing:
    def test_returns_probe(self, client, fake_services) -> None:
        from services.clip_processor.clip_processor import ClipProbeResult

        fake_services.clip_processor.result = ClipProbeResult(
            duration_seconds=3.5,
            width=1920,
            height=1080,
            fps=30.0,
            frame_count=105,
            has_audio=True,
            video_codec="h264",
        )
        r = client.post("/api/lora/probe-clip", json={"videoPath": "/tmp/a.mp4"})
        assert r.status_code == 200, r.text
        probe = r.json()["probe"]
        assert probe["width"] == 1920
        assert probe["height"] == 1080
        assert probe["hasAudio"] is True
        assert probe["frameCount"] == 105
        assert fake_services.clip_processor.calls[-1] == "/tmp/a.mp4"

    def test_maps_processor_error_status(self, client, fake_services) -> None:
        from services.clip_processor.clip_processor import ClipProcessorError

        fake_services.clip_processor.error = ClipProcessorError(
            "Clip not found", status_code=400
        )
        r = client.post("/api/lora/probe-clip", json={"videoPath": "/tmp/missing.mp4"})
        assert r.status_code == 400
        assert "not found" in r.json()["message"]

    def test_apply_edits_renders_and_probes(self, client, fake_services) -> None:
        r = client.post(
            "/api/lora/apply-edits",
            json={
                "sourcePath": "/tmp/a.mp4",
                "edits": {"trim": {"startSeconds": 1.0, "endSeconds": 4.0}, "crop": None},
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["derivedPath"]
        assert body["probe"]["width"] == 1280
        call = fake_services.clip_processor.render_calls[-1]
        assert call["trim"] is not None
        assert call["crop"] is None

    def test_apply_edits_rejects_empty(self, client) -> None:
        r = client.post(
            "/api/lora/apply-edits",
            json={"sourcePath": "/tmp/a.mp4", "edits": {"trim": None, "crop": None}},
        )
        assert r.status_code == 400
        assert "No edits" in r.json()["message"]

    def test_apply_edits_carries_speed_fps_mute_reverse(self, client, fake_services) -> None:
        r = client.post(
            "/api/lora/apply-edits",
            json={
                "sourcePath": "/tmp/a.mp4",
                "edits": {
                    "scale": {"width": 768, "height": 768},
                    "fps": 24.0,
                    "speed": 2.0,
                    "mute": True,
                    "reverse": True,
                },
            },
        )
        assert r.status_code == 200, r.text
        plan = fake_services.clip_processor.render_calls[-1]["plan"]
        assert plan.scale is not None and plan.scale.width == 768
        assert plan.fps == 24.0
        assert plan.speed == 2.0
        assert plan.mute is True
        assert plan.reverse is True

    def test_apply_edits_speed_only_is_not_empty(self, client) -> None:
        r = client.post(
            "/api/lora/apply-edits",
            json={"sourcePath": "/tmp/a.mp4", "edits": {"speed": 1.5}},
        )
        assert r.status_code == 200, r.text

    def test_scene_split_returns_segments(self, client, fake_services) -> None:
        from services.clip_processor.clip_processor import SceneSpan

        fake_services.clip_processor.scenes = [
            SceneSpan(start_seconds=0.0, end_seconds=2.0),
            SceneSpan(start_seconds=2.0, end_seconds=5.0),
        ]
        r = client.post("/api/lora/scene-split", json={"sourcePath": "/tmp/long.mp4"})
        assert r.status_code == 200, r.text
        scenes = r.json()["scenes"]
        assert len(scenes) == 2
        assert scenes[0]["startSeconds"] == 0.0
        assert scenes[1]["endSeconds"] == 5.0
        assert all(s["probe"]["width"] == 1280 for s in scenes)

    def test_persists_edits_on_dataset_clip(self, client) -> None:
        r = client.post(
            "/api/lora/datasets",
            json={
                "name": "edited",
                "clips": [
                    {
                        "localPath": "/tmp/derived.mp4",
                        "caption": "a cat",
                        "sourcePath": "/tmp/original.mp4",
                        "edits": {
                            "trim": {"startSeconds": 1.0, "endSeconds": 3.0},
                            "crop": {"x": 0, "y": 0, "width": 640, "height": 640},
                        },
                    }
                ],
            },
        )
        assert r.status_code == 200, r.text
        clip = r.json()["clips"][0]
        assert clip["sourcePath"] == "/tmp/original.mp4"
        assert clip["edits"]["trim"]["startSeconds"] == 1.0
        assert clip["edits"]["crop"]["width"] == 640

    def test_persists_probe_on_dataset_clip(self, client) -> None:
        r = client.post(
            "/api/lora/datasets",
            json={
                "name": "with-probe",
                "clips": [
                    {
                        "localPath": "/tmp/a.mp4",
                        "caption": "a cat",
                        "probe": {
                            "durationSeconds": 4.0,
                            "width": 1280,
                            "height": 720,
                            "fps": 24.0,
                            "frameCount": 96,
                            "hasAudio": False,
                        },
                    }
                ],
            },
        )
        assert r.status_code == 200, r.text
        clip = r.json()["clips"][0]
        assert clip["probe"]["width"] == 1280
        assert clip["origin"] == "imported"


class TestAiPrep:
    def test_edit_frame_uses_default_model(self, client, fake_services) -> None:
        r = client.post(
            "/api/lora/edit-frame",
            json={"sourcePath": "/tmp/a.mp4", "prompt": "remove the logo"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["framePath"].endswith(".png")
        assert fake_services.clip_processor.frame_calls[-1]["video_path"] == "/tmp/a.mp4"
        # Default tier from settings is nano-banana-2.
        assert fake_services.image_editor.calls[-1]["model"] == "nano-banana-2"

    def test_edit_frame_honours_model_override(self, client, fake_services) -> None:
        r = client.post(
            "/api/lora/edit-frame",
            json={"sourcePath": "/tmp/a.mp4", "prompt": "x", "model": "nano-banana-pro"},
        )
        assert r.status_code == 200, r.text
        assert fake_services.image_editor.calls[-1]["model"] == "nano-banana-pro"

    def test_edit_frame_maps_editor_error(self, client, fake_services) -> None:
        from services.image_editor.image_editor import ImageEditorError

        fake_services.image_editor.error = ImageEditorError("bad key", status_code=400)
        r = client.post(
            "/api/lora/edit-frame",
            json={"sourcePath": "/tmp/a.mp4", "prompt": "x"},
        )
        assert r.status_code == 400
        assert "bad key" in r.json()["message"]

    def test_edit_frame_klein_uses_local_pipeline(self, client, test_state, fake_services, make_test_image) -> None:
        from runtime_config.model_download_specs import resolve_model_path

        klein_dir = resolve_model_path(test_state.config.default_models_dir, "flux-2-klein-9b")
        klein_dir.mkdir(parents=True, exist_ok=True)
        (klein_dir / "model.safetensors").write_bytes(b"\x00" * 1024)
        # The fake clip processor returns dummy bytes; swap in a real PNG so
        # the image-edit handler can load the extracted frame as a reference.
        real_png = make_test_image().getvalue()
        fake_services.clip_processor.extract_frame = lambda *, video_path, time_seconds: real_png  # type: ignore[assignment]

        r = client.post(
            "/api/lora/edit-frame",
            json={"sourcePath": "/tmp/a.mp4", "prompt": "remove the logo", "engine": "klein"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["framePath"].endswith(".png")
        # The local Klein pipeline ran an instruction edit with one reference
        # image (the extracted frame); the Fal/Nano Banana editor was not used.
        assert len(fake_services.image_edit_pipeline.generate_with_references_calls) == 1
        assert len(fake_services.image_edit_pipeline.generate_calls) == 0
        call = fake_services.image_edit_pipeline.generate_with_references_calls[0]
        assert call["prompt"] == "remove the logo"
        assert len(call["reference_images"]) == 1
        assert fake_services.image_editor.calls == []

    def test_edit_frame_klein_not_downloaded_returns_409(self, client) -> None:
        r = client.post(
            "/api/lora/edit-frame",
            json={"sourcePath": "/tmp/a.mp4", "prompt": "x", "engine": "klein"},
        )
        assert r.status_code == 409
        assert r.json()["code"] == "KLEIN_NOT_DOWNLOADED"

    def test_animate_frame_returns_clip(self, client, fake_services, tmp_path) -> None:
        image = tmp_path / "frame.png"
        image.write_bytes(b"png-bytes")
        r = client.post(
            "/api/lora/animate-frame",
            json={"imagePath": str(image), "prompt": "make it move"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["derivedPath"].endswith(".mp4")
        assert body["probe"]["width"] == 1280
        assert fake_services.video_restyler.animate_calls[-1]["prompt"] == "make it move"

    def test_restyle_clip_returns_clip(self, client, fake_services, tmp_path) -> None:
        source = tmp_path / "src.mp4"
        source.write_bytes(b"mp4-bytes")
        r = client.post(
            "/api/lora/restyle-clip",
            json={"sourcePath": str(source), "prompt": "claymation"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["derivedPath"].endswith(".mp4")
        assert fake_services.video_restyler.restyle_calls[-1]["prompt"] == "claymation"

    def test_restyle_maps_error(self, client, fake_services, tmp_path) -> None:
        from services.video_restyler.video_restyler import VideoRestylerError

        source = tmp_path / "src.mp4"
        source.write_bytes(b"mp4-bytes")
        fake_services.video_restyler.error = VideoRestylerError("fal down", status_code=502)
        r = client.post(
            "/api/lora/restyle-clip",
            json={"sourcePath": str(source), "prompt": "x"},
        )
        assert r.status_code == 502
        assert "fal down" in r.json()["message"]

    def test_motion_edit_ltx_uses_v2v_with_reference(self, client, fake_services, tmp_path) -> None:
        source = tmp_path / "src.mp4"
        source.write_bytes(b"mp4-bytes")
        frame = tmp_path / "edited.png"
        frame.write_bytes(b"png-bytes")
        r = client.post(
            "/api/lora/motion-edit",
            json={
                "sourcePath": str(source),
                "referenceImagePath": str(frame),
                "prompt": "snow falling",
                "engine": "ltx_v2v",
                "videoStrength": 0.4,
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["derivedPath"].endswith(".mp4")
        call = fake_services.video_restyler.motion_edit_calls[-1]
        assert call["prompt"] == "snow falling"
        assert call["video_strength"] == 0.4

    def test_motion_edit_kling_uses_motion_transfer(self, client, fake_services, tmp_path) -> None:
        source = tmp_path / "src.mp4"
        source.write_bytes(b"mp4-bytes")
        frame = tmp_path / "edited.png"
        frame.write_bytes(b"png-bytes")
        r = client.post(
            "/api/lora/motion-edit",
            json={
                "sourcePath": str(source),
                "referenceImagePath": str(frame),
                "engine": "kling_motion",
                "characterOrientation": "video",
            },
        )
        assert r.status_code == 200, r.text
        call = fake_services.video_restyler.motion_transfer_calls[-1]
        assert call["character_orientation"] == "video"

    def test_motion_edit_kling_o3_trims_long_source(self, test_state, fake_services, tmp_path) -> None:
        # Kling O3 rejects clips > ~10s, so a longer source must be trimmed to
        # the cap before upload (the no-edit path goes straight to the handler).
        from dataclasses import replace

        fake_services.clip_processor.result = replace(
            fake_services.clip_processor.result, duration_seconds=22.0
        )
        source = tmp_path / "src.mp4"
        source.write_bytes(b"mp4-bytes-long")
        test_state.lora_training.motion_edit_clip(
            source_path=str(source),
            reference_image_path=None,
            prompt="xray",
            engine="kling_o3",
            video_strength=0.5,
            character_orientation="video",
            keep_audio=True,
        )
        trims = [c["trim"] for c in fake_services.clip_processor.render_calls if c["trim"] is not None]
        assert any(t.start_seconds == 0.0 and abs(t.end_seconds - 10.0) < 1e-6 for t in trims)
        # The trimmed placeholder bytes (not the full source) were uploaded.
        assert fake_services.video_restyler.kling_v2v_edit_calls[-1]["video_size"] == len(b"fake")

    def test_motion_edit_kling_o3_keeps_short_source(self, test_state, fake_services, tmp_path) -> None:
        # A source already within the cap is uploaded as-is (no trim render).
        source = tmp_path / "short.mp4"
        source.write_bytes(b"short-mp4")
        test_state.lora_training.motion_edit_clip(
            source_path=str(source),
            reference_image_path=None,
            prompt="xray",
            engine="kling_o3",
            video_strength=0.5,
            character_orientation="video",
            keep_audio=True,
        )
        assert fake_services.clip_processor.render_calls == []
        assert fake_services.video_restyler.kling_v2v_edit_calls[-1]["video_size"] == len(b"short-mp4")

    def test_motion_edit_route_forwards_keep_audio(self, client, fake_services, tmp_path) -> None:
        # The /motion-edit route must forward the client's keepAudio to the
        # restyler — regression guard for the route dropping the flag and always
        # defaulting to keep_audio=True.
        source = tmp_path / "short.mp4"
        source.write_bytes(b"short-mp4")
        frame = tmp_path / "edited.png"
        frame.write_bytes(b"png-bytes")
        r = client.post(
            "/api/lora/motion-edit",
            json={
                "sourcePath": str(source),
                "referenceImagePath": str(frame),
                "prompt": "xray",
                "engine": "kling_o3",
                "videoStrength": 0.5,
                "keepAudio": False,
            },
        )
        assert r.status_code == 200, r.text
        assert fake_services.video_restyler.kling_v2v_edit_calls[-1]["keep_audio"] is False

    def test_motion_edit_kling_o3_downscales_wide_source(self, test_state, fake_services, tmp_path) -> None:
        # Kling O3 rejects clips wider than 2160px, so an over-wide source must
        # be downscaled (aspect-preserving) before upload.
        from dataclasses import replace

        fake_services.clip_processor.result = replace(
            fake_services.clip_processor.result, width=4096, height=2160, duration_seconds=5.0
        )
        source = tmp_path / "wide.mp4"
        source.write_bytes(b"mp4-bytes-wide")
        test_state.lora_training.motion_edit_clip(
            source_path=str(source),
            reference_image_path=None,
            prompt="xray",
            engine="kling_o3",
            video_strength=0.5,
            character_orientation="video",
            keep_audio=True,
        )
        scales = [c["scale"] for c in fake_services.clip_processor.render_calls if c["scale"] is not None]
        assert len(scales) == 1
        assert scales[0].width == 2160
        # 2160 * (2160/4096) = 1139.06 -> nearest even = 1140.
        assert scales[0].height == 1140
        # No trim was needed (duration within the cap).
        assert all(c["trim"] is None for c in fake_services.clip_processor.render_calls)
        assert fake_services.video_restyler.kling_v2v_edit_calls[-1]["video_size"] == len(b"fake")

    def test_motion_edit_maps_error(self, client, fake_services, tmp_path) -> None:
        from services.video_restyler.video_restyler import VideoRestylerError

        source = tmp_path / "src.mp4"
        source.write_bytes(b"mp4-bytes")
        frame = tmp_path / "edited.png"
        frame.write_bytes(b"png-bytes")
        fake_services.video_restyler.error = VideoRestylerError("fal down", status_code=502)
        r = client.post(
            "/api/lora/motion-edit",
            json={"sourcePath": str(source), "referenceImagePath": str(frame)},
        )
        assert r.status_code == 502
        assert "fal down" in r.json()["message"]


class TestPexels:
    def test_search_requires_key(self, client) -> None:
        r = client.post("/api/lora/pexels/search", json={"query": "ocean", "media": "video"})
        assert r.status_code == 400
        assert "Pexels API key" in r.json()["message"]

    def test_search_returns_items(self, client, test_state, fake_services) -> None:
        test_state.state.app_settings.pexels_api_key = "pk"
        r = client.post(
            "/api/lora/pexels/search",
            json={"query": "ocean", "media": "video", "page": 1, "perPage": 24},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["items"][0]["kind"] == "video"
        assert body["items"][0]["author"] == "Jane Doe"
        assert body["totalResults"] == 1
        call = fake_services.pexels_client.search_calls[-1]
        assert call["query"] == "ocean" and call["api_key"] == "pk"

    def test_download_video_returns_path_and_probe(self, client, test_state, fake_services) -> None:
        test_state.state.app_settings.pexels_api_key = "pk"
        r = client.post(
            "/api/lora/pexels/download",
            json={"url": "https://videos.pexels.com/file.mp4", "kind": "video", "ext": "mp4"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["localPath"].endswith(".mp4")
        assert body["probe"]["width"] == 1280
        assert fake_services.pexels_client.download_calls[-1]["url"].endswith("file.mp4")

    def test_download_photo_skips_probe(self, client, test_state) -> None:
        test_state.state.app_settings.pexels_api_key = "pk"
        r = client.post(
            "/api/lora/pexels/download",
            json={"url": "https://images.pexels.com/file.jpg", "kind": "photo", "ext": "jpg"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["localPath"].endswith(".jpg")
        assert body["probe"] is None

    def test_search_maps_error(self, client, test_state, fake_services) -> None:
        from services.pexels_client.pexels_client import PexelsError

        test_state.state.app_settings.pexels_api_key = "pk"
        fake_services.pexels_client.error = PexelsError("rate limited", status_code=429)
        r = client.post("/api/lora/pexels/search", json={"query": "x", "media": "photo"})
        assert r.status_code == 429
        assert "rate limited" in r.json()["message"]


class TestConnection:
    def test_ok(self, client) -> None:
        r = client.post("/api/lora/test-connection", json={"provider": "runpod"})
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_failure_maps_to_ok_false(self, client, fake_services) -> None:
        from services.trainer_target.trainer_target import TrainerTargetError

        fake_services.trainer_target.raise_on_test_connection = TrainerTargetError(
            "bad creds"
        )
        r = client.post("/api/lora/test-connection", json={})
        assert r.status_code == 200
        assert r.json()["ok"] is False
        assert "bad creds" in r.json()["message"]


class TestConnectRunpod:
    def test_returns_gpus_without_creating_paid_storage(
        self, client, fake_services
    ) -> None:
        client.post("/api/settings", json={"runpodKeepModelCached": True})
        r = client.post("/api/lora/runpod/connect")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        # GPU picker is populated from the account's discovered GPUs.
        assert body["gpus"] and body["gpus"][0]["memoryGb"] == 80
        # Connect is read-only; paid storage requires an explicit create action.
        assert body["activeVolumeId"] is None
        assert not fake_services.trainer_target.ensure_volume_calls

    def test_stale_volume_id_is_healed(self, client, fake_services) -> None:
        # A deleted volume id is cleared, never silently recreated.
        client.post(
            "/api/settings",
            json={"runpodNetworkVolumeId": "deleted-vol", "runpodKeepModelCached": True},
        )
        r = client.post("/api/lora/runpod/connect")
        assert r.status_code == 200
        assert r.json()["ok"] is True
        assert r.json()["activeVolumeId"] is None
        assert not fake_services.trainer_target.ensure_volume_calls
        # Settings no longer reference the deleted volume.
        s = client.get("/api/settings").json()
        assert s["runpodNetworkVolumeId"] == ""

    def test_skips_volume_when_caching_off(self, client, fake_services) -> None:
        client.post("/api/settings", json={"runpodKeepModelCached": False})
        r = client.post("/api/lora/runpod/connect")
        assert r.status_code == 200
        assert r.json()["ok"] is True
        assert r.json()["activeVolumeId"] is None
        assert not fake_services.trainer_target.ensure_volume_calls

    def test_failure_maps_to_ok_false(self, client, fake_services) -> None:
        from services.trainer_target.trainer_target import TrainerTargetError

        fake_services.trainer_target.raise_on_connect_account = TrainerTargetError(
            "invalid api key"
        )
        r = client.post("/api/lora/runpod/connect")
        assert r.status_code == 200
        assert r.json()["ok"] is False
        assert "invalid api key" in r.json()["message"]

    def test_returns_active_pods(self, client) -> None:
        r = client.post("/api/lora/runpod/connect")
        assert r.status_code == 200
        pods = r.json()["pods"]
        assert pods and pods[0]["id"] == "fake-pod-1"
        assert pods[0]["createdByApp"] is True

    def test_cache_off_leaves_stale_volume_inactive(
        self, client, fake_services
    ) -> None:
        from dataclasses import replace

        from services.trainer_target.trainer_target import NetworkVolume

        target = fake_services.trainer_target
        target.account_info = replace(
            target.account_info,
            volumes=(
                NetworkVolume(
                    id="old-cache",
                    name="ltx-desktop-lora",
                    size_gb=500,
                    datacenter_id="EU-RO-1",
                    created_by_app=True,
                ),
            ),
        )
        client.post(
            "/api/settings",
            json={
                "runpodKeepModelCached": False,
                "runpodNetworkVolumeId": "old-cache",
            },
        )

        body = client.post("/api/lora/runpod/connect").json()

        assert body["activeVolumeId"] is None
        assert body["cacheEnabled"] is False
        assert body["volumes"][0]["active"] is False
        assert client.get("/api/settings").json()["runpodNetworkVolumeId"] == "old-cache"

    def test_existing_app_volume_is_available_for_per_run_auto_association(
        self, client, fake_services
    ) -> None:
        from dataclasses import replace

        from services.trainer_target.trainer_target import NetworkVolume

        target = fake_services.trainer_target
        target.account_info = replace(
            target.account_info,
            volumes=(
                NetworkVolume(
                    id="existing",
                    name="ltx-desktop-lora",
                    size_gb=500,
                    datacenter_id="EU-RO-1",
                    created_by_app=True,
                ),
            ),
        )
        client.post("/api/settings", json={"runpodKeepModelCached": True})

        body = client.post("/api/lora/runpod/connect").json()

        assert body["activeVolumeId"] is None
        assert body["requiresVolumeSelection"] is False
        assert body["volumes"][0]["savedModelReadiness"] == "unknown"
        assert body["volumes"][0]["availableGpuIds"] == []
        assert not target.ensure_volume_calls


class TestRunpodVolumeLifecycle:
    def test_create_select_disable_and_delete(self, client, fake_services) -> None:
        created = client.post(
            "/api/lora/runpod/volumes/create",
            json={"datacenterId": "US-TX-1", "sizeGb": 750},
        )
        assert created.status_code == 200, created.text
        volume_id = created.json()["volume"]["id"]
        assert created.json()["provisioningRequired"] is True
        assert fake_services.trainer_target.ensure_volume_datacenters == ["US-TX-1"]

        disabled = client.post("/api/lora/runpod/cache/disable")
        assert disabled.status_code == 200
        settings = client.get("/api/settings").json()
        assert settings["runpodKeepModelCached"] is False
        assert settings["runpodNetworkVolumeId"] == volume_id

        selected = client.post(
            "/api/lora/runpod/volumes/select", json={"volumeId": volume_id}
        )
        assert selected.status_code == 200
        assert client.get("/api/settings").json()["runpodKeepModelCached"] is True

        client.post("/api/lora/runpod/cache/disable")
        deleted = client.delete(f"/api/lora/runpod/volumes/{volume_id}")
        assert deleted.status_code == 200, deleted.text
        assert fake_services.trainer_target.deleted_volumes == [volume_id]
        settings = client.get("/api/settings").json()
        assert settings["runpodNetworkVolumeId"] == ""

    def test_select_and_delete_refuse_foreign_volume(
        self, client, fake_services
    ) -> None:
        from dataclasses import replace

        from services.trainer_target.trainer_target import NetworkVolume

        foreign = NetworkVolume(
            id="foreign",
            name="my-data",
            size_gb=1000,
            datacenter_id="US-TX-1",
            created_by_app=False,
        )
        fake_services.trainer_target.account_info = replace(
            fake_services.trainer_target.account_info, volumes=(foreign,)
        )
        selected = client.post(
            "/api/lora/runpod/volumes/select", json={"volumeId": "foreign"}
        )
        assert selected.status_code == 409
        deleted = client.delete("/api/lora/runpod/volumes/foreign")
        assert deleted.status_code == 409


class TestTerminateRunpodPod:
    def test_terminates_pod(self, client, fake_services) -> None:
        r = client.post("/api/lora/runpod/pods/fake-pod-1/terminate")
        assert r.status_code == 200
        assert r.json()["ok"] is True
        assert fake_services.trainer_target.released == 1

    def test_failure_maps_to_ok_false(self, client, fake_services) -> None:
        from services.trainer_target.trainer_target import TrainerTargetError

        fake_services.trainer_target.raise_on_release = TrainerTargetError(
            "pod not found"
        )
        r = client.post("/api/lora/runpod/pods/missing/terminate")
        assert r.status_code == 200
        assert r.json()["ok"] is False
        assert "pod not found" in r.json()["message"]


class TestListRunpodPods:
    def test_returns_pods_with_normalized_lifecycle(self, client) -> None:
        r = client.get("/api/lora/runpod/pods")
        assert r.status_code == 200
        pods = r.json()
        assert pods and pods[0]["id"] == "fake-pod-1"
        # The fake seeds a running pod — the compute panel relies on these
        # normalized fields to pick Stop vs Resume, so they must be populated.
        assert pods[0]["desiredStatus"] == "RUNNING"
        assert pods[0]["running"] is True
        assert pods[0]["createdByApp"] is True
        assert pods[0]["uptimeSeconds"] == 3_600
        assert pods[0]["lastStartedAt"] == "2026-01-01T00:00:00+00:00"

    def test_failure_maps_to_502(self, client, fake_services) -> None:
        from services.trainer_target.trainer_target import TrainerTargetError

        fake_services.trainer_target.raise_on_list_pods = TrainerTargetError(
            "invalid api key"
        )
        r = client.get("/api/lora/runpod/pods")
        assert r.status_code == 502
        assert "invalid api key" in r.json()["message"]


class TestStopRunpodPod:
    def test_stops_pod(self, client, fake_services) -> None:
        r = client.post("/api/lora/runpod/pods/fake-pod-1/stop")
        assert r.status_code == 200
        assert r.json() == {"ok": True, "message": "Pod stopped"}
        assert fake_services.trainer_target.stopped_pods == ["fake-pod-1"]
        # Stop flips the pod's normalized lifecycle so a follow-up list shows it
        # as stopped (Resume action) rather than still running.
        pod = next(
            p for p in fake_services.trainer_target.pods if p.id == "fake-pod-1"
        )
        assert pod.running is False
        assert pod.desired_status == "STOPPED"

    def test_failure_maps_to_ok_false(self, client, fake_services) -> None:
        from services.trainer_target.trainer_target import TrainerTargetError

        fake_services.trainer_target.raise_on_stop_pod = TrainerTargetError(
            "stop failed"
        )
        r = client.post("/api/lora/runpod/pods/fake-pod-1/stop")
        assert r.status_code == 200
        assert r.json()["ok"] is False
        assert "stop failed" in r.json()["message"]


class TestResumeRunpodPod:
    def test_resumes_pod(self, client, fake_services) -> None:
        # Stop first so resume has a stopped pod to act on.
        client.post("/api/lora/runpod/pods/fake-pod-1/stop")
        r = client.post("/api/lora/runpod/pods/fake-pod-1/resume")
        assert r.status_code == 200
        assert r.json() == {"ok": True, "message": "Pod resumed"}
        assert fake_services.trainer_target.resumed_pods == ["fake-pod-1"]
        pod = next(
            p for p in fake_services.trainer_target.pods if p.id == "fake-pod-1"
        )
        assert pod.running is True
        assert pod.desired_status == "RUNNING"

    def test_failure_maps_to_ok_false(self, client, fake_services) -> None:
        from services.trainer_target.trainer_target import TrainerTargetError

        fake_services.trainer_target.raise_on_resume_pod = TrainerTargetError(
            "resume failed"
        )
        r = client.post("/api/lora/runpod/pods/fake-pod-1/resume")
        assert r.status_code == 200
        assert r.json()["ok"] is False
        assert "resume failed" in r.json()["message"]


@pytest.mark.parametrize(
    ("action", "error_attribute"),
    [
        ("terminate", "raise_on_release"),
        ("stop", "raise_on_stop_pod"),
        ("resume", "raise_on_resume_pod"),
    ],
)
def test_pod_controls_return_403_for_foreign_pods(
    client, fake_services, action: str, error_attribute: str
) -> None:
    from services.trainer_target.trainer_target import TrainerTargetError

    setattr(
        fake_services.trainer_target,
        error_attribute,
        TrainerTargetError(
            "pod was not created by LTX Desktop",
            code="ownership_violation",
        ),
    )
    response = client.post(f"/api/lora/runpod/pods/foreign/{action}")

    assert response.status_code == 403
    assert response.json()["code"] == "LORA_POD_NOT_OWNED"


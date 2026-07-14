"""Tests for the in-app LoRA inference registry (Gen Space "Apply LoRA")."""

from __future__ import annotations

from pathlib import Path

from api_types import LoraInferenceRegistryResponseApi
from runtime_config.model_download_specs import (
    is_cp_downloaded,
    resolve_model_path,
)
from state.lora_training_state import (
    LoraClip,
    LoraDataset,
    PreprocessedDataset,
    TrainingConfig,
    TrainingJob,
)

_UNION_CP_ID = "ltx-2.3-22b-ic-lora-union-control-ref0.5"


def _write_union_checkpoint(test_state) -> Path:
    path = resolve_model_path(test_state.config.default_models_dir, _UNION_CP_ID)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * 16)
    return path


def _inject_completed_run(
    handler,
    tmp_path: Path,
    *,
    job_id: str,
    dataset_type: str,
    status: str = "completed",
    weights_exist: bool = True,
) -> TrainingJob:
    clip = tmp_path / f"{job_id}-clip.mp4"
    clip.write_bytes(b"\x00\x01")
    weights = tmp_path / f"{job_id}.safetensors"
    if weights_exist:
        weights.write_bytes(b"\x00" * 8)
    ds = LoraDataset(
        id=f"ds-{job_id}",
        name=f"Dataset {job_id}",
        created_at="2026-01-01T00:00:00Z",
        status="uploaded",
        trigger_word="MYTOK",
        remote_dataset_dir=f"/w/datasets/{job_id}",
        type=dataset_type,  # type: ignore[arg-type]
        clips=[LoraClip(id="c1", local_path=str(clip), caption="a cat", duration_seconds=3.0)],
    )
    pre = PreprocessedDataset(
        id=f"pre-{job_id}",
        dataset_id=ds.id,
        created_at="2026-01-01T00:00:00Z",
        status="ready",
        resolution_buckets="768x448x89",
        remote_precomputed_dir=f"/w/.precomputed-{job_id}",
    )
    job = TrainingJob(
        id=job_id,
        preprocessed_id=pre.id,
        name=f"Run {job_id}",
        created_at="2026-01-01T00:00:00Z",
        status=status,  # type: ignore[arg-type]
        config=TrainingConfig(rank=64, trigger_word=ds.trigger_word),
        provider="runpod",
        local_lora_path=str(weights) if status == "completed" else None,
    )
    handler._datasets.datasets.append(ds)
    handler._preprocessed.items.append(pre)
    handler._training.items.append(job)
    return job


def test_official_union_entry_unavailable_when_checkpoint_missing(test_state) -> None:
    entries = test_state.lora_inference_registry.list_entries()
    union = next(e for e in entries if e.kind == "official_union")
    assert union.variant == "union_control"
    assert union.available is False
    assert union.localPath is None
    assert union.conditioningTypes == ["canny", "depth", "pose"]
    assert union.sourceTrainingId is None


def test_official_union_entry_available_when_checkpoint_downloaded(test_state) -> None:
    path = _write_union_checkpoint(test_state)
    assert is_cp_downloaded(test_state.config.default_models_dir, _UNION_CP_ID)
    entries = test_state.lora_inference_registry.list_entries()
    union = next(e for e in entries if e.kind == "official_union")
    assert union.available is True
    assert union.localPath == str(path)


def test_user_trained_standard_lora_appears_as_standard_variant(
    test_state, tmp_path
) -> None:
    _inject_completed_run(test_state.lora_training, tmp_path, job_id="std-1", dataset_type="standard")
    entries = test_state.lora_inference_registry.list_entries()
    user = next(e for e in entries if e.id == "user-std-1")
    assert user.kind == "user_trained"
    assert user.variant == "standard"
    assert user.available is True
    assert user.conditioningTypes == []
    assert user.sourceTrainingId == "std-1"


def test_user_trained_ic_lora_appears_as_video_input_variant(
    test_state, tmp_path
) -> None:
    _inject_completed_run(test_state.lora_training, tmp_path, job_id="ic-1", dataset_type="ic_lora")
    entries = test_state.lora_inference_registry.list_entries()
    user = next(e for e in entries if e.id == "user-ic-1")
    assert user.variant == "video_input_ic_lora"
    assert user.sourceTrainingId == "ic-1"


def test_non_completed_job_or_missing_weights_excluded(test_state, tmp_path) -> None:
    _inject_completed_run(
        test_state.lora_training, tmp_path, job_id="failed-1", dataset_type="standard", status="failed"
    )
    _inject_completed_run(
        test_state.lora_training,
        tmp_path,
        job_id="completed-missing-weights",
        dataset_type="standard",
        weights_exist=False,
    )
    entries = test_state.lora_inference_registry.list_entries()
    ids = {e.id for e in entries}
    assert "user-failed-1" not in ids
    assert "user-completed-missing-weights" not in ids


def test_registry_route_returns_entries(test_state, client, tmp_path) -> None:
    _inject_completed_run(test_state.lora_training, tmp_path, job_id="route-std", dataset_type="standard")
    _write_union_checkpoint(test_state)
    resp = client.get("/api/lora-inference/registry")
    assert resp.status_code == 200
    parsed = LoraInferenceRegistryResponseApi.model_validate(resp.json())
    assert any(e.kind == "official_union" and e.available for e in parsed.entries)
    assert any(e.id == "user-route-std" and e.variant == "standard" for e in parsed.entries)


# ------------------------------------------------------------------
# Library metadata enrichment (createdAt / fileSizeBytes / description)
# ------------------------------------------------------------------


def test_official_union_entry_carries_file_size_when_downloaded(test_state) -> None:
    _write_union_checkpoint(test_state)
    entries = test_state.lora_inference_registry.list_entries()
    union = next(e for e in entries if e.kind == "official_union")
    assert union.fileSizeBytes == 16
    assert union.createdAt is None
    assert union.huggingfaceUrl is None


def test_user_trained_entry_carries_created_at_and_file_size(test_state, tmp_path) -> None:
    _inject_completed_run(test_state.lora_training, tmp_path, job_id="meta-1", dataset_type="standard")
    entries = test_state.lora_inference_registry.list_entries()
    user = next(e for e in entries if e.id == "user-meta-1")
    assert user.createdAt == "2026-01-01T00:00:00Z"
    assert user.fileSizeBytes == 8
    assert user.huggingfaceUrl is None
    assert user.description is None


def test_user_trained_entry_surfaces_job_description(test_state, tmp_path) -> None:
    job = _inject_completed_run(
        test_state.lora_training, tmp_path, job_id="meta-2", dataset_type="standard"
    )
    test_state.lora_training.update_training_meta(job.id, description="My style notes")
    entries = test_state.lora_inference_registry.list_entries()
    user = next(e for e in entries if e.id == "user-meta-2")
    assert user.description == "My style notes"


# ------------------------------------------------------------------
# Trained LoRA management (rename / description / delete via the library)
# ------------------------------------------------------------------


def test_update_trained_renames_and_sets_description(test_state, client, tmp_path) -> None:
    job = _inject_completed_run(
        test_state.lora_training, tmp_path, job_id="upd-1", dataset_type="standard"
    )
    resp = client.patch(
        f"/api/lora-inference/trained/user-{job.id}",
        json={"name": "Renamed Run", "description": "notes"},
    )
    assert resp.status_code == 200, resp.text
    entry = resp.json()["entry"]
    assert entry["name"] == "Renamed Run"
    assert entry["description"] == "notes"
    # The training job itself reflects the rename.
    updated = next(j for j in test_state.lora_training.get_training_state().items if j.id == job.id)
    assert updated.name == "Renamed Run"
    assert updated.description == "notes"


def test_update_trained_blank_name_returns_409(test_state, client, tmp_path) -> None:
    job = _inject_completed_run(
        test_state.lora_training, tmp_path, job_id="upd-2", dataset_type="standard"
    )
    resp = client.patch(f"/api/lora-inference/trained/user-{job.id}", json={"name": "   "})
    assert resp.status_code == 409
    assert resp.json()["code"] == "LORA_INVALID_TRANSITION"


def test_update_trained_unknown_returns_404(test_state, client) -> None:
    resp = client.patch("/api/lora-inference/trained/user-nope", json={"name": "x"})
    assert resp.status_code == 404
    assert resp.json()["code"] == "LORA_NOT_FOUND"


def test_update_trained_rejects_imported_id(test_state, client, tmp_path) -> None:
    resp = client.patch("/api/lora-inference/trained/imported-x", json={"name": "x"})
    assert resp.status_code == 404  # not in registry -> LORA_NOT_FOUND


def test_delete_trained_removes_job(test_state, client, tmp_path) -> None:
    job = _inject_completed_run(
        test_state.lora_training, tmp_path, job_id="del-1", dataset_type="standard"
    )
    resp = client.delete(f"/api/lora-inference/trained/user-{job.id}")
    assert resp.status_code == 204
    ids = {j.id for j in test_state.lora_training.get_training_state().items}
    assert job.id not in ids


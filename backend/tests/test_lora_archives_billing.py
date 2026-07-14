from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from handlers.lora_training_handler import LoraTransitionError
from services.trainer_target.trainer_target import TrainerTargetError
from state.app_settings import AppSettingsPatch
from state.lora_training_state import (
    AutoPipelineSpec,
    LoraClip,
    PendingTraining,
    TargetHandle,
    TrainingConfig,
)


def _dataset(handler, tmp_path: Path, name: str = "dataset"):
    clip = tmp_path / f"{name}.mp4"
    clip.write_bytes(b"video")
    return handler.create_dataset(
        name=name,
        trigger_word="TOK",
        clips=[LoraClip(id=f"{name}-clip", local_path=str(clip), caption="TOK cat")],
    )


def _completed_run(handler, tmp_path: Path):
    dataset = _dataset(handler, tmp_path)
    handler.request_upload(dataset.id)
    handler.mark_dataset_uploaded(
        dataset.id,
        remote_dataset_dir="/workspace/datasets/source",
        handle=TargetHandle(provider="runpod", pod_id="fake-pod-1"),
    )
    preprocessed = handler.create_preprocessing(
        dataset_id=dataset.id,
        resolution_buckets="768x448x49",
        with_audio=False,
        auto_caption=False,
        captioner_type="qwen_omni",
    )
    handler.mark_preprocess_ready(
        preprocessed.id, remote_precomputed_dir="/workspace/.precomputed/source"
    )
    job = handler.start_training(
        preprocessed_id=preprocessed.id,
        name="run",
        config=TrainingConfig(),
        provider="runpod",
    )
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"weights")
    handler.mark_training_completed(job.id, local_lora_path=str(weights))
    return dataset, preprocessed, handler.get_training(job.id)


def test_archive_routes_filter_independently_and_restore(
    client, test_state, tmp_path
) -> None:
    handler = test_state.lora_training
    test_state.lora_training_runner.stop()
    dataset, _, job = _completed_run(handler, tmp_path)
    assert job is not None

    archived_dataset = client.post(f"/api/lora/datasets/{dataset.id}/archive")
    archived_job = client.post(f"/api/lora/training/{job.id}/archive")
    assert archived_dataset.status_code == 200
    assert archived_job.status_code == 200
    assert archived_dataset.json()["archivedAt"]
    assert archived_job.json()["archivedAt"]
    assert client.get("/api/lora/datasets").json()["datasets"] == []
    assert client.get("/api/lora/training").json()["items"] == []
    assert len(
        client.get("/api/lora/datasets?includeArchived=true").json()["datasets"]
    ) == 1
    assert len(
        client.get("/api/lora/training?includeArchived=true").json()["items"]
    ) == 1

    assert handler.get_dataset(dataset.id) is not None
    assert handler.get_training(job.id) is not None
    assert client.post(f"/api/lora/datasets/{dataset.id}/unarchive").status_code == 200
    assert client.post(f"/api/lora/training/{job.id}/unarchive").status_code == 200


def test_archive_guards_active_and_cancel_finalizing(test_state, tmp_path) -> None:
    handler = test_state.lora_training
    dataset = _dataset(handler, tmp_path)
    handler.request_upload(dataset.id)
    with pytest.raises(LoraTransitionError, match="active"):
        handler.archive_dataset(dataset.id)

    # A terminal-looking row still cannot archive while cancellation cleanup is
    # outstanding.
    _, _, job = _completed_run(handler, tmp_path)
    assert job is not None
    for stored in handler._training.items:
        if stored.id == job.id:
            stored.cancel_requested = True
    with pytest.raises(LoraTransitionError, match="active"):
        handler.archive_training(job.id)


def test_archived_completed_run_remains_in_inference_registry(
    test_state, tmp_path
) -> None:
    handler = test_state.lora_training
    _, _, job = _completed_run(handler, tmp_path)
    assert job is not None
    handler.archive_training(job.id)

    entries = test_state.lora_inference_registry.list_entries()
    assert any(entry.sourceTrainingId == job.id for entry in entries)


def test_reconciler_queries_exclude_archived_rows(test_state, tmp_path) -> None:
    handler = test_state.lora_training
    dataset = _dataset(handler, tmp_path)
    handler.request_upload(dataset.id)
    # Simulate a migrated/inconsistent ledger defensively: query filters must
    # never schedule an archived entity even if its old status says active.
    handler._datasets.datasets[0].archived_at = datetime.now(timezone.utc).isoformat()
    assert handler.list_datasets_to_upload() == []


def test_one_click_cost_attribution_covers_pipeline_and_closes(
    test_state, tmp_path
) -> None:
    handler = test_state.lora_training
    dataset = _dataset(handler, tmp_path)
    spec = AutoPipelineSpec(
        resolution_buckets="768x448x49",
        with_audio=False,
        auto_caption=False,
        captioner_type="qwen_omni",
        training=PendingTraining(config=TrainingConfig(steps=10), name="costed"),
    )
    handler.start_training_pipeline(dataset_id=dataset.id, spec=spec)
    for _ in range(6):
        test_state.lora_training_runner.reconcile_once()

    job = handler.get_training_state().items[0]
    assert job.status == "completed"
    assert job.workload_billing_started_at is not None
    assert job.workload_billing_ended_at is not None
    assert job.captured_hourly_rate == pytest.approx(1.89)
    assert job.attributed_seconds is not None and job.attributed_seconds >= 0
    assert job.attributed_cost is not None and job.attributed_cost >= 0
    assert job.pod_preparation_started_at == job.workload_billing_started_at
    assert job.training_setup_started_at is not None
    assert job.training_setup_ended_at is not None


def test_manual_training_starts_its_own_billing_interval(
    test_state, tmp_path
) -> None:
    handler = test_state.lora_training
    dataset = _dataset(handler, tmp_path)
    handler.request_upload(dataset.id)
    handler.mark_dataset_uploaded(
        dataset.id,
        remote_dataset_dir="/workspace/datasets/manual",
        handle=TargetHandle(provider="runpod", pod_id="fake-pod-1"),
    )
    preprocessed = handler.create_preprocessing(
        dataset_id=dataset.id,
        resolution_buckets="768x448x49",
        with_audio=False,
        auto_caption=False,
        captioner_type="qwen_omni",
    )
    handler.mark_preprocess_ready(
        preprocessed.id, remote_precomputed_dir="/workspace/.precomputed/manual"
    )
    job = handler.start_training(
        preprocessed_id=preprocessed.id,
        name="manual",
        config=TrainingConfig(steps=10),
        provider="runpod",
    )
    assert job.workload_billing_started_at is None

    test_state.lora_training_runner.reconcile_once()
    started = handler.get_training(job.id)
    assert started is not None and started.workload_billing_started_at is not None
    assert started.workload_billing_started_at != dataset.upload_started_at
    test_state.lora_training_runner.reconcile_once()
    completed = handler.get_training(job.id)
    assert completed is not None and completed.status == "completed"
    assert completed.workload_billing_ended_at is not None
    assert completed.attributed_cost is not None


@pytest.mark.parametrize("terminal", ["failed", "cancelled"])
def test_failure_and_cancellation_close_billing(
    test_state, tmp_path, terminal: str
) -> None:
    handler = test_state.lora_training
    _, preprocessed, _ = _completed_run(handler, tmp_path)
    job = handler.start_training(
        preprocessed_id=preprocessed.id,
        name=f"{terminal}-run",
        config=TrainingConfig(),
        provider="runpod",
    )
    handler.begin_training_billing(
        job.id,
        started_at=(datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat(),
        hourly_rate=1.8,
    )
    if terminal == "failed":
        handler.fail_training(job.id, "boom")
    else:
        handler.mark_training_cancelled(job.id)
    closed = handler.get_training(job.id)
    assert closed is not None
    assert closed.workload_billing_ended_at is not None
    assert closed.attributed_seconds is not None and closed.attributed_seconds >= 29
    assert closed.attributed_cost is not None and closed.attributed_cost > 0


def test_keepalive_expiry_and_release_failure_are_durable(
    client, test_state, tmp_path, fake_services
) -> None:
    handler = test_state.lora_training
    test_state.lora_training_runner.stop()
    dataset, _, _ = _completed_run(handler, tmp_path)
    for stored in handler._datasets.datasets:
        if stored.id == dataset.id:
            stored.final_activity_at = "2000-01-01T00:00:00+00:00"
            stored.last_active_at = stored.final_activity_at
    test_state.settings.update_settings(
        AppSettingsPatch(runpod_idle_stop_minutes=10)
    )

    response = client.post(
        "/api/lora/runpod/pods/fake-pod-1/keep-alive", json={"minutes": 30}
    )
    assert response.status_code == 200, response.text
    test_state.lora_training_runner.reconcile_once()
    assert fake_services.trainer_target.released == 0

    for stored in handler._datasets.datasets:
        if stored.id == dataset.id:
            stored.keep_alive_until = (
                datetime.now(timezone.utc) - timedelta(minutes=1)
            ).isoformat()
            stored.final_activity_at = "2000-01-01T00:00:00+00:00"
            stored.last_active_at = stored.final_activity_at
    fake_services.trainer_target.raise_on_release = TrainerTargetError(
        "release unavailable", retryable=True
    )
    test_state.lora_training_runner.reconcile_once()
    failed = handler.get_dataset(dataset.id)
    assert failed is not None
    assert failed.release_status == "failed"
    assert failed.release_error == "release unavailable"
    assert failed.release_attempted_at is not None

    fake_services.trainer_target.raise_on_release = None
    test_state.lora_training_runner.reconcile_once()
    released = handler.get_dataset(dataset.id)
    assert released is not None
    assert released.release_status == "released"
    assert released.target is not None and released.target.pod_id is None


def test_idle_stop_protects_queued_compatible_workspace(
    test_state, tmp_path, fake_services
) -> None:
    handler = test_state.lora_training
    first, _, _ = _completed_run(handler, tmp_path)
    second = _dataset(handler, tmp_path, "queued")
    for dataset in handler._datasets.datasets:
        dataset.workspace_policy = "primary_cache"
        dataset.cache_volume_id = "vol-shared"
        if dataset.id == first.id:
            dataset.final_activity_at = "2000-01-01T00:00:00+00:00"
            dataset.last_active_at = dataset.final_activity_at
    handler.request_upload(second.id)
    test_state.settings.update_settings(
        AppSettingsPatch(
            runpod_idle_stop_minutes=10,
            runpod_keep_model_cached=True,
            runpod_network_volume_id="vol-shared",
        )
    )

    settings = test_state.settings.get_settings_snapshot()
    test_state.lora_training_runner._reconcile_idle_stops(settings, [], [])
    # The queued upload reserves the compatible cache workspace; terminal work
    # on the first dataset must not stop that shared pod out from under it.
    assert fake_services.trainer_target.released == 0

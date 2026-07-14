from __future__ import annotations

from pathlib import Path

import pytest

from handlers.lora_cost_estimator import EstimateInputs, estimate_cost
from handlers.lora_training_handler import LoraTransitionError
from services.trainer_target.runpod_trainer_target import (
    RunPodTrainerTarget,
    _stock_is_available,
)
from services.trainer_target.trainer_target import (
    GpuOffer,
    TrainerCredentials,
    TrainerTargetError,
)
from state.lora_training_state import (
    AutoPipelineSpec,
    LoraClip,
    PendingTraining,
    RunpodSelection,
    TargetHandle,
    TrainingConfig,
)


class RegionalTarget(RunPodTrainerTarget):
    def _storage_datacenters(self, credentials: TrainerCredentials) -> list[str]:
        del credentials
        return ["EU-1", "US-1"]

    def _discover_gpus_graphql(
        self, credentials: TrainerCredentials, datacenter: str
    ) -> list[GpuOffer] | None:
        del credentials
        if datacenter == "EU-1":
            return [GpuOffer("A100", "A100", 80, 2.0, False)]
        if datacenter == "US-1":
            return [GpuOffer("A100", "A100", 80, 1.5, True)]
        if datacenter == "":
            return [
                GpuOffer("A100", "A100", 80, 1.5, True),
                GpuOffer("H200", "H200", 141, 3.0, True),
            ]
        return None


def _credentials() -> TrainerCredentials:
    return TrainerCredentials(
        provider="runpod",
        workspace_dir="/workspace",
        model_path="/workspace/model",
        text_encoder_path="/workspace/encoder",
        runpod_api_key="key",
    )


@pytest.mark.parametrize(
    ("status", "counts", "expected"),
    [
        ("High", [1, 2, 4], True),
        ("Medium", [1], True),
        ("Low", [1, 2], True),
        ("None", [], False),
        ("None", [1], False),
        (None, None, False),
        ("High", [2, 4], False),
    ],
)
def test_runpod_stock_status_is_parsed_explicitly(
    status: object,
    counts: object,
    expected: bool,
) -> None:
    assert _stock_is_available(status, counts) is expected


def test_regional_inventory_reports_elsewhere_stock() -> None:
    offers = RegionalTarget()._discover_gpus_region_aware(
        _credentials(), object(), "EU-1"
    )
    assert offers[0].active_region_available is False
    assert offers[0].available_elsewhere is True
    assert offers[0].best_available_region == "US-1"
    assert offers[0].recommended is True


def test_global_inventory_is_not_limited_to_a_cache_region() -> None:
    offers = RegionalTarget()._discover_gpus_region_aware(
        _credentials(), object(), ""
    )
    assert {offer.id for offer in offers if offer.available} == {"A100", "H200"}
    assert next(offer for offer in offers if offer.id == "A100").best_available_region == "US-1"


def test_saved_model_readiness_tracks_fingerprint(test_state) -> None:
    handler = test_state.lora_training
    assert handler.saved_model_readiness(
        volume_id="vol-1", fingerprint="v1", estimated_download_bytes=123
    ) == ("unknown", 123)
    handler.mark_saved_model_ready(
        volume_id="vol-1", fingerprint="v1", estimated_download_bytes=123
    )
    assert handler.saved_model_readiness(
        volume_id="vol-1", fingerprint="v1", estimated_download_bytes=123
    ) == ("ready", 0)
    assert handler.saved_model_readiness(
        volume_id="vol-1", fingerprint="v2", estimated_download_bytes=456
    ) == ("missing", 456)


def test_capacity_reselection_preserves_local_dataset(
    test_state, tmp_path: Path, fake_services
) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"video")
    selection = RunpodSelection(
        gpu_type="A100",
        gpu_vram_gb=80,
        datacenter="EU-1",
        workspace_policy="ephemeral_any_region",
    )
    dataset = test_state.lora_training.create_dataset(
        name="dataset",
        trigger_word=None,
        clips=[LoraClip(id="clip", local_path=str(clip))],
    )
    spec = AutoPipelineSpec(
        resolution_buckets="768x448x49",
        training=PendingTraining(
            config=TrainingConfig(steps=10),
            name="run",
            gpu_type="A100",
            gpu_vram_gb=80,
            runpod_selection=selection,
        ),
        runpod_selection=selection,
    )
    test_state.lora_training.start_training_pipeline(
        dataset_id=dataset.id,
        spec=spec,
        workspace_policy=selection.workspace_policy,
    )
    fake_services.trainer_target.raise_on_ensure_workspace = TrainerTargetError(
        "A100 sold out", code="capacity_unavailable"
    )
    test_state.lora_training_runner.reconcile_once()
    waiting = test_state.lora_training.get_dataset(dataset.id)
    assert waiting is not None
    assert waiting.status == "gpu_selection_required"
    replacement = selection.model_copy(
        update={"gpu_type": "H100", "datacenter": "US-1"}
    )
    resumed = test_state.lora_training.reselect_dataset(dataset.id, replacement)
    assert resumed.status == "uploading"
    assert resumed.runpod_selection == replacement
    assert clip.exists()


def test_cost_estimate_has_tiered_storage_and_unknown_download() -> None:
    estimate = estimate_cost(
        EstimateInputs(
            steps=1000,
            clip_count=8,
            total_clip_seconds=40,
            preprocessed=False,
            resolution_buckets="768x448x49",
            mode="standard",
            with_audio=False,
            gpu_price_per_hr=2.0,
            storage_readiness="unknown",
            estimated_model_download_bytes=None,
            idle_timeout_minutes=10,
            storage_size_gb=1500,
        )
    )
    assert estimate.confidence == "low"
    assert estimate.low_gpu_cost < estimate.high_gpu_cost
    assert estimate.download_bytes is None
    assert estimate.storage_monthly_cost == 95.0


def test_preprocessed_artifacts_only_switch_gpu_on_same_volume_region(
    test_state, tmp_path: Path
) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"video")
    original = RunpodSelection(
        gpu_type="A100",
        gpu_vram_gb=80,
        datacenter="EU-1",
        workspace_policy="primary_cache",
        volume_id="vol-eu",
    )
    dataset = test_state.lora_training.create_dataset(
        name="dataset",
        trigger_word=None,
        clips=[LoraClip(id="clip", local_path=str(clip))],
    )
    test_state.lora_training.start_training_pipeline(
        dataset_id=dataset.id,
        spec=AutoPipelineSpec(
            resolution_buckets="768x448x49",
            training=PendingTraining(
                config=TrainingConfig(steps=10),
                name="pipeline",
                runpod_selection=original,
            ),
            runpod_selection=original,
        ),
        workspace_policy="primary_cache",
        cache_volume_id="vol-eu",
    )
    test_state.lora_training.mark_dataset_uploaded(
        dataset.id,
        remote_dataset_dir="/workspace/dataset",
        handle=TargetHandle(provider="runpod", pod_id="pod-1"),
    )
    preprocessed = test_state.lora_training.create_preprocessing(
        dataset_id=dataset.id,
        resolution_buckets="768x448x49",
        with_audio=False,
        auto_caption=False,
        captioner_type="qwen_omni",
    )
    test_state.lora_training.mark_preprocess_ready(
        preprocessed.id, remote_precomputed_dir="/workspace/precomputed"
    )
    cross_region = original.model_copy(
        update={"gpu_type": "H100", "datacenter": "US-1"}
    )
    with pytest.raises(LoraTransitionError, match="original cache volume and region"):
        test_state.lora_training.start_training(
            preprocessed_id=preprocessed.id,
            name="cross-region",
            config=TrainingConfig(steps=10),
            provider="runpod",
            runpod_selection=cross_region,
        )
    same_volume = original.model_copy(update={"gpu_type": "H100"})
    job = test_state.lora_training.start_training(
        preprocessed_id=preprocessed.id,
        name="same-volume",
        config=TrainingConfig(steps=10),
        provider="runpod",
        runpod_selection=same_volume,
    )
    assert job.runpod_selection == same_volume


def test_estimator_api_returns_separate_storage_cost(client) -> None:
    response = client.post(
        "/api/lora/training/estimate",
        json={
            "gpuPricePerHr": 2.0,
            "storageReadiness": "unknown",
            "storageSizeGb": 250,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["lowSeconds"] < body["highSeconds"]
    assert body["downloadBytes"] is None
    assert body["storageMonthlyCost"] == 17.5

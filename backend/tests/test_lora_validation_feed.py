"""Tests for the training-results feed + GPU-status state plumbing.

Covers the handler mutation methods (feed dedup + bounded retention, sample
ref registration, GPU status) and the runner's end-to-end detection +
download of validation samples via the FakeTrainerTarget.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from services.trainer_target.trainer_target import (
    RemoteCommandStatus,
    ValidationArtifact,
)
from api_types import DEFAULT_VALIDATION_PROMPT
from state.lora_training_state import (
    VALIDATION_FEED_MAX_ITEMS,
    GpuStatus,
    LoraDataset,
    TrainingConfig,
    ValidationFeedItem,
    ValidationSampleRef,
    CheckpointArtifact,
)


def _make_clip_file(tmp_path: Path, name: str = "clip0.mp4") -> str:
    path = tmp_path / name
    path.write_bytes(b"fake-video-bytes")
    return str(path)


def _dataset_with(captions: list[str], *, trigger: str | None = "TOK") -> LoraDataset:
    from state.lora_training_state import LoraClip

    return LoraDataset(
        id="ds",
        name="ds",
        created_at="2024-01-01T00:00:00Z",
        status="draft",
        type="standard",
        trigger_word=trigger,
        clips=[
            LoraClip(id=f"c{i}", local_path=f"/tmp/c{i}.mp4", caption=cap)
            for i, cap in enumerate(captions)
        ],
    )


def _uploaded_dataset(test_state, tmp_path):
    from state.lora_training_state import LoraClip

    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    # Helpers below advance the runner synchronously. Stop the lifespan worker
    # so it cannot delete/rebuild the same staging directory concurrently.
    runner.stop()
    clip_path = _make_clip_file(tmp_path)
    dataset = handler.create_dataset(
        name="ds",
        trigger_word="TOK",
        clips=[LoraClip(id="c0", local_path=clip_path, caption="a cat")],
    )
    handler.request_upload(dataset.id)
    runner.reconcile_once()
    return handler.get_dataset(dataset.id)


def _running_job(test_state, tmp_path, fake_services):
    """Drive upload -> preprocess -> train -> running, kept running via the fake."""
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target
    dataset = _uploaded_dataset(test_state, tmp_path)
    pre = handler.create_preprocessing(
        dataset_id=dataset.id,
        resolution_buckets="768x448x49",
        with_audio=False,
        auto_caption=False,
        captioner_type="qwen_omni",
    )
    # The background reconciler may claim one transition between these calls;
    # drive until ready rather than assuming exactly two ticks.
    for _ in range(5):
        runner.reconcile_once()
        if handler.get_preprocessed(pre.id).status == "ready":
            break
    job = handler.start_training(
        preprocessed_id=pre.id,
        name="run1",
        config=TrainingConfig(steps=500, validation_interval=50),
        provider="runpod",
    )
    # Keep the train command "running" on every poll so the reconciler doesn't
    # complete the job before we exercise the feed.
    target.command_results["train.py"] = RemoteCommandStatus(state="running")
    runner.reconcile_once()  # pending -> running (sets refs + remote_output_dir)
    return handler.get_training(job.id)


# ---------------------------------------------------------------------
# Handler: feed dedup + bounded retention
# ---------------------------------------------------------------------


def test_set_validation_sample_refs_stored(test_state, tmp_path, fake_services) -> None:
    job = _running_job(test_state, tmp_path, fake_services)
    handler = test_state.lora_training
    # _start_training auto-seeded the validation prompt from the dataset's
    # caption ("a cat") instead of the generic placeholder default.
    assert job.validation_sample_refs == [
        ValidationSampleRef(prompt="a cat"),
    ]
    # Direct mutation replaces the list.
    handler.set_validation_sample_refs(
        job.id,
        [ValidationSampleRef(prompt="p1"), ValidationSampleRef(prompt="p2", source="holdout")],
    )
    refreshed = handler.get_training(job.id)
    assert [r.prompt for r in refreshed.validation_sample_refs] == ["p1", "p2"]
    assert refreshed.validation_sample_refs[1].source == "holdout"


def test_append_feed_dedupes_by_step_index_and_extension(
    test_state, tmp_path, fake_services
) -> None:
    job = _running_job(test_state, tmp_path, fake_services)
    handler = test_state.lora_training
    handler.append_validation_feed_items(
        job.id,
        [
            ValidationFeedItem(
                step=50, sample_index=1, local_path="/a.mp4", prompt="p"
            ),
        ],
    )
    # Re-appending the same (step, sample_index, extension) is a no-op even with a
    # different local_path — a reconciler re-poll must not double-add.
    handler.append_validation_feed_items(
        job.id,
        [
            ValidationFeedItem(
                step=50, sample_index=1, local_path="/a-dup.mp4", prompt="p"
            ),
        ],
    )
    refreshed = handler.get_training(job.id)
    assert len(refreshed.validation_feed) == 1
    assert refreshed.validation_feed[0].local_path == "/a.mp4"

    # Audio and video can both be emitted for one validation sample.
    handler.append_validation_feed_items(
        job.id,
        [
            ValidationFeedItem(
                step=50,
                sample_index=1,
                extension="wav",
                local_path="/a.wav",
                prompt="p",
            ),
        ],
    )
    refreshed = handler.get_training(job.id)
    assert {(item.extension, item.local_path) for item in refreshed.validation_feed} == {
        ("mp4", "/a.mp4"),
        ("wav", "/a.wav"),
    }


def test_append_feed_caps_to_max_items_keeping_newest(
    test_state, tmp_path, fake_services
) -> None:
    job = _running_job(test_state, tmp_path, fake_services)
    handler = test_state.lora_training
    # Push MAX + 5 items in ascending step order; the oldest should be trimmed.
    items = [
        ValidationFeedItem(
            step=s, sample_index=1, local_path=f"/{s}.mp4", prompt="p"
        )
        for s in range(1, VALIDATION_FEED_MAX_ITEMS + 6)
    ]
    handler.append_validation_feed_items(job.id, items)
    refreshed = handler.get_training(job.id)
    assert len(refreshed.validation_feed) == VALIDATION_FEED_MAX_ITEMS
    # Newest retained: the highest step is the last appended step.
    assert refreshed.validation_feed[-1].step == VALIDATION_FEED_MAX_ITEMS + 5
    assert refreshed.validation_feed[0].step == 6  # first 5 trimmed


def test_set_training_gpu_status_stored(test_state, tmp_path, fake_services) -> None:
    job = _running_job(test_state, tmp_path, fake_services)
    handler = test_state.lora_training
    status = GpuStatus(
        name="NVIDIA RTX 5090",
        vram_total_mb=32510,
        vram_used_mb=12345,
        gpu_util_pct=87,
        mem_util_pct=38,
        temp_c=65,
        updated_at="2026-07-01T10:00:00+00:00",
    )
    handler.set_training_gpu_status(job.id, status)
    refreshed = handler.get_training(job.id)
    assert refreshed.gpu_status == status


# ---------------------------------------------------------------------
# Runner: end-to-end validation-feed detection + download
# ---------------------------------------------------------------------


def test_runner_downloads_new_validation_samples_into_feed(
    test_state, tmp_path, fake_services
) -> None:
    job = _running_job(test_state, tmp_path, fake_services)
    handler = test_state.lora_training
    target = fake_services.trainer_target
    remote_samples = job.remote_output_dir.rstrip("/") + "/samples"

    target.validation_artifacts = [
        ValidationArtifact(50, 1, f"{remote_samples}/step_000050_1.mp4", "mp4"),
    ]
    test_state.lora_training_runner.reconcile_once()  # polls -> downloads step 50

    refreshed = handler.get_training(job.id)
    assert len(refreshed.validation_feed) == 1
    item = refreshed.validation_feed[0]
    assert item.step == 50
    assert item.sample_index == 1
    assert item.source == "prompt"
    assert item.prompt == "a cat"  # auto-seeded from the dataset caption
    assert item.local_path.endswith("step_000050_1.mp4")
    # The artifact was actually downloaded through the target seam.
    assert any(remote == f"{remote_samples}/step_000050_1.mp4"
               for remote, _ in target.downloaded_files)
    # The local file exists (the fake materializes downloads).
    assert Path(item.local_path).exists()


def test_runner_only_downloads_samples_newer_than_feed(
    test_state, tmp_path, fake_services
) -> None:
    job = _running_job(test_state, tmp_path, fake_services)
    handler = test_state.lora_training
    target = fake_services.trainer_target
    remote_samples = job.remote_output_dir.rstrip("/") + "/samples"

    # First tick: step 50 lands.
    target.validation_artifacts = [
        ValidationArtifact(50, 1, f"{remote_samples}/step_000050_1.mp4", "mp4"),
    ]
    test_state.lora_training_runner.reconcile_once()
    downloads_after_50 = len(target.downloaded_files)

    # Second tick: step 50 already in the feed (since_step=50), only step 100
    # is new — the runner must NOT re-download step 50.
    target.validation_artifacts = [
        ValidationArtifact(50, 1, f"{remote_samples}/step_000050_1.mp4", "mp4"),
        ValidationArtifact(100, 1, f"{remote_samples}/step_000100_1.mp4", "mp4"),
    ]
    test_state.lora_training_runner.reconcile_once()

    refreshed = handler.get_training(job.id)
    steps = sorted(i.step for i in refreshed.validation_feed)
    assert steps == [50, 100]
    # Exactly one new download (step 100), not a re-download of step 50.
    assert len(target.downloaded_files) == downloads_after_50 + 1
    assert target.downloaded_files[-1][0] == f"{remote_samples}/step_000100_1.mp4"


# ---------------------------------------------------------------------
# Runner + handler: live checkpoint download / retention
# ---------------------------------------------------------------------


def test_runner_downloads_new_checkpoints(test_state, tmp_path, fake_services) -> None:
    job = _running_job(test_state, tmp_path, fake_services)
    handler = test_state.lora_training
    target = fake_services.trainer_target
    remote_ckpts = job.remote_output_dir.rstrip("/") + "/checkpoints"

    target.checkpoints_steps = [250]
    test_state.lora_training_runner.reconcile_once()  # polls -> downloads step 250

    refreshed = handler.get_training(job.id)
    assert [c.step for c in refreshed.checkpoints] == [250]
    ckpt = refreshed.checkpoints[0]
    assert ckpt.local_path.endswith("lora_weights_step_00250.safetensors")
    # Downloaded through the target seam from the run's checkpoints dir.
    assert any(
        remote == f"{remote_ckpts}/lora_weights_step_00250.safetensors"
        for remote, _ in target.downloaded_files
    )
    # The local file exists (the fake materializes downloads) -> reveal-able.
    assert Path(ckpt.local_path).exists()


def test_runner_only_downloads_new_checkpoints(test_state, tmp_path, fake_services) -> None:
    job = _running_job(test_state, tmp_path, fake_services)
    handler = test_state.lora_training
    target = fake_services.trainer_target

    target.checkpoints_steps = [250]
    test_state.lora_training_runner.reconcile_once()
    downloads_after_250 = len(target.downloaded_files)

    # Second tick: step 250 already downloaded; only 500 is new.
    target.checkpoints_steps = [250, 500]
    test_state.lora_training_runner.reconcile_once()

    refreshed = handler.get_training(job.id)
    assert sorted(c.step for c in refreshed.checkpoints) == [250, 500]
    # Exactly one new download (step 500), not a re-download of step 250.
    assert len(target.downloaded_files) == downloads_after_250 + 1
    assert target.downloaded_files[-1][0].endswith("lora_weights_step_00500.safetensors")


def test_append_checkpoint_artifacts_dedupes_and_trims(
    test_state, tmp_path, fake_services
) -> None:
    job = _running_job(test_state, tmp_path, fake_services)
    handler = test_state.lora_training

    def _ckpt(step: int) -> CheckpointArtifact:
        # Materialize a local file so pruning can assert it gets deleted.
        p = tmp_path / f"lora_weights_step_{step:05d}.safetensors"
        p.write_bytes(b"fake")
        return CheckpointArtifact(
            step=step,
            remote_path=f"/out/checkpoints/lora_weights_step_{step:05d}.safetensors",
            local_path=str(p),
        )

    # Default config keeps the last 3 checkpoints; push 4 ascending steps.
    handler.append_checkpoint_artifacts(
        job.id, [_ckpt(250), _ckpt(500), _ckpt(750), _ckpt(1000)]
    )
    refreshed = handler.get_training(job.id)
    assert [c.step for c in refreshed.checkpoints] == [500, 750, 1000]
    # The pruned (oldest) local file is deleted; kept ones remain.
    pruned_path = tmp_path / "lora_weights_step_00250.safetensors"
    assert not pruned_path.exists()
    assert (tmp_path / "lora_weights_step_01000.safetensors").exists()

    # Re-appending an existing step is a no-op (dedupe).
    handler.append_checkpoint_artifacts(job.id, [_ckpt(1000)])
    refreshed = handler.get_training(job.id)
    assert [c.step for c in refreshed.checkpoints] == [500, 750, 1000]


def test_training_state_response_includes_checkpoints(
    client, test_state, tmp_path, fake_services
) -> None:
    job = _running_job(test_state, tmp_path, fake_services)
    target = fake_services.trainer_target
    target.checkpoints_steps = [250]
    test_state.lora_training_runner.reconcile_once()

    r = client.get("/api/lora/training")
    assert r.status_code == 200
    api_job = next(i for i in r.json()["items"] if i["id"] == job.id)
    assert len(api_job["checkpoints"]) == 1
    assert api_job["checkpoints"][0]["step"] == 250
    assert api_job["checkpoints"][0]["localPath"].endswith(
        "lora_weights_step_00250.safetensors"
    )


def test_runner_polls_gpu_status_while_running(
    test_state, tmp_path, fake_services
) -> None:
    job = _running_job(test_state, tmp_path, fake_services)
    handler = test_state.lora_training
    target = fake_services.trainer_target

    test_state.lora_training_runner.reconcile_once()  # polls GPU status

    refreshed = handler.get_training(job.id)
    assert refreshed.gpu_status is not None
    assert refreshed.gpu_status.name == target.gpu_telemetry.name
    assert refreshed.gpu_status.vram_total_mb == target.gpu_telemetry.vram_total_mb
    assert refreshed.gpu_status.vram_used_mb == target.gpu_telemetry.vram_used_mb
    assert refreshed.gpu_status.gpu_util_pct == target.gpu_telemetry.gpu_util_pct
    assert refreshed.gpu_status.temp_c == target.gpu_telemetry.temp_c
    assert target.query_gpu_calls >= 1


def test_runner_gpu_query_error_keeps_last_status(
    test_state, tmp_path, fake_services
) -> None:
    job = _running_job(test_state, tmp_path, fake_services)
    handler = test_state.lora_training
    target = fake_services.trainer_target
    runner = test_state.lora_training_runner

    runner.reconcile_once()  # establishes a status
    refreshed = handler.get_training(job.id)
    assert refreshed.gpu_status is not None
    first_updated_at = refreshed.gpu_status.updated_at

    # Force a query failure on the next poll; the existing status must persist
    # (best-effort: a transport blip doesn't wipe the panel).
    from services.trainer_target.trainer_target import TrainerTargetError

    target.raise_on_query_gpu = TrainerTargetError("boom", retryable=True)
    runner.reconcile_once()
    refreshed = handler.get_training(job.id)
    assert refreshed.gpu_status is not None
    assert refreshed.gpu_status.updated_at == first_updated_at


def test_validation_media_route_serves_downloaded_sample(
    client, test_state, tmp_path, fake_services
) -> None:
    job = _running_job(test_state, tmp_path, fake_services)
    handler = test_state.lora_training
    target = fake_services.trainer_target
    remote_samples = job.remote_output_dir.rstrip("/") + "/samples"
    target.validation_artifacts = [
        ValidationArtifact(50, 1, f"{remote_samples}/step_000050_1.mp4", "mp4"),
    ]
    test_state.lora_training_runner.reconcile_once()
    refreshed = handler.get_training(job.id)
    assert len(refreshed.validation_feed) == 1

    # The secure route serves the downloaded file by (step, sampleIndex) — no
    # filesystem path is exposed to the client.
    r = client.get(
        f"/api/lora/training/{job.id}/validation-media",
        params={"step": 50, "sampleIndex": 1},
    )
    assert r.status_code == 200
    assert r.content  # the fake materializes a non-empty file
    assert "video/mp4" in r.headers.get("content-type", "")

    # A non-existent (step, sampleIndex) for a real job -> 404 (no traversal).
    r_missing = client.get(
        f"/api/lora/training/{job.id}/validation-media",
        params={"step": 999, "sampleIndex": 1},
    )
    assert r_missing.status_code == 404

    # A non-existent job -> 404.
    r_nojob = client.get(
        "/api/lora/training/no-such-job/validation-media",
        params={"step": 50, "sampleIndex": 1},
    )
    assert r_nojob.status_code == 404


def test_training_state_response_includes_feed_and_gpu_status(
    client, test_state, tmp_path, fake_services
) -> None:
    job = _running_job(test_state, tmp_path, fake_services)
    target = fake_services.trainer_target
    remote_samples = job.remote_output_dir.rstrip("/") + "/samples"
    target.validation_artifacts = [
        ValidationArtifact(50, 1, f"{remote_samples}/step_000050_1.mp4", "mp4"),
    ]
    test_state.lora_training_runner.reconcile_once()  # downloads feed + polls GPU

    r = client.get("/api/lora/training")
    assert r.status_code == 200
    items = r.json()["items"]
    api_job = next(i for i in items if i["id"] == job.id)
    # Feed item carries a browser-loadable mediaUrl + the prompt source.
    assert len(api_job["validationFeed"]) == 1
    feed_item = api_job["validationFeed"][0]
    assert feed_item["step"] == 50
    assert feed_item["sampleIndex"] == 1
    assert feed_item["source"] == "prompt"
    assert feed_item["mediaType"] == "video"
    assert feed_item["mediaUrl"].endswith(
        f"/api/lora/training/{job.id}/validation-media?step=50&sampleIndex=1&extension=mp4"
    )
    # GPU status is surfaced for the running job.
    assert api_job["gpuStatus"] is not None
    assert api_job["gpuStatus"]["name"] == target.gpu_telemetry.name
    assert api_job["gpuStatus"]["vramTotalMb"] == target.gpu_telemetry.vram_total_mb


def test_runner_skips_feed_when_no_samples_configured(
    test_state, tmp_path, fake_services
) -> None:
    # Standard/t2v with no captions and explicit empty validation prompts ->
    # auto-seed finds nothing -> empty validation_sample_refs -> feed poll is a
    # no-op (validation disabled), so no listing/download happens.
    from state.lora_training_state import LoraClip

    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target
    clip_path = _make_clip_file(tmp_path, "nocap.mp4")
    dataset = handler.create_dataset(
        name="nocap",
        trigger_word=None,
        clips=[LoraClip(id="c0", local_path=clip_path, caption="")],
    )
    handler.request_upload(dataset.id)
    runner.reconcile_once()
    pre = handler.create_preprocessing(
        dataset_id=dataset.id, resolution_buckets="768x448x49",
        with_audio=False, auto_caption=False, captioner_type="qwen_omni",
    )
    runner.reconcile_once()
    runner.reconcile_once()
    job = handler.start_training(
        preprocessed_id=pre.id, name="run",
        config=TrainingConfig(steps=500, validation_interval=50, validation_prompts=[]),
        provider="runpod",
    )
    target.command_results["train.py"] = RemoteCommandStatus(state="running")
    runner.reconcile_once()
    job = handler.get_training(job.id)
    # No captions, no trigger word, empty prompts -> no fallback -> no samples,
    # so the feed poll is a no-op (validation disabled).
    assert job.validation_sample_refs == []

    target.validation_artifacts = [
        ValidationArtifact(50, 1, "/out/samples/step_000050_1.mp4", "mp4"),
    ]
    runner.reconcile_once()
    refreshed = handler.get_training(job.id)
    assert refreshed.validation_feed == []
    assert target.list_validation_calls == []  # never even listed


def test_runner_auto_picks_ic_lora_validation_clip_when_no_holdout(
    test_state, tmp_path, fake_services
) -> None:
    """IC-LoRA with no curated holdout: the first training clip's reference is
    auto-staged and registered as a validation sample, so the feed still covers
    IC-LoRA (monitors progress; the clip is also in the training set)."""
    from services.clip_processor.clip_processor import ClipProbeResult
    from state.lora_training_state import LoraClip

    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target
    fake_services.clip_processor.result = ClipProbeResult(
        duration_seconds=2.0, width=1024, height=576, fps=25.0,
        frame_count=49, has_audio=False, video_codec="h264",
    )
    input_path = _make_clip_file(tmp_path, "input.mp4")
    output_path = _make_clip_file(tmp_path, "output.mp4")
    dataset = handler.create_dataset(
        name="icds",
        dataset_type="ic_lora",
        trigger_word="TOK",
        clips=[
            LoraClip(id="inp", local_path=input_path, caption="", triage="reject"),
            LoraClip(
                id="out", local_path=output_path, caption="A clean-shaven cat.",
                reference_path=input_path,
            ),
        ],
    )
    handler.request_upload(dataset.id)
    runner.reconcile_once()  # upload (auto-stages holdout/out.mp4)
    pre = handler.create_preprocessing(
        dataset_id=dataset.id, resolution_buckets="768x448x49",
        with_audio=False, auto_caption=False, captioner_type="qwen_omni",
    )
    runner.reconcile_once()  # pending -> preprocessing
    runner.reconcile_once()  # preprocessing -> ready
    job = handler.start_training(
        preprocessed_id=pre.id, name="icrun",
        config=TrainingConfig(steps=500, validation_interval=50), provider="runpod",
    )
    target.command_results["train.py"] = RemoteCommandStatus(state="running")
    runner.reconcile_once()  # pending -> running (registers auto-picked sample ref)

    job = handler.get_training(job.id)
    # One reference-conditioned sample, auto-picked from the training clip
    # (without the auto-pick fallback, IC-LoRA with no holdout yields no refs).
    assert len(job.validation_sample_refs) == 1
    ref = job.validation_sample_refs[0]
    assert ref.source == "holdout"
    assert ref.prompt == "A clean-shaven cat."

    remote_samples = job.remote_output_dir.rstrip("/") + "/samples"
    target.validation_artifacts = [
        ValidationArtifact(50, 1, f"{remote_samples}/step_000050_1.mp4", "mp4"),
    ]
    runner.reconcile_once()  # polls -> downloads the auto-picked sample

    refreshed = handler.get_training(job.id)
    assert len(refreshed.validation_feed) == 1
    item = refreshed.validation_feed[0]
    assert item.source == "holdout"
    assert item.prompt == "A clean-shaven cat."
    assert Path(item.local_path).exists()


def test_runner_feeds_ic_lora_holdout_samples(
    test_state, tmp_path, fake_services
) -> None:
    """IC-LoRA with a held-out clip: the holdout reference is staged, registered
    as a validation sample ref, and a downloaded artifact maps to a feed item
    with source="holdout" — the feed covers IC-LoRA."""
    from services.clip_processor.clip_processor import ClipProbeResult
    from state.lora_training_state import LoraClip

    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target
    fake_services.clip_processor.result = ClipProbeResult(
        duration_seconds=2.0, width=1024, height=576, fps=25.0,
        frame_count=49, has_audio=False, video_codec="h264",
    )

    input_path = _make_clip_file(tmp_path, "input.mp4")
    kept_out = _make_clip_file(tmp_path, "kept_out.mp4")
    holdout_out = _make_clip_file(tmp_path, "holdout_out.mp4")
    dataset = handler.create_dataset(
        name="icds",
        dataset_type="ic_lora",
        trigger_word="TOK",
        clips=[
            LoraClip(id="inp", local_path=input_path, caption="", triage="reject"),
            LoraClip(
                id="kept", local_path=kept_out,
                caption="A clean-shaven cat.", reference_path=input_path,
            ),
            LoraClip(
                id="hold", local_path=holdout_out,
                caption="A held-out cat looks around.", reference_path=input_path,
                triage="holdout",
            ),
        ],
    )
    handler.request_upload(dataset.id)
    runner.reconcile_once()  # upload (stages holdout/{id}.mp4)
    pre = handler.create_preprocessing(
        dataset_id=dataset.id, resolution_buckets="768x448x49",
        with_audio=False, auto_caption=False, captioner_type="qwen_omni",
    )
    runner.reconcile_once()  # pending -> preprocessing
    runner.reconcile_once()  # preprocessing -> ready
    job = handler.start_training(
        preprocessed_id=pre.id, name="icrun",
        config=TrainingConfig(steps=500, validation_interval=50), provider="runpod",
    )
    target.command_results["train.py"] = RemoteCommandStatus(state="running")
    runner.reconcile_once()  # pending -> running (registers holdout sample ref)

    job = handler.get_training(job.id)
    assert [r.source for r in job.validation_sample_refs] == ["holdout"]

    remote_samples = job.remote_output_dir.rstrip("/") + "/samples"
    target.validation_artifacts = [
        ValidationArtifact(50, 1, f"{remote_samples}/step_000050_1.mp4", "mp4"),
    ]
    runner.reconcile_once()  # polls -> downloads the holdout sample

    refreshed = handler.get_training(job.id)
    assert len(refreshed.validation_feed) == 1
    item = refreshed.validation_feed[0]
    assert item.step == 50
    assert item.source == "holdout"
    assert item.prompt == "A held-out cat looks around."
    assert item.local_path.endswith("step_000050_1.mp4")
    assert Path(item.local_path).exists()


# ---------------------------------------------------------------------
# Runner: validation-prompt auto-seeding from captions
# ---------------------------------------------------------------------


def test_auto_seed_prefers_trigger_word_caption(test_state) -> None:
    runner = test_state.lora_training_runner
    ds = _dataset_with(["a dog plays", "a TOK cat rests"], trigger="TOK")
    seeded = runner._auto_seed_from_captions(ds)
    # The caption mentioning the trigger word is seeded first.
    assert seeded[0] == "a TOK cat rests"
    assert set(seeded) == {"a TOK cat rests", "a dog plays"}


def test_auto_seed_skips_rejected_and_holdout(test_state) -> None:
    from state.lora_training_state import LoraClip

    runner = test_state.lora_training_runner
    ds = LoraDataset(
        id="ds", name="ds", created_at="2024-01-01T00:00:00Z",
        status="draft", type="standard", trigger_word="TOK",
        clips=[
            LoraClip(id="r", local_path="/tmp/r.mp4", caption="rejected", triage="reject"),
            LoraClip(id="h", local_path="/tmp/h.mp4", caption="holdout", triage="holdout"),
            LoraClip(id="k", local_path="/tmp/k.mp4", caption="a TOK cat"),
        ],
    )
    assert runner._auto_seed_from_captions(ds) == ["a TOK cat"]


def test_effective_prompts_keeps_user_override_and_tops_up_with_captions(test_state) -> None:
    runner = test_state.lora_training_runner
    ds = _dataset_with(["a TOK cat"])
    # Always auto-seed: the user's explicit prompt is honored first, then a
    # trigger-word caption is merged in for diversity (deduped).
    assert runner._effective_validation_prompts(["my custom prompt"], ds) == [
        "my custom prompt",
        "a TOK cat",
    ]


def test_effective_prompts_replaces_placeholder_with_caption(test_state) -> None:
    runner = test_state.lora_training_runner
    ds = _dataset_with(["a TOK cat"])
    assert runner._effective_validation_prompts([DEFAULT_VALIDATION_PROMPT], ds) == ["a TOK cat"]


def test_effective_prompts_empty_no_captions_keeps_empty(test_state) -> None:
    runner = test_state.lora_training_runner
    ds = _dataset_with([""], trigger=None)
    assert runner._effective_validation_prompts([], ds) == []


def test_effective_prompts_fallback_trigger_word_when_no_captions(test_state) -> None:
    runner = test_state.lora_training_runner
    ds = _dataset_with([""], trigger="Zeev")
    # No captions to seed from -> richer trigger-word fallback prompts.
    prompts = runner._effective_validation_prompts([], ds)
    assert all("Zeev" in p for p in prompts)
    assert len(prompts) == 2


def test_auto_seed_caps_at_three(test_state) -> None:
    runner = test_state.lora_training_runner
    ds = _dataset_with(["a TOK cat", "a TOK dog", "a TOK bird", "a TOK fish"])
    assert len(runner._auto_seed_from_captions(ds)) == 3


def test_resolve_validation_dims_matches_bucket_when_default(test_state) -> None:
    from state.lora_training_state import PreprocessedDataset, TrainingConfig, TrainingJob

    runner = test_state.lora_training_runner
    job = TrainingJob(
        id="j",
        name="j",
        created_at="2024-01-01T00:00:00Z",
        preprocessed_id="p",
        status="pending",
        config=TrainingConfig(),  # default validation dims 576x576x49
    )
    pre = PreprocessedDataset(
        id="p",
        dataset_id="ds",
        created_at="2024-01-01T00:00:00Z",
        status="ready",
        resolution_buckets="512x512x49",
    )
    assert runner._resolve_validation_dims(job, pre) == (512, 512, 49)


def test_resolve_validation_dims_respects_custom_dims(test_state) -> None:
    from state.lora_training_state import PreprocessedDataset, TrainingConfig, TrainingJob

    runner = test_state.lora_training_runner
    job = TrainingJob(
        id="j",
        name="j",
        created_at="2024-01-01T00:00:00Z",
        preprocessed_id="p",
        status="pending",
        config=TrainingConfig(
            validation_video_width=640,
            validation_video_height=384,
            validation_video_frames=49,
        ),
    )
    pre = PreprocessedDataset(
        id="p",
        dataset_id="ds",
        created_at="2024-01-01T00:00:00Z",
        status="ready",
        resolution_buckets="512x512x49",
    )
    # Custom (non-sentinel) validation dims are left alone.
    assert runner._resolve_validation_dims(job, pre) is None


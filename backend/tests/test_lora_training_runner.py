"""End-to-end reconciler tests against the FakeTrainerTarget.

Drives the upload -> preprocess -> train -> complete state machine
synchronously via `reconcile_once()` so each transition is asserted
deterministically (the fake target's commands succeed by default; tests
override `status_by_job` to exercise the running/failed branches).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from state.lora_training_state import TrainingConfig
from services.trainer_target.trainer_target import (
    RemoteCommandStatus,
    TrainerTargetError,
)


def _make_clip_file(tmp_path: Path, name: str = "clip0.mp4") -> str:
    path = tmp_path / name
    path.write_bytes(b"fake-video-bytes")
    return str(path)


def _uploaded_dataset(test_state, tmp_path, *, trigger: str | None = "TOK"):
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    clip_path = _make_clip_file(tmp_path)
    from state.lora_training_state import LoraClip

    dataset = handler.create_dataset(
        name="ds",
        trigger_word=trigger,
        clips=[LoraClip(id="c0", local_path=clip_path, caption="a cat")],
    )
    handler.request_upload(dataset.id)
    runner.reconcile_once()
    return handler.get_dataset(dataset.id)


def test_upload_marks_dataset_uploaded(test_state, tmp_path, fake_services) -> None:
    dataset = _uploaded_dataset(test_state, tmp_path)
    assert dataset is not None
    assert dataset.status == "uploaded"
    assert dataset.remote_dataset_dir is not None
    assert dataset.target is not None and dataset.target.pod_id == "fake-pod-1"
    # Staging dir was uploaded.
    assert fake_services.trainer_target.uploaded_dirs


def test_new_dataset_reuses_compatible_idle_runpod_workspace(
    test_state, tmp_path, fake_services
) -> None:
    first = _uploaded_dataset(test_state, tmp_path)
    assert first is not None and first.target is not None
    clip_path = _make_clip_file(tmp_path, name="second.mp4")
    from state.lora_training_state import LoraClip

    second = test_state.lora_training.create_dataset(
        name="second",
        trigger_word=None,
        clips=[LoraClip(id="c1", local_path=clip_path, caption="a dog")],
    )
    test_state.lora_training.request_upload(second.id)
    test_state.lora_training_runner.reconcile_once()

    reused = fake_services.trainer_target.ensure_workspace_calls[-1]
    assert reused is not None
    assert reused.pod_id == first.target.pod_id


def test_compatible_busy_workspace_queues_instead_of_renting_second_pod(
    test_state, tmp_path
) -> None:
    first = _uploaded_dataset(test_state, tmp_path)
    assert first is not None
    test_state.lora_training.create_preprocessing(
        dataset_id=first.id,
        resolution_buckets="768x448x49",
        with_audio=False,
        auto_caption=False,
        captioner_type="gemini_flash",
    )
    clip_path = _make_clip_file(tmp_path, name="waiting.mp4")
    from state.lora_training_state import LoraClip

    second = test_state.lora_training.create_dataset(
        name="waiting",
        trigger_word=None,
        clips=[LoraClip(id="c2", local_path=clip_path, caption="a bird")],
    )
    test_state.lora_training.request_upload(second.id)
    pending = test_state.lora_training.get_dataset(second.id)
    assert pending is not None
    settings = test_state.settings.get_settings_snapshot()
    creds = test_state.lora_training_runner._credentials(settings)

    handle, busy = test_state.lora_training_runner._reusable_runpod_handle(
        pending, creds
    )

    assert handle is None
    assert busy is True


def test_oversized_standard_clip_is_downscaled_for_upload(
    test_state, tmp_path, fake_services
) -> None:
    from services.clip_processor.clip_processor import ClipProbeResult

    # A 4K source (short side 2160 > 768) must be downscaled before upload so the
    # remote trainer doesn't decode huge frames / we don't ship a giant file.
    cp = fake_services.clip_processor
    cp.result = ClipProbeResult(
        duration_seconds=5.0, width=3840, height=2160, fps=24.0,
        frame_count=120, has_audio=False, video_codec="h264",
    )
    dataset = _uploaded_dataset(test_state, tmp_path)
    assert dataset is not None and dataset.status == "uploaded"
    # Exactly one downscale render, capping the short side at 768, aspect kept.
    scales = [c["scale"] for c in cp.render_calls if c["scale"] is not None]
    assert len(scales) == 1
    assert scales[0].height == 768  # short side capped
    assert scales[0].width == 1366  # 3840 * 768/2160, rounded to even


def test_normal_standard_clip_is_copied_not_reencoded(
    test_state, tmp_path, fake_services
) -> None:
    # A 720p source (<= 768) is copied byte-for-byte — no downscale re-encode.
    _uploaded_dataset(test_state, tmp_path)  # fake probes 1280x720 by default
    assert not fake_services.clip_processor.render_calls


def test_one_click_pipeline_runs_upload_preprocess_train(
    test_state, tmp_path, fake_services
) -> None:
    from state.lora_training_state import AutoPipelineSpec, LoraClip, PendingTraining

    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    clip_path = _make_clip_file(tmp_path)
    dataset = handler.create_dataset(
        name="ds", trigger_word="TOK",
        clips=[LoraClip(id="c0", local_path=clip_path, caption="a cat sitting.")],
    )
    spec = AutoPipelineSpec(
        resolution_buckets="768x448x49", with_audio=False, auto_caption=False,
        captioner_type="qwen_omni",
        training=PendingTraining(
            config=TrainingConfig(steps=500), name="run1",
            description="Removes foreground subjects",
            gpu_type="NVIDIA B200", gpu_vram_gb=180,
        ),
    )
    handler.start_training_pipeline(dataset_id=dataset.id, spec=spec)
    assert handler.get_dataset(dataset.id).status == "uploading"

    # No further user action — the reconciler advances every stage on its own.
    for _ in range(6):
        runner.reconcile_once()

    jobs = handler.get_training_state().items
    assert len(jobs) == 1
    assert jobs[0].name == "run1"
    assert jobs[0].description == "Removes foreground subjects"
    assert jobs[0].config.steps == 500
    assert jobs[0].gpu_type == "NVIDIA B200"
    assert jobs[0].status == "completed"
    # Both carry-forward intents were consumed.
    assert handler.get_dataset(dataset.id).auto_pipeline is None
    # Descriptive filename in a per-run folder, with summary + config alongside.
    lora = Path(jobs[0].local_lora_path)
    assert lora.name == "run1-rank32-500steps.safetensors"
    assert lora.parent.name.startswith("run1-")
    assert (lora.parent / "run-summary.md").exists()
    assert (lora.parent / "training-config.json").exists()
    assert (lora.parent / "training-config.yaml").exists()  # exact YAML the trainer ran


def test_one_click_pipeline_threads_local_provider(
    test_state, tmp_path, fake_services
) -> None:
    # A pipeline started with provider="local" must run every stage on the
    # local backend: the chosen provider is persisted on the dataset (read by
    # the upload stage, before any target handle exists), flows into the
    # provisioning credentials, and lands on the auto-started training job.
    from state.lora_training_state import AutoPipelineSpec, LoraClip, PendingTraining

    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target
    clip_path = _make_clip_file(tmp_path)
    dataset = handler.create_dataset(
        name="ds", trigger_word="TOK",
        clips=[LoraClip(id="c0", local_path=clip_path, caption="a cat sitting.")],
    )
    spec = AutoPipelineSpec(
        resolution_buckets="768x448x49", with_audio=False, auto_caption=False,
        captioner_type="qwen_omni",
        training=PendingTraining(config=TrainingConfig(steps=500), name="run1"),
    )
    handler.start_training_pipeline(
        dataset_id=dataset.id, spec=spec, provider="local"
    )
    # Provider persisted on the dataset immediately (drives the upload stage).
    assert handler.get_dataset(dataset.id).provider == "local"

    for _ in range(6):
        runner.reconcile_once()

    # The upload stage provisioned the LOCAL workspace (creds.provider == local).
    assert target.ensure_provisioned_calls
    assert target.ensure_provisioned_calls[0].provider == "local"
    # And the auto-started run inherited the provider end-to-end.
    jobs = handler.get_training_state().items
    assert len(jobs) == 1
    assert jobs[0].provider == "local"


def test_one_click_pipeline_defaults_to_runpod_provider(
    test_state, tmp_path, fake_services
) -> None:
    # Omitting the provider preserves the original behavior: every stage runs
    # on RunPod (default), unchanged from before this phase.
    from state.lora_training_state import AutoPipelineSpec, LoraClip, PendingTraining

    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target
    clip_path = _make_clip_file(tmp_path)
    dataset = handler.create_dataset(
        name="ds", trigger_word="TOK",
        clips=[LoraClip(id="c0", local_path=clip_path, caption="a cat sitting.")],
    )
    spec = AutoPipelineSpec(
        resolution_buckets="768x448x49", with_audio=False, auto_caption=False,
        captioner_type="qwen_omni",
        training=PendingTraining(config=TrainingConfig(steps=500), name="run1"),
    )
    handler.start_training_pipeline(dataset_id=dataset.id, spec=spec)
    assert handler.get_dataset(dataset.id).provider == "runpod"

    for _ in range(6):
        runner.reconcile_once()

    assert target.ensure_provisioned_calls[0].provider == "runpod"
    jobs = handler.get_training_state().items
    assert len(jobs) == 1
    assert jobs[0].provider == "runpod"


def test_upload_provisions_workspace_first(test_state, tmp_path, fake_services) -> None:
    target = fake_services.trainer_target
    dataset = _uploaded_dataset(test_state, tmp_path)
    assert dataset is not None and dataset.status == "uploaded"
    # The workspace was provisioned (bootstrap) before the upload ran.
    assert target.ensure_provisioned_calls
    assert target.uploaded_dirs
    creds = target.ensure_provisioned_calls[0]
    assert creds.auto_provision is True
    assert creds.trainer_repo_url  # default repo url flows through settings


def test_provision_failure_fails_upload(test_state, tmp_path, fake_services) -> None:
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target
    target.raise_on_ensure_provisioned = TrainerTargetError(
        "bootstrap blew up", retryable=False
    )

    from state.lora_training_state import LoraClip

    clip_path = _make_clip_file(tmp_path)
    dataset = handler.create_dataset(
        name="ds",
        trigger_word="TOK",
        clips=[LoraClip(id="c0", local_path=clip_path, caption="a cat")],
    )
    handler.request_upload(dataset.id)
    runner.reconcile_once()

    failed = handler.get_dataset(dataset.id)
    assert failed is not None
    assert failed.status == "upload_failed"
    assert failed.error is not None and "bootstrap blew up" in failed.error
    # Provisioning failed -> nothing was uploaded.
    assert not target.uploaded_dirs
    assert target.released == 1
    assert failed.target is not None and failed.target.pod_id is None


def test_upload_cancel_releases_pod_and_marks_cancelled(
    test_state, tmp_path, fake_services
) -> None:
    """A cancel requested after the pod was acquired makes the next reconcile
    tick release the pod and finalize to `cancelled` — without uploading."""
    from state.lora_training_state import LoraClip, TargetHandle

    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target

    clip_path = _make_clip_file(tmp_path)
    dataset = handler.create_dataset(
        name="ds",
        trigger_word="TOK",
        clips=[LoraClip(id="c0", local_path=clip_path, caption="a cat")],
    )
    handler.request_upload(dataset.id)
    # Simulate "pod acquired, transfer not yet started": the runner is blocked
    # elsewhere and the user cancels.
    handler.set_dataset_target(
        dataset.id, TargetHandle(provider="runpod", pod_id="fake-pod-1")
    )
    before = target.released
    handler.request_cancel_upload(dataset.id)

    runner.reconcile_once()

    done = handler.get_dataset(dataset.id)
    assert done is not None
    assert done.status == "cancelled"
    assert done.cancel_requested is False
    assert done.target is not None and done.target.pod_id is None
    # The pod was released (reclaim billing) and the clips were never uploaded.
    assert target.released == before + 1
    assert not target.uploaded_dirs


def test_upload_cancel_release_failure_retries(
    test_state, tmp_path, fake_services
) -> None:
    """If the pod release fails, the runner must NOT flip to `cancelled` (the
    pod may still be billing); it leaves `cancel_requested` set so the next
    tick retries the release."""
    from state.lora_training_state import LoraClip, TargetHandle

    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target
    target.raise_on_release = TrainerTargetError("release failed", retryable=True)

    clip_path = _make_clip_file(tmp_path)
    dataset = handler.create_dataset(
        name="ds",
        trigger_word="TOK",
        clips=[LoraClip(id="c0", local_path=clip_path, caption="a cat")],
    )
    handler.request_upload(dataset.id)
    handler.set_dataset_target(
        dataset.id, TargetHandle(provider="runpod", pod_id="fake-pod-1")
    )
    handler.request_cancel_upload(dataset.id)

    runner.reconcile_once()

    still_cancelling = handler.get_dataset(dataset.id)
    assert still_cancelling is not None
    # Stays `uploading` with the flag set so the next tick retries the release.
    assert still_cancelling.status == "uploading"
    assert still_cancelling.cancel_requested is True

    # Now the release succeeds -> next tick finalizes the cancel.
    target.raise_on_release = None
    runner.reconcile_once()
    done = handler.get_dataset(dataset.id)
    assert done is not None and done.status == "cancelled"
    assert done.cancel_requested is False


def test_upload_cancel_skips_auto_advance_to_preprocessing(
    test_state, tmp_path, fake_services
) -> None:
    """A cancel that lands during the (non-interruptible) transfer still
    honors the cancel: once the upload finishes, the runner releases the pod
    and marks cancelled instead of auto-advancing into preprocessing."""
    from state.lora_training_state import (
        AutoPipelineSpec,
        LoraClip,
        PendingTraining,
        TargetHandle,
        TrainingConfig,
    )

    handler = test_state.lora_training
    runner = test_state.lora_training_runner

    clip_path = _make_clip_file(tmp_path)
    dataset = handler.create_dataset(
        name="ds",
        trigger_word="TOK",
        clips=[LoraClip(id="c0", local_path=clip_path, caption="a cat")],
    )
    spec = AutoPipelineSpec(
        resolution_buckets="768x448x89",
        with_audio=False,
        auto_caption=True,
        captioner_type="gemini_flash",
        training=PendingTraining(config=TrainingConfig(steps=10), name="run"),
    )
    handler.start_training_pipeline(dataset_id=dataset.id, spec=spec, provider="runpod")
    # Pod already acquired (simulating mid-transfer cancel).
    handler.set_dataset_target(
        dataset.id, TargetHandle(provider="runpod", pod_id="fake-pod-1")
    )
    handler.request_cancel_upload(dataset.id)

    runner.reconcile_once()

    done = handler.get_dataset(dataset.id)
    assert done is not None and done.status == "cancelled"
    # No preprocessing was auto-created from the pipeline intent.
    pre = handler.get_preprocessed_state()
    assert pre.items == []


def test_full_preprocess_and_train_flow(test_state, tmp_path, fake_services) -> None:
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target

    dataset = _uploaded_dataset(test_state, tmp_path)
    assert dataset is not None

    pre = handler.create_preprocessing(
        dataset_id=dataset.id,
        resolution_buckets="768x448x49",
        with_audio=False,
        auto_caption=True,
        captioner_type="gemini_flash",
    )

    # Tick 1: pending -> captioning (caption command submitted).
    runner.reconcile_once()
    assert handler.get_preprocessed(pre.id).status == "captioning"
    assert any("caption_videos.py" in c for c in target.started_commands)

    # Tick 2: caption succeeded -> preprocessing (process_dataset submitted).
    runner.reconcile_once()
    assert handler.get_preprocessed(pre.id).status == "preprocessing"
    assert any("process_dataset.py" in c for c in target.started_commands)
    # Trigger word flows into process_dataset.
    assert any("--lora-trigger" in c for c in target.started_commands)

    # Tick 3: preprocessing succeeded -> ready.
    runner.reconcile_once()
    ready = handler.get_preprocessed(pre.id)
    assert ready.status == "ready"
    assert ready.remote_precomputed_dir is not None
    assert ready.remote_precomputed_dir.endswith(f".precomputed-{pre.id}")

    job = handler.start_training(
        preprocessed_id=pre.id,
        name="run1",
        config=TrainingConfig(steps=500),
        provider="runpod",
    )

    # Tick: pending -> running (train.py submitted, config uploaded).
    runner.reconcile_once()
    running = handler.get_training_state().items[0]
    assert running.status == "running"
    assert any("train.py" in c for c in target.started_commands)
    assert running.remote_output_dir is not None

    # Tick: running -> completed (artifact downloaded locally).
    runner.reconcile_once()
    done = handler.get_training_state().items[0]
    assert done.status == "completed"
    assert done.local_lora_path is not None
    assert Path(done.local_lora_path).exists()
    assert target.downloaded_files


def test_preprocess_low_vram_loads_text_encoder_in_8bit(test_state, tmp_path, fake_services) -> None:
    # low_vram preset -> process_dataset.py gets --load-text-encoder-in-8bit
    # (Gemma3 12B is 23 GB in bf16 and OOMs a 32 GB GPU under WSL2).
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
        preset="low_vram",
    )
    runner.reconcile_once()  # pending -> preprocessing (process_dataset submitted)
    process_cmds = [c for c in target.started_commands if "process_dataset.py" in c]
    assert process_cmds, "process_dataset.py command was not submitted"
    assert "--load-text-encoder-in-8bit" in process_cmds[-1]
    # The preset is stamped on the entity for the training stage to inherit.
    assert handler.get_preprocessed(pre.id).preset == "low_vram"


def test_preprocess_standard_keeps_bf16_text_encoder(test_state, tmp_path, fake_services) -> None:
    # standard preset -> bf16 text encoder; the 8-bit flag must NOT appear.
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target

    dataset = _uploaded_dataset(test_state, tmp_path)
    handler.create_preprocessing(
        dataset_id=dataset.id,
        resolution_buckets="768x448x49",
        with_audio=False,
        auto_caption=False,
        captioner_type="qwen_omni",
        preset="standard",
    )
    runner.reconcile_once()  # pending -> preprocessing
    process_cmds = [c for c in target.started_commands if "process_dataset.py" in c]
    assert process_cmds, "process_dataset.py command was not submitted"
    assert "--load-text-encoder-in-8bit" not in process_cmds[-1]


def test_caption_gemini_key_passed_via_env_not_cli(
    test_state, tmp_path, fake_services
) -> None:
    # The Gemini key must reach caption_videos.py via GEMINI_API_KEY env, never
    # --api-key on the command line (which leaks in `ps` + the job-log echo).
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target
    test_state.state.app_settings.gemini_api_key = "secret-key"

    dataset = _uploaded_dataset(test_state, tmp_path)
    handler.create_preprocessing(
        dataset_id=dataset.id,
        resolution_buckets="768x448x49",
        with_audio=False,
        auto_caption=True,
        captioner_type="gemini_flash",
    )
    runner.reconcile_once()  # pending -> captioning (caption command submitted)
    caption_cmds = [c for c in target.started_commands if "caption_videos.py" in c]
    assert caption_cmds, "caption_videos.py command was not submitted"
    cmd = caption_cmds[-1]
    assert "--api-key" not in cmd
    assert "GEMINI_API_KEY=secret-key " in cmd
    # The key value isn't shlex-quoted here because it's a simple token; but a
    # shell-hostile key would be quoted (covered in test_gemini_key_env_prefix).


def test_caption_qwen_omni_under_40gb_is_rejected(
    test_state, tmp_path, fake_services
) -> None:
    # Qwen3-Omni-30B FP8 needs >=40 GiB VRAM; the fake's default 32 GiB GPU is
    # rejected upfront with a clear message instead of a confusing vLLM OOM,
    # and no caption command is started.
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target

    dataset = _uploaded_dataset(test_state, tmp_path)
    pre = handler.create_preprocessing(
        dataset_id=dataset.id,
        resolution_buckets="768x448x49",
        with_audio=False,
        auto_caption=True,
        captioner_type="qwen_omni",
        preset="low_vram",
    )
    runner.reconcile_once()  # -> rejected (gate fires before captioning)
    item = handler.get_preprocessed(pre.id)
    assert item is not None
    assert item.status == "failed"
    assert "40" in (item.error or "")
    assert "Gemini" in (item.error or "")
    caption_cmds = [c for c in target.started_commands if "caption_videos.py" in c]
    assert not caption_cmds


def test_caption_qwen_omni_40gb_starts_server_flow(
    test_state, tmp_path, fake_services
) -> None:
    # On a >=40 GiB GPU, qwen_omni starts serve_captioner.py + caption_videos.py
    # with --vllm-url (the two-process flow), and never emits the invalid
    # --use-8bit flag (quantization is a serve_captioner.py concern).
    import dataclasses

    fake_services.trainer_target.gpu_telemetry = dataclasses.replace(
        fake_services.trainer_target.gpu_telemetry, vram_total_mb=49152
    )
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target

    dataset = _uploaded_dataset(test_state, tmp_path)
    handler.create_preprocessing(
        dataset_id=dataset.id,
        resolution_buckets="768x448x49",
        with_audio=False,
        auto_caption=True,
        captioner_type="qwen_omni",
        preset="low_vram",
    )
    runner.reconcile_once()  # -> captioning
    caption_cmds = [c for c in target.started_commands if "caption_videos.py" in c]
    assert caption_cmds
    cmd = caption_cmds[-1]
    assert "serve_captioner.py" in cmd
    assert "--quantization fp8" in cmd
    assert "--vllm-url http://127.0.0.1:8001/v1" in cmd
    assert "trap cleanup EXIT" in cmd
    assert "--use-8bit" not in cmd


def _uploaded_ic_lora_dataset(test_state, tmp_path, fake_services):
    """Upload a minimal IC-LoRA dataset (one input->output pair) and return it."""
    from services.clip_processor.clip_processor import ClipProbeResult
    from state.lora_training_state import LoraClip

    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    # Both clips probe as a valid 25fps / 49-frame clip so upload normalizes them.
    fake_services.clip_processor.result = ClipProbeResult(
        duration_seconds=2.0, width=1024, height=576, fps=25.0,
        frame_count=49, has_audio=False, video_codec="h264",
    )
    input_path = _make_clip_file(tmp_path, "input.mp4")
    output_path = _make_clip_file(tmp_path, "output.mp4")
    dataset = handler.create_dataset(
        name="ic ds",
        dataset_type="ic_lora",
        trigger_word="TOK",
        clips=[
            LoraClip(id="inp", local_path=input_path, caption="", triage="reject"),
            LoraClip(
                id="out",
                local_path=output_path,
                caption="A clean-shaven cat.",
                reference_path=input_path,
            ),
        ],
    )
    handler.request_upload(dataset.id)
    runner.reconcile_once()
    return handler.get_dataset(dataset.id)


def test_preprocess_ic_lora_low_vram_downscales_references(
    test_state, tmp_path, fake_services
) -> None:
    # IC-LoRA + low_vram -> process_dataset.py gets --reference-downscale-factor 2
    # so the concatenated ref+target sequence's backward fits a 32 GB card.
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target

    dataset = _uploaded_ic_lora_dataset(test_state, tmp_path, fake_services)
    assert dataset is not None and dataset.type == "ic_lora"
    handler.create_preprocessing(
        dataset_id=dataset.id,
        resolution_buckets="768x448x49",
        with_audio=False,
        auto_caption=False,
        captioner_type="gemini_flash",
        preset="low_vram",
    )
    runner.reconcile_once()  # IC-LoRA skips captioning -> process_dataset submitted
    process_cmds = [c for c in target.started_commands if "process_dataset.py" in c]
    assert process_cmds, "process_dataset.py command was not submitted"
    assert "--reference-downscale-factor 2" in process_cmds[-1]


def test_preprocess_ic_lora_standard_keeps_full_references(
    test_state, tmp_path, fake_services
) -> None:
    # IC-LoRA + standard (80 GB+ card) -> full-size references, matching the
    # official v2v_ic_lora.yaml (downscale_factor: 1); no downscale flag.
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target

    dataset = _uploaded_ic_lora_dataset(test_state, tmp_path, fake_services)
    handler.create_preprocessing(
        dataset_id=dataset.id,
        resolution_buckets="768x448x49",
        with_audio=False,
        auto_caption=False,
        captioner_type="gemini_flash",
        preset="standard",
    )
    runner.reconcile_once()
    process_cmds = [c for c in target.started_commands if "process_dataset.py" in c]
    assert process_cmds, "process_dataset.py command was not submitted"
    assert "--reference-downscale-factor" not in process_cmds[-1]


def test_preprocess_ic_lora_restages_for_selected_bucket(
    test_state, tmp_path, fake_services
) -> None:
    from services.clip_processor.clip_processor import ClipProbeResult

    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target
    dataset = _uploaded_ic_lora_dataset(test_state, tmp_path, fake_services)
    initial_uploads = len(target.uploaded_dirs)

    fake_services.clip_processor.result = ClipProbeResult(
        duration_seconds=4.0,
        width=1024,
        height=576,
        fps=25.0,
        frame_count=81,
        has_audio=False,
        video_codec="h264",
    )
    handler.create_preprocessing(
        dataset_id=dataset.id,
        resolution_buckets="768x448x81",
        with_audio=False,
        auto_caption=False,
        captioner_type="gemini_flash",
    )

    runner.reconcile_once()

    assert len(target.uploaded_dirs) == initial_uploads + 1
    assert fake_services.clip_processor.normalize_calls[-1]["frames"] == 81
    refreshed = handler.get_dataset(dataset.id)
    assert refreshed.ic_staged_short_side == 448
    assert refreshed.ic_staged_bucket_frames == 81


def test_preprocess_ic_lora_low_vram_collapses_multi_bucket_to_single(
    test_state, tmp_path, fake_services
) -> None:
    # process_dataset.py rejects multiple buckets when reference downscaling is
    # on, so a multi-bucket IC-LoRA low_vram run must collapse to a single
    # bucket (the first) AND keep the downscale flag.
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target

    dataset = _uploaded_ic_lora_dataset(test_state, tmp_path, fake_services)
    pre = handler.create_preprocessing(
        dataset_id=dataset.id,
        resolution_buckets="768x448x49;512x512x49",
        with_audio=False,
        auto_caption=False,
        captioner_type="gemini_flash",
        preset="low_vram",
    )
    runner.reconcile_once()
    process_cmds = [c for c in target.started_commands if "process_dataset.py" in c]
    assert process_cmds, "process_dataset.py command was not submitted"
    cmd = process_cmds[-1]
    assert "--reference-downscale-factor 2" in cmd
    # Only the first bucket is passed; the second is dropped.
    assert "768x448x49" in cmd
    assert "512x512x49" not in cmd
    # The bucket list has no semicolon (single bucket).
    buckets_arg = cmd.split("--resolution-buckets")[1].split()[0]
    assert ";" not in buckets_arg
    # The collapse is surfaced: the effective (collapsed) bucket is persisted
    # on the preprocessed record so the UI/run-summary shows the real trained
    # resolution instead of the uncollapsed list.
    assert handler.get_preprocessed(pre.id).effective_resolution_buckets == "768x448x49"


def test_preprocess_standard_dataset_never_downscales_references(
    test_state, tmp_path, fake_services
) -> None:
    # Text-to-video has no reference latents, so the downscale flag must never
    # appear even under low_vram (it's gated on the dataset being IC-LoRA).
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target

    dataset = _uploaded_dataset(test_state, tmp_path)
    handler.create_preprocessing(
        dataset_id=dataset.id,
        resolution_buckets="768x448x49",
        with_audio=False,
        auto_caption=False,
        captioner_type="qwen_omni",
        preset="low_vram",
    )
    runner.reconcile_once()
    process_cmds = [c for c in target.started_commands if "process_dataset.py" in c]
    assert process_cmds, "process_dataset.py command was not submitted"
    assert "--reference-downscale-factor" not in process_cmds[-1]


def test_retry_download_recovers_finished_run(test_state, tmp_path, fake_services) -> None:
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
    runner.reconcile_once()  # pending -> preprocessing
    runner.reconcile_once()  # preprocessing -> ready
    job = handler.start_training(
        preprocessed_id=pre.id, name="run1", config=TrainingConfig(steps=500),
        provider="runpod",
    )
    runner.reconcile_once()  # pending -> running
    runner.reconcile_once()  # running -> completed (download)
    assert handler.get_training_state().items[0].status == "completed"

    # Simulate the original download having failed (the trained adapter still
    # lives on the volume). The retry must re-fetch it without re-training.
    handler.fail_training(job.id, "Download failed: Remote artifact not found: …")
    downloads_before = len(target.downloaded_files)
    started_before = len(target.started_commands)

    retried = handler.request_training_redownload(job.id)
    assert retried.status == "running" and retried.redownload_requested is True

    runner.reconcile_once()
    done = handler.get_training_state().items[0]
    assert done.status == "completed"
    assert done.local_lora_path is not None
    # It downloaded again but did NOT start any new training command.
    assert len(target.downloaded_files) > downloads_before
    assert len(target.started_commands) == started_before


def test_retry_download_rejects_non_failed_run(test_state, tmp_path, fake_services) -> None:
    from handlers.lora_training_handler import LoraTransitionError

    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    dataset = _uploaded_dataset(test_state, tmp_path)
    pre = handler.create_preprocessing(
        dataset_id=dataset.id, resolution_buckets="768x448x49",
        with_audio=False, auto_caption=False, captioner_type="qwen_omni",
    )
    runner.reconcile_once()
    runner.reconcile_once()
    job = handler.start_training(
        preprocessed_id=pre.id, name="run1", config=TrainingConfig(steps=500),
        provider="runpod",
    )
    # pending (not failed) -> retry is rejected.
    with pytest.raises(LoraTransitionError):
        handler.request_training_redownload(job.id)


def test_redownload_picks_highest_existing_remote_checkpoint(
    test_state, tmp_path, fake_services
) -> None:
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target

    dataset = _uploaded_dataset(test_state, tmp_path)
    pre = handler.create_preprocessing(
        dataset_id=dataset.id, resolution_buckets="768x448x49",
        with_audio=False, auto_caption=False, captioner_type="qwen_omni",
    )
    runner.reconcile_once()
    runner.reconcile_once()
    job = handler.start_training(
        preprocessed_id=pre.id, name="run1", config=TrainingConfig(steps=500),
        provider="runpod",
    )
    runner.reconcile_once()  # -> running
    runner.reconcile_once()  # -> completed (download)
    assert handler.get_training_state().items[0].status == "completed"

    # The original download failed; the trained adapters still live on the
    # volume. Simulate the trainer having saved several checkpoints, with the
    # highest NOT equal to the configured final step (500) — a fresh-pod
    # redownload can't read the (gone) training log, so it must list the
    # checkpoints dir and pick the highest existing file.
    handler.fail_training(job.id, "Download failed: transient")
    target.downloaded_files.clear()
    target.checkpoints_steps = [100, 250, 475]

    retried = handler.request_training_redownload(job.id)
    assert retried.status == "running" and retried.redownload_requested is True

    runner.reconcile_once()
    done = handler.get_training_state().items[0]
    assert done.status == "completed"
    # The redownload fetched the highest existing checkpoint (475), not the
    # configured final step (500) — and it actually consulted the remote dir.
    assert target.list_checkpoints_calls
    assert target.downloaded_files
    # The weights download is the safetensors entry (the run-config YAML is
    # also fetched, best-effort, right after).
    weights = [p for p, _ in target.downloaded_files if p.endswith(".safetensors")]
    assert weights, target.downloaded_files
    assert weights[-1].endswith("lora_weights_step_00475.safetensors")


def test_training_persists_latest_checkpoint_step(
    test_state, tmp_path, fake_services
) -> None:
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target

    dataset = _uploaded_dataset(test_state, tmp_path)
    pre = handler.create_preprocessing(
        dataset_id=dataset.id, resolution_buckets="768x448x49",
        with_audio=False, auto_caption=False, captioner_type="qwen_omni",
    )
    runner.reconcile_once()
    runner.reconcile_once()
    handler.start_training(
        preprocessed_id=pre.id, name="run1", config=TrainingConfig(steps=500),
        provider="runpod",
    )
    runner.reconcile_once()  # -> running
    running = handler.get_training_state().items[0]
    assert running.target is not None
    # Keep training running and emit a checkpoint-save log line.
    target.status_by_job[running.target.remote_job_id] = [
        RemoteCommandStatus(state="running")
    ]
    target.logs_by_job[running.target.remote_job_id] = [
        "step 150/500 loss 0.12",
        "saved in checkpoints/lora_weights_step_00150.safetensors",
    ]
    runner.reconcile_once()  # poll -> persists latest_checkpoint_step
    polled = handler.get_training_state().items[0]
    assert polled.status == "running"
    assert polled.latest_checkpoint_step == 150


def test_preprocess_resume_skips_recaptioning_and_reruns_process_dataset(
    test_state, tmp_path, fake_services
) -> None:
    from handlers.lora_training_handler import LoraTransitionError

    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target

    dataset = _uploaded_dataset(test_state, tmp_path)
    pre = handler.create_preprocessing(
        dataset_id=dataset.id,
        resolution_buckets="768x448x89",
        with_audio=False,
        auto_caption=True,  # captioning runs before process_dataset
        captioner_type="gemini_flash",
    )
    runner.reconcile_once()  # pending -> captioning
    assert handler.get_preprocessed(pre.id).status == "captioning"
    runner.reconcile_once()  # captioning -> preprocessing (process_dataset submitted)
    assert handler.get_preprocessed(pre.id).captioning_completed is True

    # process_dataset fails (e.g. OOM during latent caching).
    target.command_results["process_dataset.py"] = RemoteCommandStatus(
        state="failed", exit_code=137, error="OOM"
    )
    runner.reconcile_once()  # poll -> failed
    assert handler.get_preprocessed(pre.id).status == "failed"

    # Resume: must reuse the workspace + captions, NOT re-caption.
    target.command_results.pop("process_dataset.py", None)  # succeed on re-run
    resumed = handler.request_preprocess_resume(pre.id)
    assert resumed.status == "pending"
    assert resumed.captioning_completed is True

    caption_cmds_before = [c for c in target.started_commands if "caption_videos.py" in c]
    runner.reconcile_once()  # pending -> preprocessing (straight to process_dataset)
    after = handler.get_preprocessed(pre.id)
    assert after.status == "preprocessing"
    caption_cmds_after = [c for c in target.started_commands if "caption_videos.py" in c]
    assert caption_cmds_after == caption_cmds_before, "resume re-captioned instead of skipping"
    assert any("process_dataset.py" in c for c in target.started_commands)

    # Resume is rejected for a non-terminal (active) item.
    with pytest.raises(LoraTransitionError):
        handler.request_preprocess_resume(pre.id)


def test_preprocess_reset_wipes_precomputed_and_recaptions(
    test_state, tmp_path, fake_services
) -> None:
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target

    dataset = _uploaded_dataset(test_state, tmp_path)
    pre = handler.create_preprocessing(
        dataset_id=dataset.id,
        resolution_buckets="768x448x89",
        with_audio=False,
        auto_caption=True,
        captioner_type="gemini_flash",
    )
    runner.reconcile_once()  # -> captioning
    runner.reconcile_once()  # -> preprocessing (captioning_completed=True)
    target.command_results["process_dataset.py"] = RemoteCommandStatus(
        state="failed", exit_code=1, error="boom"
    )
    runner.reconcile_once()  # -> failed
    assert handler.get_preprocessed(pre.id).status == "failed"

    target.deleted_remote_paths.clear()
    reset = handler.request_preprocess_reset(pre.id)
    assert reset.status == "pending"
    assert reset.captioning_completed is False
    assert reset.reset_requested is True

    runner.reconcile_once()  # wipe .precomputed, then re-caption (fresh start)
    after = handler.get_preprocessed(pre.id)
    assert after.reset_requested is False
    # The remote .precomputed latent cache was wiped.
    assert target.deleted_remote_paths
    assert any(".precomputed" in p for p in target.deleted_remote_paths[-1])
    # Reset re-captions (captioning_completed was cleared), so a caption command
    # was just submitted — back at `captioning`.
    assert after.status == "captioning"
    assert any("caption_videos.py" in c for c in target.started_commands)
    assert "--override" in target.started_commands[-1]


def test_training_resume_sets_load_checkpoint_from_latest_step(
    test_state, tmp_path, fake_services
) -> None:
    from handlers.lora_command_builder import lora_checkpoint_path_in

    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target

    dataset = _uploaded_dataset(test_state, tmp_path)
    pre = handler.create_preprocessing(
        dataset_id=dataset.id, resolution_buckets="768x448x49",
        with_audio=False, auto_caption=False, captioner_type="qwen_omni",
    )
    runner.reconcile_once()
    runner.reconcile_once()  # -> ready
    job = handler.start_training(
        preprocessed_id=pre.id, name="run1", config=TrainingConfig(steps=500),
        provider="runpod",
    )
    runner.reconcile_once()  # -> running
    running = handler.get_training_state().items[0]
    assert running.target is not None and running.remote_output_dir is not None
    assert running.compute_rate_per_hr == 1.89

    # Emit a checkpoint-save log line so latest_checkpoint_step persists, then
    # fail the run (e.g. OOM after the first checkpoint).
    target.status_by_job[running.target.remote_job_id] = [
        RemoteCommandStatus(state="running")
    ]
    target.logs_by_job[running.target.remote_job_id] = [
        "saved in checkpoints/lora_weights_step_00250.safetensors",
    ]
    runner.reconcile_once()  # poll -> persists latest_checkpoint_step=250
    assert handler.get_training_state().items[0].latest_checkpoint_step == 250

    handler.fail_training(job.id, "OOM mid-run")
    assert handler.get_training_state().items[0].status == "failed"

    resumed = handler.request_training_resume(job.id)
    assert resumed.status == "pending"
    # load_checkpoint points at the highest saved checkpoint, so train.py
    # continues from step 250 instead of restarting at step 0.
    expected = lora_checkpoint_path_in(running.remote_output_dir, 250)
    assert resumed.config.load_checkpoint == expected

    runner.reconcile_once()  # pending -> running (train.py re-submitted)
    again = handler.get_training_state().items[0]
    assert again.status == "running"
    assert any("train.py" in c for c in target.started_commands)


def test_training_reset_wipes_remote_output_and_local_rundir(
    test_state, tmp_path, fake_services
) -> None:
    from handlers.lora_training_handler import LoraTransitionError

    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target

    dataset = _uploaded_dataset(test_state, tmp_path)
    pre = handler.create_preprocessing(
        dataset_id=dataset.id, resolution_buckets="768x448x49",
        with_audio=False, auto_caption=False, captioner_type="qwen_omni",
    )
    runner.reconcile_once()
    runner.reconcile_once()  # -> ready
    job = handler.start_training(
        preprocessed_id=pre.id, name="run1", config=TrainingConfig(steps=500),
        provider="runpod",
    )
    runner.reconcile_once()  # -> running
    runner.reconcile_once()  # -> completed (artifact downloaded)
    completed = handler.get_training_state().items[0]
    assert completed.status == "completed"
    assert completed.remote_output_dir is not None
    local_run_dir = runner._local_run_dir(completed)
    assert local_run_dir.exists()

    target.deleted_remote_paths.clear()
    reset = handler.request_training_reset(job.id)
    assert reset.status == "pending"
    assert reset.config.load_checkpoint is None
    assert reset.reset_requested is True
    assert reset.validation_feed == []
    assert reset.current_step is None

    runner.reconcile_once()  # wipe remote output + local run dir, then re-train
    after = handler.get_training_state().items[0]
    assert after.reset_requested is False
    assert target.deleted_remote_paths
    assert target.deleted_remote_paths[-1] == [completed.remote_output_dir]
    assert not local_run_dir.exists()
    assert after.status == "running"
    assert any("train.py" in c for c in target.started_commands)

    # Reset is rejected for an active (running) job.
    with pytest.raises(LoraTransitionError):
        handler.request_training_reset(job.id)


def test_upload_uses_readable_slug_and_renames_clips(test_state, tmp_path, fake_services) -> None:
    import json

    dataset = _uploaded_dataset(test_state, tmp_path)
    assert dataset is not None
    # Folder is browseable: "<name>-<shortid>", not a bare UUID.
    assert dataset.remote_dataset_dir is not None
    assert dataset.remote_dataset_dir.endswith(f"/datasets/ds-{dataset.id[:8]}")
    # Manual caption present -> dataset.json staged with the renamed clip.
    local_dir, _ = fake_services.trainer_target.uploaded_dirs[-1]
    rows = json.loads((Path(local_dir) / "dataset.json").read_text())
    assert len(rows) == 1
    assert rows[0]["media_path"].endswith("/clips/0001_clip0.mp4")
    assert "reference_path" not in rows[0]  # standard LoRA has no references


def test_ic_lora_upload_emits_pairs_and_skips_caption(test_state, tmp_path, fake_services) -> None:
    import json

    from services.clip_processor.clip_processor import ClipProbeResult
    from state.lora_training_state import LoraClip

    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target
    # Every normalized output probes as a valid 25fps / 49-frame / 1024x576 clip.
    fake_services.clip_processor.result = ClipProbeResult(
        duration_seconds=2.0, width=1024, height=576, fps=25.0,
        frame_count=49, has_audio=False, video_codec="h264",
    )

    input_path = _make_clip_file(tmp_path, "input.mp4")
    output_path = _make_clip_file(tmp_path, "output.mp4")
    dataset = handler.create_dataset(
        name="edit ds",
        dataset_type="ic_lora",
        trigger_word="TOK",
        # The output is the training target; the input is its reference. Mark
        # the input rejected so it isn't a training row, but it must still be
        # pulled in + named as the reference.
        clips=[
            LoraClip(id="inp", local_path=input_path, caption="", triage="reject"),
            LoraClip(
                id="out",
                local_path=output_path,
                caption="A clean-shaven cat.",
                reference_path=input_path,
            ),
        ],
    )
    handler.request_upload(dataset.id)
    runner.reconcile_once()

    uploaded = handler.get_dataset(dataset.id)
    assert uploaded is not None and uploaded.status == "uploaded"
    local_dir, _ = target.uploaded_dirs[-1]
    rows = json.loads((Path(local_dir) / "dataset.json").read_text())
    # Exactly one PAIR in the trainer's {caption, video, reference_video} schema.
    assert len(rows) == 1
    assert set(rows[0]) == {"caption", "video", "reference_video"}
    assert rows[0]["video"].endswith("/clips/0001_output_output.mp4")
    assert rows[0]["reference_video"].endswith("/clips/0001_reference_input.mp4")
    staged = {p.name for p in (Path(local_dir) / "clips").iterdir()}
    assert staged == {"0001_output_output.mp4", "0001_reference_input.mp4"}

    # Auto-caption requested, but IC-LoRA must skip remote captioning (it would
    # clobber the reference pairing) and go straight to process_dataset. The
    # trainer auto-detects the reference_video column, so no flag is emitted;
    # IC-LoRA never trains audio, so it must pass --skip-audio.
    pre = handler.create_preprocessing(
        dataset_id=dataset.id,
        resolution_buckets="768x448x49",
        with_audio=False,
        auto_caption=True,
        captioner_type="gemini_flash",
    )
    runner.reconcile_once()
    assert handler.get_preprocessed(pre.id).status == "preprocessing"
    assert not any("caption_videos.py" in c for c in target.started_commands)
    assert not any("--reference-column" in c for c in target.started_commands)
    assert any(
        "process_dataset.py" in c and "--skip-audio" in c
        for c in target.started_commands
    )


def test_preprocess_failure_surfaces(test_state, tmp_path, fake_services) -> None:
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target

    dataset = _uploaded_dataset(test_state, tmp_path)
    assert dataset is not None
    pre = handler.create_preprocessing(
        dataset_id=dataset.id,
        resolution_buckets="768x448x89",
        with_audio=False,
        auto_caption=False,  # skip caption -> first step is process_dataset
        captioner_type="gemini_flash",
    )
    # Make the process_dataset command fail.
    target.command_results["process_dataset.py"] = RemoteCommandStatus(
        state="failed", exit_code=1, error="boom"
    )

    runner.reconcile_once()  # pending -> preprocessing (submitted)
    assert handler.get_preprocessed(pre.id).status == "preprocessing"
    runner.reconcile_once()  # poll -> failed
    failed = handler.get_preprocessed(pre.id)
    assert failed.status == "failed"
    assert failed.error is not None


def test_preprocess_audio_guard_fails_when_no_audio_latents(
    test_state, tmp_path, fake_services
) -> None:
    # with_audio on but process_dataset.py wrote 0 audio_latents (e.g. the audio
    # model failed to load) -> the guard fails the preprocess fast with a clear
    # message instead of marking it ready and crashing the trainer later with
    # "No valid samples found".
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target

    dataset = _uploaded_dataset(test_state, tmp_path)
    pre = handler.create_preprocessing(
        dataset_id=dataset.id,
        resolution_buckets="768x448x89",
        with_audio=True,
        auto_caption=False,  # skip caption -> first step is process_dataset
        captioner_type="gemini_flash",
    )
    # Default fake reports latents=8 but no audio_latents -> audio count is 0.
    assert target.precomputed_source_counts.get("audio_latents", 0) == 0

    runner.reconcile_once()  # pending -> preprocessing (submitted)
    runner.reconcile_once()  # process_dataset succeeds -> guard runs
    guarded = handler.get_preprocessed(pre.id)
    assert guarded.status == "failed"
    assert guarded.error is not None
    assert "audio" in guarded.error.lower()
    assert "audio_latents" in target.count_precomputed_source_calls


def test_preprocess_audio_guard_surfaces_trainer_skip_summary(
    test_state, tmp_path, fake_services
) -> None:
    # When the trainer's own log reports "Audio processing: 0 ... skipped" (the
    # signature of torchaudio's ffmpeg backend being unavailable on the pod),
    # the guard must fold that report into the failure message so "prep failed"
    # self-diagnoses instead of looking like a generic audio-model error.
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target

    dataset = _uploaded_dataset(test_state, tmp_path)
    pre = handler.create_preprocessing(
        dataset_id=dataset.id,
        resolution_buckets="768x448x89",
        with_audio=True,
        auto_caption=False,
        captioner_type="gemini_flash",
    )
    runner.reconcile_once()  # pending -> process_dataset submitted
    # Inject the trainer's audio-skip self-report into the fake's log tail.
    job = handler.get_preprocessed(pre.id).target
    assert job is not None and job.remote_job_id is not None
    target.logs_by_job[job.remote_job_id] = [
        "Processing videos 100%",
        "Audio processing: 0 videos with audio, 8 without audio (skipped)",
        "Could not extract audio from clips/0001.mp4: Failed to load audio",
    ]
    runner.reconcile_once()  # guard runs -> fails with the skip detail
    guarded = handler.get_preprocessed(pre.id)
    assert guarded.status == "failed"
    assert guarded.error is not None
    assert "0 clip(s) with audio, 8 skipped" in guarded.error
    assert "Failed to load audio" in guarded.error


def test_preprocess_audio_guard_passes_when_audio_latents_present(
    test_state, tmp_path, fake_services
) -> None:
    # with_audio on AND audio_latents were written -> guard passes, preprocess
    # becomes ready (regression guard: the guard must not false-fail a good run).
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target

    dataset = _uploaded_dataset(test_state, tmp_path)
    pre = handler.create_preprocessing(
        dataset_id=dataset.id,
        resolution_buckets="768x448x89",
        with_audio=True,
        auto_caption=False,
        captioner_type="gemini_flash",
    )
    target.precomputed_source_counts["audio_latents"] = 8

    runner.reconcile_once()  # pending -> preprocessing (submitted)
    runner.reconcile_once()  # process_dataset succeeds -> guard passes -> ready
    ready = handler.get_preprocessed(pre.id)
    assert ready.status == "ready"
    assert ready.remote_precomputed_dir is not None
    assert "audio_latents" in target.count_precomputed_source_calls


def test_preprocess_guard_fails_when_no_latents(
    test_state, tmp_path, fake_services
) -> None:
    # process_dataset.py exited 0 but wrote no latents at all -> guard fails with
    # the latents message (audio is not queried when with_audio is off).
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target

    dataset = _uploaded_dataset(test_state, tmp_path)
    pre = handler.create_preprocessing(
        dataset_id=dataset.id,
        resolution_buckets="768x448x89",
        with_audio=False,
        auto_caption=False,
        captioner_type="gemini_flash",
    )
    target.precomputed_source_counts["latents"] = 0

    runner.reconcile_once()  # pending -> preprocessing (submitted)
    runner.reconcile_once()  # process_dataset succeeds -> guard fails (no latents)
    guarded = handler.get_preprocessed(pre.id)
    assert guarded.status == "failed"
    assert guarded.error is not None
    assert "latents" in guarded.error.lower()
    assert "audio_latents" not in target.count_precomputed_source_calls


def test_preprocess_guard_fails_when_conditions_are_incomplete(
    test_state, tmp_path, fake_services
) -> None:
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target
    dataset = _uploaded_dataset(test_state, tmp_path)
    pre = handler.create_preprocessing(
        dataset_id=dataset.id,
        resolution_buckets="768x448x49",
        with_audio=False,
        auto_caption=False,
        captioner_type="gemini_flash",
    )
    target.precomputed_source_counts["conditions"] = 7

    runner.reconcile_once()
    runner.reconcile_once()

    guarded = handler.get_preprocessed(pre.id)
    assert guarded.status == "failed"
    assert guarded.error is not None
    assert "conditioning" in guarded.error.lower()


def test_ic_preprocess_guard_fails_when_references_are_incomplete(
    test_state, tmp_path, fake_services
) -> None:
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target
    dataset = _uploaded_ic_lora_dataset(test_state, tmp_path, fake_services)
    pre = handler.create_preprocessing(
        dataset_id=dataset.id,
        resolution_buckets="768x448x49",
        with_audio=False,
        auto_caption=False,
        captioner_type="gemini_flash",
    )
    target.precomputed_source_counts["reference_latents"] = 0

    runner.reconcile_once()
    runner.reconcile_once()

    guarded = handler.get_preprocessed(pre.id)
    assert guarded.status == "failed"
    assert guarded.error is not None
    assert "reference cache" in guarded.error.lower()


def test_training_cancel_terminates_remote(test_state, tmp_path, fake_services) -> None:
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target

    dataset = _uploaded_dataset(test_state, tmp_path)
    assert dataset is not None
    pre = handler.create_preprocessing(
        dataset_id=dataset.id,
        resolution_buckets="768x448x89",
        with_audio=False,
        auto_caption=False,
        captioner_type="gemini_flash",
    )
    runner.reconcile_once()  # submit process
    runner.reconcile_once()  # ready
    assert handler.get_preprocessed(pre.id).status == "ready"

    job = handler.start_training(
        preprocessed_id=pre.id,
        name="run1",
        config=TrainingConfig(),
        provider="runpod",
    )
    # Keep training "running" so cancel exercises the terminate path.
    runner.reconcile_once()  # -> running
    running = handler.get_training_state().items[0]
    assert running.status == "running"
    assert running.target is not None
    target.status_by_job[running.target.remote_job_id] = [
        RemoteCommandStatus(state="running")
    ]

    handler.request_cancel_training(job.id)
    runner.reconcile_once()  # observes cancel_requested -> terminate -> cancelled
    cancelled = handler.get_training_state().items[0]
    assert cancelled.status == "cancelled"
    assert target.terminated


def test_preprocess_cancel_terminates_remote(test_state, tmp_path, fake_services) -> None:
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target

    dataset = _uploaded_dataset(test_state, tmp_path)
    assert dataset is not None
    pre = handler.create_preprocessing(
        dataset_id=dataset.id,
        resolution_buckets="768x448x89",
        with_audio=False,
        auto_caption=True,  # first step is caption -> reaches `captioning` with a remote job
        captioner_type="gemini_flash",
    )
    runner.reconcile_once()  # pending -> captioning (caption command submitted)
    captioning = handler.get_preprocessed(pre.id)
    assert captioning.status == "captioning"
    assert captioning.target is not None and captioning.target.remote_job_id
    remote_job_id = captioning.target.remote_job_id

    # User cancels: the handler must NOT flip to `cancelled` on the spot (that
    # would orphan the still-running remote caption job). It sets a flag and
    # keeps status as `captioning` so the reconciler can see + terminate it.
    handler.request_cancel_preprocessing(pre.id)
    flagged = handler.get_preprocessed(pre.id)
    assert flagged.status == "captioning"
    assert flagged.cancel_requested is True

    runner.reconcile_once()  # observes cancel_requested -> terminate -> cancelled
    cancelled = handler.get_preprocessed(pre.id)
    assert cancelled.status == "cancelled"
    assert cancelled.cancel_requested is False
    assert remote_job_id in target.terminated


def test_preprocess_cancel_terminate_failure_retries(
    test_state, tmp_path, fake_services
) -> None:
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target

    dataset = _uploaded_dataset(test_state, tmp_path)
    assert dataset is not None
    pre = handler.create_preprocessing(
        dataset_id=dataset.id,
        resolution_buckets="768x448x89",
        with_audio=False,
        auto_caption=True,
        captioner_type="gemini_flash",
    )
    runner.reconcile_once()  # -> captioning
    captioning = handler.get_preprocessed(pre.id)
    remote_job_id = captioning.target.remote_job_id
    handler.request_cancel_preprocessing(pre.id)

    # Terminate fails (transient SSH blip): the runner must NOT declare
    # `cancelled` while the remote job keeps billing — leave `cancel_requested`
    # set and status `captioning` so the next tick retries.
    target.raise_on_terminate = TrainerTargetError("ssh: connection reset", retryable=True)
    runner.reconcile_once()
    still_cancelling = handler.get_preprocessed(pre.id)
    assert still_cancelling.status == "captioning"
    assert still_cancelling.cancel_requested is True
    assert target.terminated == []

    # Next tick, terminate succeeds -> finalized.
    target.raise_on_terminate = None
    runner.reconcile_once()
    cancelled = handler.get_preprocessed(pre.id)
    assert cancelled.status == "cancelled"
    assert remote_job_id in target.terminated


def test_training_cancel_terminate_failure_retries(
    test_state, tmp_path, fake_services
) -> None:
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target

    dataset = _uploaded_dataset(test_state, tmp_path)
    assert dataset is not None
    pre = handler.create_preprocessing(
        dataset_id=dataset.id,
        resolution_buckets="768x448x89",
        with_audio=False,
        auto_caption=False,
        captioner_type="gemini_flash",
    )
    runner.reconcile_once()  # submit process
    runner.reconcile_once()  # ready

    job = handler.start_training(
        preprocessed_id=pre.id,
        name="run1",
        config=TrainingConfig(),
        provider="runpod",
    )
    runner.reconcile_once()  # -> running
    running = handler.get_training_state().items[0]
    assert running.status == "running"
    assert running.target is not None
    target.status_by_job[running.target.remote_job_id] = [
        RemoteCommandStatus(state="running")
    ]

    handler.request_cancel_training(job.id)
    # Terminate fails: must stay `running` (cancel_requested still set) and
    # NOT mark `cancelled` while the remote job keeps billing the GPU.
    target.raise_on_terminate = TrainerTargetError("pod gone", retryable=True)
    runner.reconcile_once()
    still_running = handler.get_training_state().items[0]
    assert still_running.status == "running"
    assert still_running.cancel_requested is True
    assert target.terminated == []

    # Next tick succeeds -> cancelled.
    target.raise_on_terminate = None
    runner.reconcile_once()
    cancelled = handler.get_training_state().items[0]
    assert cancelled.status == "cancelled"


def test_training_cancel_without_remote_job_id_finalizes(
    test_state, tmp_path, fake_services
) -> None:
    job, _dataset_id, _remote_job_id = _running_job(
        test_state, tmp_path, fake_services
    )
    handler = test_state.lora_training
    running = handler._training.items[0]
    assert running.target is not None
    running.target = running.target.model_copy(
        update={"remote_job_id": None}
    )
    handler.request_cancel_training(job.id)

    test_state.lora_training_runner.reconcile_once()

    cancelled = handler.get_training_state().items[0]
    assert cancelled.status == "cancelled"
    assert fake_services.trainer_target.terminated == []


def _running_job(test_state, tmp_path, fake_services):
    """Drive upload -> preprocess -> start training -> reach a `running` job.

    Returns (job, dataset_id, remote_job_id). Keeps the remote command
    "running" so the reconciler doesn't complete it during cancel/retry tests.
    """
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target

    dataset = _uploaded_dataset(test_state, tmp_path)
    assert dataset is not None
    pre = handler.create_preprocessing(
        dataset_id=dataset.id,
        resolution_buckets="768x448x89",
        with_audio=False,
        auto_caption=False,
        captioner_type="gemini_flash",
    )
    runner.reconcile_once()  # submit process
    runner.reconcile_once()  # ready
    job = handler.start_training(
        preprocessed_id=pre.id,
        name="run1",
        config=TrainingConfig(),
        provider="runpod",
    )
    runner.reconcile_once()  # -> running
    running = handler.get_training_state().items[0]
    assert running.status == "running"
    assert running.target is not None
    target.status_by_job[running.target.remote_job_id] = [
        RemoteCommandStatus(state="running")
    ]
    return running, dataset.id, running.target.remote_job_id


def test_training_transient_failure_surfaced_and_cleared(
    test_state, tmp_path, fake_services
) -> None:
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target

    job, dataset_id, _ = _running_job(test_state, tmp_path, fake_services)

    # Simulate a transient re-provision blip: idle-stop cleared the pod, and
    # re-provisioning fails with a retryable SSH error. This escapes the
    # reconcile function (no inner try around `_ensure_active_pod`), so the
    # outer guard must record it on the entity instead of silently swallowing.
    handler.mark_dataset_pod_stopped(dataset_id)
    target.raise_on_ensure_provisioned = TrainerTargetError(
        "ssh: connection reset", retryable=True
    )
    runner.reconcile_once()
    flagged = handler.get_training_state().items[0]
    assert flagged.status == "running"  # not failed — within retry budget
    assert flagged.consecutive_failures == 1
    assert flagged.status_detail is not None
    assert flagged.status_detail.startswith("Retrying after error:")

    # Clean tick: the blip clears -> the retry surface is reset.
    target.raise_on_ensure_provisioned = None
    runner.reconcile_once()
    recovered = handler.get_training_state().items[0]
    assert recovered.status == "running"
    assert recovered.consecutive_failures == 0
    assert recovered.status_detail is None


def test_training_transient_failure_budget_escalates_to_failed(
    test_state, tmp_path, fake_services
) -> None:
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target

    job, dataset_id, _ = _running_job(test_state, tmp_path, fake_services)
    handler.mark_dataset_pod_stopped(dataset_id)
    target.raise_on_ensure_provisioned = TrainerTargetError(
        "pod won't come back", retryable=True
    )

    # Reconcile up to the budget; the last tick should escalate to `failed`
    # rather than looping on a permanently-broken re-provision forever.
    from handlers.lora_training_handler import _TRANSIENT_FAILURE_BUDGET

    for _ in range(_TRANSIENT_FAILURE_BUDGET):
        runner.reconcile_once()

    failed = handler.get_training_state().items[0]
    assert failed.status == "failed"
    assert failed.error is not None and failed.error.startswith("Repeated failures:")
    assert failed.status_detail is None


def test_training_fatal_failure_marks_failed_immediately(
    test_state, tmp_path, fake_services
) -> None:
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target

    job, dataset_id, _ = _running_job(test_state, tmp_path, fake_services)
    handler.mark_dataset_pod_stopped(dataset_id)
    # Non-retryable target error (e.g. bad credentials) is fatal: no retry,
    # mark `failed` on the first tick with the detail surfaced.
    target.raise_on_ensure_provisioned = TrainerTargetError(
        "invalid API key", retryable=False
    )
    runner.reconcile_once()

    failed = handler.get_training_state().items[0]
    assert failed.status == "failed"
    assert failed.error == "invalid API key"
    assert failed.consecutive_failures == 0


def test_parse_setup_progress_extracts_percent_and_eta() -> None:
    from handlers.lora_training_runner import _parse_setup_progress

    # tqdm checkpoint bar -> training model, percent, ETA from "<03:01".
    phase, pct, eta = _parse_setup_progress(
        "Downloading ltx-2.3-22b-dev.safetensors: 47%|████ | 21.0G/44.0G [02:41<03:01, 130MB/s]"
    )
    assert phase == "Downloading training model"
    assert pct == 47
    assert eta == 3 * 60 + 1
    assert _parse_setup_progress(
        "Downloading gemma-3-text_encoder.safetensors: 12%|█ | 1.0G/8.0G [00:10<01:20]"
    ) == ("Downloading text encoder", 12, 80)

    # Our step markers -> clean phase, no bar.
    assert _parse_setup_progress("[provision] cloning trainer") == ("Cloning trainer", None, None)
    assert _parse_setup_progress("[provision] installing dependencies (uv sync)") == (
        "Installing dependencies",
        None,
        None,
    )
    # Plain line -> generic phase, no percent.
    phase3, pct3, eta3 = _parse_setup_progress("Setting up pod…")
    assert pct3 is None and eta3 is None

    # uv dependency lines surface the package name so the card ticks visibly.
    assert _parse_setup_progress("Downloaded torchvision") == (
        "Installing dependencies — torchvision",
        None,
        None,
    )
    # A git-clone progress bar is labeled as cloning, not a model download.
    assert _parse_setup_progress("Updating files:  80% (183/228)")[0] == "Cloning trainer"


def test_parse_preprocess_progress_reads_tail() -> None:
    from handlers.lora_training_runner import _parse_preprocess_progress

    # A tqdm bar with percent + ETA -> Caching latents with both parsed.
    parsed = _parse_preprocess_progress(
        ["Loading checkpoint…", "Precomputing latents:  37%|██ | 10/27 [00:20<00:34, 2s/it]"]
    )
    assert parsed == ("Caching latents", 37, 34)

    # An `n/m` count with no explicit percent -> percent derived from the count.
    parsed2 = _parse_preprocess_progress(["Encoding clip 9/27"])
    assert parsed2 is not None and parsed2[0] == "Caching latents" and parsed2[1] == 33

    # Model-load lines surface as a distinct phase (no count yet).
    assert _parse_preprocess_progress(["Loading VAE weights"]) == ("Loading models", None, None)

    # Newest informative line wins; blank lines are skipped.
    assert _parse_preprocess_progress(["captioning clips 5/10", ""]) == (
        "Captioning clips",
        50,
        None,
    )

    # Nothing parseable yet -> None (caller leaves the current detail in place).
    assert _parse_preprocess_progress(["", "   "]) is None


def test_training_eta_from_step_rate(test_state) -> None:
    import time

    runner = test_state.lora_training_runner
    # First sighting of a job -> no estimate yet, just records the sample.
    assert runner._estimate_training_eta("fresh", 10, 2000) is None
    # Seed a sample 50s ago at step 100; now at step 150 -> ~1 step/s, 1850
    # steps remaining -> ~1850s (allow slack for the call's own elapsed time).
    runner._train_rate_samples["job-x"] = (100, time.monotonic() - 50.0, 0.0)
    eta = runner._estimate_training_eta("job-x", 150, 2000)
    assert eta is not None and 1750 <= eta <= 1900
    # A poll where the step didn't advance must not produce a spike -> None.
    runner._train_rate_samples["stuck"] = (200, time.monotonic(), 0.5)
    assert runner._estimate_training_eta("stuck", 200, 2000) is None
    # Unknown total -> can't estimate.
    runner._train_rate_samples["notot"] = (10, time.monotonic() - 10.0, 0.0)
    assert runner._estimate_training_eta("notot", 20, None) is None


def test_latest_checkpoint_step_picks_highest() -> None:
    from handlers.lora_training_runner import _latest_checkpoint_step

    # The final save (highest step) wins, regardless of log order.
    log = [
        "💾 Lora weights for step 1750 saved in checkpoints/lora_weights_step_01750.safetensors",
        "💾 Lora weights for step 2000 saved in checkpoints/lora_weights_step_02000.safetensors",
        "Training complete.",
    ]
    assert _latest_checkpoint_step(log) == 2000
    # No checkpoint lines -> None (caller falls back to the configured step).
    assert _latest_checkpoint_step(["Loss: 0.1", ""]) is None


def test_format_eta_is_human_readable() -> None:
    from handlers.lora_training_runner import _format_eta

    assert _format_eta(None) == "—"
    assert _format_eta(-1) == "—"
    assert _format_eta(45) == "45s"
    assert _format_eta(90) == "1m 30s"
    assert _format_eta(3700) == "1h 1m"


def test_latest_meaningful_line_skips_blanks_and_truncates() -> None:
    from handlers.lora_training_runner import _latest_meaningful_line

    # Blank lines are skipped; the last non-empty line wins.
    assert _latest_meaningful_line(["", "  ", "Loading model shards"]) == "Loading model shards"
    assert _latest_meaningful_line(["first", "", "second"]) == "second"
    # All blank -> empty (caller skips logging).
    assert _latest_meaningful_line(["", "   "]) == ""
    # Long lines are truncated with an ellipsis so the log stays readable.
    out = _latest_meaningful_line(["x" * 500])
    assert len(out) == 200 and out.endswith("…")


def test_remote_failure_detail_always_includes_exit_code() -> None:
    from handlers.lora_training_runner import LoraTrainingRunner

    # A SIGKILL (137) hard-kill leaves an innocuous final log line and no
    # traceback — the exit code is the only signal that it wasn't a trainer
    # bug, so it must appear even when a tail is present.
    detail = LoraTrainingRunner._remote_failure_detail(
        ["Rank 0/1: processing 24 of 24 items", "`torch_dtype` is deprecated! Use `dtype` instead!"],
        exit_code=137,
        error="Remote command exited with code 137",
        kind="Preprocessing",
    )
    assert "[exit 137" in detail
    assert "SIGKILL" in detail
    assert "torch_dtype" in detail  # tail still surfaced

    # A segfault (139) is annotated as a native/CUDA crash.
    detail = LoraTrainingRunner._remote_failure_detail(
        ["loading model"], exit_code=139, error=None, kind="Training"
    )
    assert "[exit 139" in detail and "SIGSEGV" in detail

    # A normal Python exception (exit 1) gets a plain code, no alarmist hint.
    detail = LoraTrainingRunner._remote_failure_detail(
        ["KeyError: 'caption'"], exit_code=1, error="boom", kind="Preprocessing"
    )
    assert "[exit 1]" in detail
    assert "SIGKILL" not in detail and "SIGSEGV" not in detail

    # No tail at all -> fall back to the transport error + exit code.
    detail = LoraTrainingRunner._remote_failure_detail(
        [], exit_code=137, error="Remote command exited with code 137", kind="Training"
    )
    assert "Remote command exited with code 137" in detail
    assert "[exit 137" in detail


def test_remote_failure_detail_diagnoses_missing_exit_code_with_tail() -> None:
    from handlers.lora_training_runner import LoraTrainingRunner

    # The user's reported failure: the job produced output (so it ran) but no
    # exit status was recorded — the wrapper was killed before writing `$?`
    # (system OOM killer, WSL distro shutdown, spot preemption). The tail ends
    # on an innocuous `torch_dtype` deprecation; the detail must say the
    # process was killed mid-run rather than letting the deprecation read as
    # the cause.
    detail = LoraTrainingRunner._remote_failure_detail(
        [
            "INFO Detected column 'caption' → caption",
            "INFO Rank 0/1: processing 24 of 24 items",
            "`torch_dtype` is deprecated! Use `dtype` instead!",
        ],
        exit_code=None,
        error="WSL job is no longer running but wrote no exit status…",
        kind="Preprocessing",
    )
    assert "torch_dtype" in detail  # tail still surfaced
    assert "no exit code recorded" in detail
    assert "killed mid-run" in detail
    # No bare "[exit …]" since there was no exit code.
    assert "[exit " not in detail

    # No tail AND no exit code -> fall back to the transport error, no hint.
    detail = LoraTrainingRunner._remote_failure_detail(
        [], exit_code=None, error="transport blip", kind="Training"
    )
    assert detail == "transport blip"


def test_remote_failure_detail_redacts_secrets() -> None:
    from handlers.lora_training_runner import LoraTrainingRunner

    detail = LoraTrainingRunner._remote_failure_detail(
        [
            "downloading checkpoint",
            "HF_TOKEN=hf_private_value",
            "Authorization: Bearer remote-bearer-value",
        ],
        exit_code=1,
        error=None,
        kind="Provisioning",
    )

    assert "hf_private_value" not in detail
    assert "remote-bearer-value" not in detail
    assert detail.count("[REDACTED]") == 2
    assert "[exit 1]" in detail


def test_step_and_loss_regexes_parse_trainer_log_lines() -> None:
    from handlers.lora_training_runner import _LOSS_RE, _STEP_RE

    # Trainer emits a "step N/total" marker we parse for the card + log.
    m = _STEP_RE.search("{'loss': 0.234, 'learning_rate': 1e-4, 'step': 120/2000}")
    assert m is not None
    assert (m.group(1), m.group(2)) == ("120", "2000")
    # Loss on the same line is captured for the progress log (case-insensitive).
    assert _LOSS_RE.search("'loss': 0.2341").group(1) == "0.2341"
    # A line without a step ratio is ignored by the step parser.
    assert _STEP_RE.search("Loading model shards") is None


def _set_settings(test_state, **patch) -> None:
    from state.app_settings import AppSettingsPatch

    test_state.settings.update_settings(AppSettingsPatch.model_validate(patch))


def test_runpod_auto_provision_ignores_stale_workspace_overrides(test_state) -> None:
    # Stale remote-path overrides must not leak into the managed RunPod flow:
    # auto-provision means the app owns the layout (volume at /workspace).
    runner = test_state.lora_training_runner
    _set_settings(
        test_state,
        lora_auto_provision=True,
        lora_remote_workspace_dir="/home/user",
        lora_remote_model_path="/home/user/LTX-2/models/x.safetensors",
        lora_remote_text_encoder_path="/home/user/LTX-2/models/gemma",
    )
    settings = test_state.settings.get_settings_snapshot()
    creds = runner._credentials(settings)
    assert creds.workspace_dir == "/workspace"
    assert creds.model_path == "/workspace/models/ltx-2.3-22b-dev.safetensors"
    assert creds.text_encoder_path == "/workspace/models/gemma-text-encoder"


def test_non_managed_runpod_respects_path_overrides(test_state) -> None:
    # With auto-provision OFF (pre-baked image), the user's paths are honored.
    runner = test_state.lora_training_runner
    _set_settings(
        test_state,
        lora_auto_provision=False,
        lora_remote_workspace_dir="/opt/ltx",
        lora_remote_model_path="/opt/ltx/model.safetensors",
    )
    settings = test_state.settings.get_settings_snapshot()
    creds = runner._credentials(settings)
    assert creds.workspace_dir == "/opt/ltx"
    assert creds.model_path == "/opt/ltx/model.safetensors"


def _backdate_activity(handler, dataset_id: str) -> None:
    # White-box: push the idle clock into the past so idle-stop fires now.
    for d in handler._datasets.datasets:
        if d.id == dataset_id:
            d.last_active_at = "2000-01-01T00:00:00+00:00"


def _bind_dataset_cache(handler, dataset_id: str, volume_id: str) -> None:
    for dataset in handler._datasets.datasets:
        if dataset.id == dataset_id:
            dataset.workspace_policy = "primary_cache"
            dataset.cache_volume_id = volume_id


def test_idle_stop_releases_idle_pod_with_volume(test_state, tmp_path, fake_services) -> None:
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    dataset = _uploaded_dataset(test_state, tmp_path)
    assert dataset is not None and dataset.target is not None
    assert dataset.target.pod_id == "fake-pod-1"
    _set_settings(
        test_state,
        runpod_network_volume_id="vol-1",
        runpod_keep_model_cached=True,
        runpod_idle_stop_minutes=10,
    )
    _bind_dataset_cache(handler, dataset.id, "vol-1")
    _backdate_activity(handler, dataset.id)

    runner.reconcile_once()

    assert fake_services.trainer_target.released == 1
    refreshed = handler.get_dataset(dataset.id)
    assert refreshed is not None and refreshed.target is not None
    # Pod id cleared but the handle (provider + remote dir) is kept for re-acquire.
    assert refreshed.target.pod_id is None
    assert refreshed.status == "uploaded"


def test_idle_stop_terminates_ephemeral_pod_without_volume(test_state, tmp_path, fake_services) -> None:
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    dataset = _uploaded_dataset(test_state, tmp_path)
    _set_settings(test_state, runpod_network_volume_id="", runpod_idle_stop_minutes=10)
    _backdate_activity(handler, dataset.id)

    runner.reconcile_once()

    # No volume means the workspace is ephemeral, but an idle GPU must not bill
    # forever. Releasing it intentionally discards that workspace.
    assert fake_services.trainer_target.released == 1
    refreshed = handler.get_dataset(dataset.id)
    assert refreshed is not None and refreshed.target is not None
    assert refreshed.target.pod_id is None


def test_idle_stop_ignores_stale_volume_and_releases_when_cache_off(
    test_state, tmp_path, fake_services
) -> None:
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    dataset = _uploaded_dataset(test_state, tmp_path)
    assert dataset is not None
    _set_settings(
        test_state,
        runpod_network_volume_id="stale-vol",
        runpod_keep_model_cached=False,
        runpod_idle_stop_minutes=10,
    )
    _backdate_activity(handler, dataset.id)

    runner.reconcile_once()

    assert fake_services.trainer_target.released == 1
    refreshed = handler.get_dataset(dataset.id)
    assert refreshed is not None and refreshed.target is not None
    assert refreshed.target.pod_id is None


def test_idle_stopped_pod_is_reacquired_on_next_preprocess(test_state, tmp_path, fake_services) -> None:
    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    dataset = _uploaded_dataset(test_state, tmp_path)
    assert dataset is not None
    _set_settings(
        test_state,
        runpod_network_volume_id="vol-1",
        runpod_keep_model_cached=True,
        runpod_idle_stop_minutes=10,
    )
    _bind_dataset_cache(handler, dataset.id, "vol-1")
    _backdate_activity(handler, dataset.id)
    runner.reconcile_once()  # idle-stop releases the pod
    assert handler.get_dataset(dataset.id).target.pod_id is None

    # Starting preprocessing must transparently re-acquire a pod.
    handler.create_preprocessing(
        dataset_id=dataset.id,
        resolution_buckets="768x448x89",
        with_audio=False,
        auto_caption=False,
        captioner_type="qwen_omni",
    )
    runner.reconcile_once()

    refreshed = handler.get_dataset(dataset.id)
    assert refreshed is not None and refreshed.target is not None
    assert refreshed.target.pod_id == "fake-pod-1"


def test_cache_relocation_refuses_active_preprocess(
    test_state, tmp_path, fake_services
) -> None:
    handler = test_state.lora_training
    dataset = _uploaded_dataset(test_state, tmp_path)
    assert dataset is not None
    _bind_dataset_cache(handler, dataset.id, "vol-1")
    _set_settings(
        test_state,
        runpod_network_volume_id="vol-1",
        runpod_keep_model_cached=True,
    )
    handler.create_preprocessing(
        dataset_id=dataset.id,
        resolution_buckets="768x448x49",
        with_audio=False,
        auto_caption=False,
        captioner_type="qwen_omni",
    )

    with pytest.raises(TrainerTargetError, match="active jobs"):
        test_state.lora_training_runner.relocate_runpod_volume(
            datacenter_id="US-TX-1", size_gb=500
        )
    with pytest.raises(TrainerTargetError, match="active jobs"):
        test_state.lora_training_runner.delete_runpod_volume("vol-1")

    assert not fake_services.trainer_target.ensure_volume_calls
    assert not fake_services.trainer_target.deleted_volumes


def test_cache_relocation_refuses_remote_recovery_artifacts(
    test_state, tmp_path, fake_services
) -> None:
    handler = test_state.lora_training
    dataset = _uploaded_dataset(test_state, tmp_path)
    assert dataset is not None and dataset.remote_dataset_dir
    _bind_dataset_cache(handler, dataset.id, "vol-1")
    _set_settings(
        test_state,
        runpod_network_volume_id="vol-1",
        runpod_keep_model_cached=True,
    )

    with pytest.raises(TrainerTargetError, match="recovery artifacts"):
        test_state.lora_training_runner.relocate_runpod_volume(
            datacenter_id="US-TX-1", size_gb=500
        )

    assert not fake_services.trainer_target.ensure_volume_calls


def test_ic_lora_holdout_excluded_from_training_and_staged_for_validation(
    test_state, tmp_path, fake_services
) -> None:
    """A holdout clip is excluded from training rows, its reference staged to
    holdout/{id}.mp4, and at training start it becomes a reference-conditioned
    validation sample (so the feed covers IC-LoRA)."""
    import json

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
            # Shared reference (input). Rejected so it's not a training target.
            LoraClip(id="inp", local_path=input_path, caption="", triage="reject"),
            # Kept training pair.
            LoraClip(
                id="kept", local_path=kept_out,
                caption="A clean-shaven cat.", reference_path=input_path,
            ),
            # Held-out pair: excluded from training, used for validation.
            LoraClip(
                id="hold", local_path=holdout_out,
                caption="A held-out cat looks around.", reference_path=input_path,
                triage="holdout",
            ),
        ],
    )
    handler.request_upload(dataset.id)
    runner.reconcile_once()

    uploaded = handler.get_dataset(dataset.id)
    assert uploaded is not None and uploaded.status == "uploaded"
    local_dir, _ = target.uploaded_dirs[-1]
    rows = json.loads((Path(local_dir) / "dataset.json").read_text())
    # Only the kept pair ships as a training row; the holdout target does not.
    assert len(rows) == 1
    assert rows[0]["caption"] == "A clean-shaven cat."
    # The holdout reference video was staged under holdout/{id}.mp4 and uploaded.
    assert (Path(local_dir) / "holdout" / "hold.mp4").is_file()

    # Drive preprocess -> ready -> start training.
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
    runner.reconcile_once()  # pending -> running (registers validation_sample_refs)

    job = handler.get_training(job.id)
    assert job is not None
    # IC-LoRA drops bare prompt samples; the holdout clip is the lone ref,
    # source="holdout", with the held-out caption as the prompt.
    assert [r.source for r in job.validation_sample_refs] == ["holdout"]
    assert job.validation_sample_refs[0].prompt == "A held-out cat looks around."
    # The YAML config staged for the trainer carries the reference condition
    # pointing at the staged holdout reference video.
    remote_ref = f"{uploaded.remote_dataset_dir}/holdout/hold.mp4"
    config_files = list(
        (test_state.config.app_data_dir / "lora" / "configs" / job.id).rglob("*.yaml")
    )
    assert config_files, "training YAML was not staged"
    yaml_text = config_files[0].read_text(encoding="utf-8")
    assert remote_ref in yaml_text
    assert "samples:" in yaml_text


def test_local_captioning_oom_logs_wsl_diagnostics(
    test_state, tmp_path, fake_services, monkeypatch
) -> None:
    # A local run that dies with NO exit code is a suspected hard kill (OOM
    # killer / WSL distro shutdown / native crash). The runner probes WSL's
    # MemTotal + the OOM-killer log via `wsl_postmortem()` and surfaces it in
    # the failure detail so it self-diagnoses instead of looking like a trainer
    # bug. The standalone function shells out to wsl.exe (can't run in CI), so
    # patch it to return a canned snapshot.
    from state.lora_training_state import AutoPipelineSpec, LoraClip, PendingTraining

    import services.wsl_memory.wsl_memory as wsl_mem

    monkeypatch.setattr(
        wsl_mem,
        "wsl_postmortem",
        lambda unit=None: "MemTotal: 1610612736 kB\nKilled process 4321 (python) total-vm:...",
    )

    handler = test_state.lora_training
    runner = test_state.lora_training_runner
    target = fake_services.trainer_target
    clip_path = _make_clip_file(tmp_path)
    dataset = handler.create_dataset(
        name="ds", trigger_word="TOK",
        clips=[LoraClip(id="c0", local_path=clip_path, caption="a cat sitting.")],
    )
    spec = AutoPipelineSpec(
        resolution_buckets="768x448x49", with_audio=False, auto_caption=True,
        captioner_type="gemini_flash",
        training=PendingTraining(config=TrainingConfig(steps=500), name="run1"),
    )
    # The caption command is hard-killed mid-run (no exit code recorded).
    target.command_results["caption_videos.py"] = RemoteCommandStatus(
        state="failed", exit_code=None, error="WSL job is no longer running but wrote no exit status.",
    )
    handler.start_training_pipeline(dataset_id=dataset.id, spec=spec, provider="local")
    for _ in range(6):
        runner.reconcile_once()

    pre = handler.get_preprocessed_state().items[0]
    assert pre.status == "failed"
    assert pre.error is not None
    assert "WSL diagnostics" in pre.error
    assert "Killed process" in pre.error
    assert "MemTotal" in pre.error


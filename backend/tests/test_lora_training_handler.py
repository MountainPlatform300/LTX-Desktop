"""Unit tests for the LoRA-trainer ledger handler and validators."""

from __future__ import annotations

import pytest

from handlers.lora_training_handler import (
    LoraEntityNotFoundError,
    LoraTransitionError,
    validate_resolution_buckets,
)
from state.lora_training_state import LoraClip, TrainingConfig


def _clip(path: str = "/tmp/a.mp4", caption: str = "") -> LoraClip:
    return LoraClip(id="c1", local_path=path, caption=caption)


def test_validate_resolution_buckets_accepts_valid() -> None:
    validate_resolution_buckets("768x448x89")
    validate_resolution_buckets("960x544x49;512x512x81")


@pytest.mark.parametrize(
    "value",
    ["", "768x448", "768x448x48", "770x448x89", "768x450x89", "abc"],
)
def test_validate_resolution_buckets_rejects_invalid(value: str) -> None:
    with pytest.raises(LoraTransitionError):
        validate_resolution_buckets(value)


def test_create_and_list_dataset(test_state) -> None:
    handler = test_state.lora_training
    created = handler.create_dataset(name="My LoRA", trigger_word="MYTOK", clips=[_clip()])
    assert created.status == "draft"
    state = handler.get_datasets_state()
    assert [d.id for d in state.datasets] == [created.id]
    assert state.datasets[0].trigger_word == "MYTOK"


def test_update_dataset_rejected_after_upload(test_state) -> None:
    handler = test_state.lora_training
    created = handler.create_dataset(name="d", trigger_word=None, clips=[_clip()])
    handler.request_upload(created.id)
    # Pretend the reconciler finished the upload.
    from state.lora_training_state import TargetHandle

    handler.mark_dataset_uploaded(
        created.id,
        remote_dataset_dir="/workspace/datasets/x",
        handle=TargetHandle(provider="runpod", pod_id="p1"),
    )
    with pytest.raises(LoraTransitionError):
        handler.update_dataset(created.id, name="new", trigger_word=None, clips=None)


def test_request_upload_empty_dataset_rejected(test_state) -> None:
    handler = test_state.lora_training
    created = handler.create_dataset(name="d", trigger_word=None, clips=[])
    with pytest.raises(LoraTransitionError):
        handler.request_upload(created.id)


def test_request_upload_stamps_selected_provider(test_state) -> None:
    handler = test_state.lora_training
    created = handler.create_dataset(name="d", trigger_word=None, clips=[_clip()])

    uploaded = handler.request_upload(created.id, provider="local")

    assert uploaded.provider == "local"


def test_create_preprocessing_requires_uploaded_dataset(test_state) -> None:
    handler = test_state.lora_training
    created = handler.create_dataset(name="d", trigger_word=None, clips=[_clip()])
    with pytest.raises(LoraTransitionError):
        handler.create_preprocessing(
            dataset_id=created.id,
            resolution_buckets="768x448x89",
            with_audio=False,
            auto_caption=True,
            captioner_type="gemini_flash",
        )


def test_create_preprocessing_stamps_preset(test_state) -> None:
    # The preset is threaded onto the preprocessed dataset so the preprocess
    # stage can match the training stage's text-encoder precision.
    from state.lora_training_state import TargetHandle

    handler = test_state.lora_training
    created = handler.create_dataset(name="d", trigger_word=None, clips=[_clip()])
    handler.request_upload(created.id)
    handler.mark_dataset_uploaded(
        created.id,
        remote_dataset_dir="/workspace/datasets/x",
        handle=TargetHandle(provider="runpod", pod_id="p1"),
    )
    pre = handler.create_preprocessing(
        dataset_id=created.id,
        resolution_buckets="768x448x89",
        with_audio=False,
        auto_caption=True,
        captioner_type="gemini_flash",
        preset="low_vram",
    )
    assert pre.preset == "low_vram"
    assert pre.trainer_repo_url == test_state.state.app_settings.lora_trainer_repo_url
    assert pre.trainer_repo_ref == test_state.state.app_settings.lora_trainer_repo_ref
    handler.request_cancel_preprocessing(pre.id)
    # Default stays standard when no preset is passed.
    pre_default = handler.create_preprocessing(
        dataset_id=created.id,
        resolution_buckets="768x448x89",
        with_audio=False,
        auto_caption=True,
        captioner_type="gemini_flash",
    )
    assert pre_default.preset == "standard"


def test_ic_preprocessing_accepts_bucket_larger_than_initial_upload_envelope(
    test_state,
) -> None:
    from state.lora_training_state import TargetHandle

    handler = test_state.lora_training
    created = handler.create_dataset(
        name="ic",
        dataset_type="ic_lora",
        trigger_word=None,
        clips=[_clip()],
    )
    handler.request_upload(created.id)
    handler.mark_dataset_uploaded(
        created.id,
        remote_dataset_dir="/workspace/datasets/ic",
        handle=TargetHandle(provider="runpod", pod_id="p1"),
    )

    pre = handler.create_preprocessing(
        dataset_id=created.id,
        resolution_buckets="768x448x81",
        with_audio=False,
        auto_caption=False,
        captioner_type="gemini_flash",
    )
    assert pre.status == "pending"
    assert pre.auto_caption is False


def test_ic_one_click_pipeline_stages_requested_bucket_envelope(test_state) -> None:
    from state.lora_training_state import (
        AutoPipelineSpec,
        PendingTraining,
        TrainingConfig,
    )

    handler = test_state.lora_training
    dataset = handler.create_dataset(
        name="ic",
        dataset_type="ic_lora",
        trigger_word=None,
        clips=[_clip()],
    )
    handler.start_training_pipeline(
        dataset_id=dataset.id,
        spec=AutoPipelineSpec(
            resolution_buckets="768x448x49;512x512x81",
            auto_caption=False,
            training=PendingTraining(name="run", config=TrainingConfig()),
        ),
    )

    staged = handler.get_dataset(dataset.id)
    assert staged.ic_staged_short_side == 512
    assert staged.ic_staged_bucket_frames == 81


def test_create_preprocessing_rejects_second_active_job(test_state) -> None:
    from state.lora_training_state import TargetHandle

    handler = test_state.lora_training
    created = handler.create_dataset(name="d", trigger_word=None, clips=[_clip()])
    handler.request_upload(created.id)
    handler.mark_dataset_uploaded(
        created.id,
        remote_dataset_dir="/workspace/datasets/x",
        handle=TargetHandle(provider="runpod", pod_id="p1"),
    )
    handler.create_preprocessing(
        dataset_id=created.id,
        resolution_buckets="768x448x89",
        with_audio=False,
        auto_caption=False,
        captioner_type="qwen_omni",
    )

    with pytest.raises(LoraTransitionError, match="active preprocessing"):
        handler.create_preprocessing(
            dataset_id=created.id,
            resolution_buckets="768x448x89",
            with_audio=False,
            auto_caption=False,
            captioner_type="qwen_omni",
        )


def test_start_training_requires_ready_preprocessed(test_state) -> None:
    handler = test_state.lora_training
    from state.lora_training_state import TargetHandle, TrainingConfig

    created = handler.create_dataset(name="d", trigger_word=None, clips=[_clip()])
    handler.request_upload(created.id)
    handler.mark_dataset_uploaded(
        created.id,
        remote_dataset_dir="/workspace/datasets/x",
        handle=TargetHandle(provider="runpod", pod_id="p1"),
    )
    pre = handler.create_preprocessing(
        dataset_id=created.id,
        resolution_buckets="768x448x89",
        with_audio=False,
        auto_caption=True,
        captioner_type="gemini_flash",
    )
    with pytest.raises(LoraTransitionError):
        handler.start_training(
            preprocessed_id=pre.id,
            name="run1",
            config=TrainingConfig(),
            provider="runpod",
        )


def test_missing_entities_raise_not_found(test_state) -> None:
    handler = test_state.lora_training
    with pytest.raises(LoraEntityNotFoundError):
        handler.request_upload("nope")
    with pytest.raises(LoraEntityNotFoundError):
        handler.request_cancel_training("nope")


def test_persistence_round_trip(test_state) -> None:
    handler = test_state.lora_training
    created = handler.create_dataset(name="persisted", trigger_word="T", clips=[_clip()])
    # Reload from disk into a fresh handler view.
    handler.load_state()
    state = handler.get_datasets_state()
    assert any(d.id == created.id and d.name == "persisted" for d in state.datasets)


@pytest.mark.parametrize(
    "file_attribute",
    [
        "_datasets_file",
        "_preprocessed_file",
        "_training_file",
        "_profiles_file",
        "_clip_jobs_file",
        "_derivation_file",
    ],
)
def test_corrupt_lora_ledger_is_backed_up_and_does_not_break_load(
    test_state, file_attribute: str
) -> None:
    handler = test_state.lora_training
    ledger = getattr(handler, file_attribute)
    ledger.write_text("{not valid json", encoding="utf-8")

    handler.load_state()

    backups = list(ledger.parent.glob(f"{ledger.stem}.corrupt-*.json"))
    assert backups
    assert backups[-1].read_text(encoding="utf-8") == "{not valid json"


def _ready_preprocessed(handler) -> str:
    from state.lora_training_state import TargetHandle

    created = handler.create_dataset(name="d", trigger_word=None, clips=[_clip()])
    handler.request_upload(created.id)
    handler.mark_dataset_uploaded(
        created.id,
        remote_dataset_dir="/workspace/datasets/x",
        handle=TargetHandle(provider="runpod", pod_id="p1"),
    )
    pre = handler.create_preprocessing(
        dataset_id=created.id,
        resolution_buckets="768x448x89",
        with_audio=True,
        auto_caption=True,
        captioner_type="gemini_flash",
    )
    handler.mark_preprocess_ready(pre.id, remote_precomputed_dir="/workspace/.precomputed")
    return pre.id


def test_start_training_rejects_second_active_job(test_state) -> None:
    from state.lora_training_state import TrainingConfig

    handler = test_state.lora_training
    pre_id = _ready_preprocessed(handler)
    handler.start_training(
        preprocessed_id=pre_id,
        name="first",
        config=TrainingConfig(),
        provider="runpod",
    )

    with pytest.raises(LoraTransitionError, match="active training"):
        handler.start_training(
            preprocessed_id=pre_id,
            name="second",
            config=TrainingConfig(),
            provider="runpod",
        )


def test_start_training_rejects_provider_mismatch(test_state) -> None:
    from state.lora_training_state import TrainingConfig

    handler = test_state.lora_training
    pre_id = _ready_preprocessed(handler)

    with pytest.raises(LoraTransitionError, match="workspace provider"):
        handler.start_training(
            preprocessed_id=pre_id,
            name="wrong-provider",
            config=TrainingConfig(),
            provider="local",
        )


def test_start_training_rejects_preset_mismatch(test_state) -> None:
    from state.lora_training_state import TrainingConfig

    handler = test_state.lora_training
    pre_id = _ready_preprocessed(handler)

    with pytest.raises(LoraTransitionError, match="does not match preprocessing"):
        handler.start_training(
            preprocessed_id=pre_id,
            name="wrong-preset",
            config=TrainingConfig(preset="low_vram"),
            provider="runpod",
        )


def test_delete_dataset_rejects_active_training(test_state) -> None:
    from state.lora_training_state import TrainingConfig

    handler = test_state.lora_training
    pre_id = _ready_preprocessed(handler)
    preprocessed = handler.get_preprocessed(pre_id)
    assert preprocessed is not None
    handler.start_training(
        preprocessed_id=pre_id,
        name="run",
        config=TrainingConfig(),
        provider="runpod",
    )

    with pytest.raises(LoraTransitionError, match="active training"):
        handler.delete_dataset(preprocessed.dataset_id)

    assert handler.get_dataset(preprocessed.dataset_id) is not None


def test_profile_crud_and_seed(test_state) -> None:
    handler = test_state.lora_training
    handler.load_state()
    profiles = handler.get_profiles_state().profiles
    by_name = {profile.name: profile for profile in profiles}
    assert set(by_name) == {"Standard LoRA", "Low VRAM", "IC-LoRA"}
    assert by_name["Standard LoRA"].config == TrainingConfig()
    assert by_name["Low VRAM"].config == TrainingConfig(
        preset="low_vram", rank=16, alpha=16
    )
    ic_lora = by_name["IC-LoRA"]
    assert ic_lora.config.rank == 32
    assert ic_lora.config.learning_rate == 2e-4
    assert ic_lora.config.steps == 3000
    assert ic_lora.config.first_frame_conditioning_p == 0.2
    assert all(profile.builtin for profile in profiles)

    with pytest.raises(LoraTransitionError, match="read-only"):
        handler.update_profile(profiles[0].id, name="Changed")
    with pytest.raises(LoraTransitionError, match="cannot be deleted"):
        handler.delete_profile(profiles[0].id)

    created = handler.create_profile(
        name="Custom",
        config=TrainingConfig(rank=128),
        description="My settings",
        dataset_types=["standard"],
    )
    assert created.config.rank == 128
    assert created.builtin is False
    assert created.description == "My settings"
    assert created.dataset_types == ["standard"]

    updated = handler.update_profile(created.id, name="Custom2", config=TrainingConfig(rank=8))
    assert updated.name == "Custom2"
    assert updated.config.rank == 8

    handler.delete_profile(created.id)
    assert all(p.id != created.id for p in handler.get_profiles_state().profiles)
    with pytest.raises(LoraEntityNotFoundError):
        handler.get_profile(created.id)


def test_legacy_profiles_migrate_to_official_set_and_preserve_customizations(
    test_state,
) -> None:
    from state.lora_training_state import (
        BUILTIN_LOW_VRAM_ID,
        BUILTIN_STANDARD_ID,
        LoraTrainingProfile,
        LoraTrainingProfilesState,
        TrainingConfig,
    )

    handler = test_state.lora_training
    legacy = LoraTrainingProfilesState(
        schema_version=4,
        profiles=[
            LoraTrainingProfile(
                id=BUILTIN_STANDARD_ID,
                name="My edited standard",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
                config=TrainingConfig(rank=48, alpha=48),
                builtin=True,
            ),
            LoraTrainingProfile(
                id=BUILTIN_LOW_VRAM_ID,
                name="Low VRAM",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
                config=TrainingConfig(preset="low_vram", rank=16, alpha=16),
                builtin=True,
            ),
        ],
    )
    handler._profiles_file.write_text(legacy.model_dump_json())  # pyright: ignore[reportPrivateUsage]
    handler.load_state()
    profiles = handler.get_profiles_state().profiles
    assert handler.get_profiles_state().schema_version == 5
    assert {p.name for p in profiles if p.builtin} == {
        "Standard LoRA",
        "Low VRAM",
        "IC-LoRA",
    }
    preserved = next(p for p in profiles if p.name == "My edited standard")
    assert preserved.builtin is False
    assert preserved.config.rank == 48


def test_start_training_snapshots_config_and_audio(test_state) -> None:
    handler = test_state.lora_training
    from state.lora_training_state import TrainingConfig

    pre_id = _ready_preprocessed(handler)
    # Profile config has with_audio False; the handler must override it from
    # the preprocessed item (which was created with_audio=True).
    config = TrainingConfig(rank=64, steps=4321, with_audio=False)
    job = handler.start_training(
        preprocessed_id=pre_id, name="run", config=config, provider="runpod"
    )
    assert job.config.rank == 64
    assert job.config.steps == 4321
    assert job.config.with_audio is True
    assert job.total_steps == 4321
    assert job.trainer_repo_url is not None
    assert job.trainer_repo_ref is not None
    # Snapshot is a copy: mutating the source config doesn't affect the job.
    config.rank = 1
    assert handler.get_training_state().items[0].config.rank == 64


def _ready_preprocessed_with_trigger(handler, trigger: str) -> str:
    from state.lora_training_state import TargetHandle

    created = handler.create_dataset(name="d", trigger_word=trigger, clips=[_clip()])
    handler.request_upload(created.id)
    handler.mark_dataset_uploaded(
        created.id,
        remote_dataset_dir="/workspace/datasets/x",
        handle=TargetHandle(provider="runpod", pod_id="p1"),
    )
    pre = handler.create_preprocessing(
        dataset_id=created.id,
        resolution_buckets="768x448x89",
        with_audio=False,
        auto_caption=True,
        captioner_type="gemini_flash",
    )
    handler.mark_preprocess_ready(pre.id, remote_precomputed_dir="/workspace/.precomputed")
    return pre.id


def test_start_training_snapshots_dataset_trigger_word(test_state) -> None:
    handler = test_state.lora_training
    from state.lora_training_state import TrainingConfig

    pre_id = _ready_preprocessed_with_trigger(handler, "MYTOK")
    # Config carries no trigger word; the handler must snapshot the dataset's
    # trigger word onto the run (mirroring `with_audio`) so the Run summary and
    # the trained LoRA's registry entry record it.
    config = TrainingConfig(steps=10, with_audio=False)
    job = handler.start_training(
        preprocessed_id=pre_id, name="run", config=config, provider="runpod"
    )
    assert job.config.trigger_word == "MYTOK"


def test_start_training_trigger_word_override_wins(test_state) -> None:
    handler = test_state.lora_training
    from state.lora_training_state import TrainingConfig

    pre_id = _ready_preprocessed_with_trigger(handler, "MYTOK")
    # An explicit per-run override (set from `triggerWordOverride` by the route)
    # must win over the dataset's trigger word.
    config = TrainingConfig(steps=10, with_audio=False, trigger_word="OVERRIDE")
    job = handler.start_training(
        preprocessed_id=pre_id, name="run", config=config, provider="runpod"
    )
    assert job.config.trigger_word == "OVERRIDE"


def test_start_training_pipeline_stamps_trigger_word_on_dataset(test_state) -> None:
    handler = test_state.lora_training
    from state.lora_training_state import (
        AutoPipelineSpec,
        PendingTraining,
        TrainingConfig,
    )

    created = handler.create_dataset(name="d", trigger_word=None, clips=[_clip()])
    spec = AutoPipelineSpec(
        resolution_buckets="768x448x49",
        with_audio=False,
        auto_caption=True,
        captioner_type="gemini_flash",
        training=PendingTraining(
            config=TrainingConfig(steps=10, trigger_word="MYTOK"),
            name="run",
        ),
    )
    handler.start_training_pipeline(dataset_id=created.id, spec=spec, provider="runpod")
    dataset = handler.get_dataset(created.id)
    assert dataset is not None
    # Preprocessing reads `dataset.trigger_word` to inject the token into
    # captions, so the pipeline must stamp the run's trigger word onto the
    # dataset before preprocess starts.
    assert dataset.trigger_word == "MYTOK"


# ----------------------------------------------------------------
# Upload cancellation
# ----------------------------------------------------------------


def test_request_cancel_upload_finalizes_when_no_pod(test_state) -> None:
    """A cancel before the pod is acquired has nothing to reclaim, so it
    finalizes to `cancelled` immediately (no `cancel_requested` limbo)."""
    from state.lora_training_state import TargetHandle

    handler = test_state.lora_training
    created = handler.create_dataset(name="d", trigger_word=None, clips=[_clip()])
    handler.request_upload(created.id)
    assert handler.get_dataset(created.id).status == "uploading"

    cancelled = handler.request_cancel_upload(created.id)
    assert cancelled.status == "cancelled"
    assert cancelled.cancel_requested is False
    # No pod was ever acquired.
    assert handler.get_dataset(created.id).target is None


def test_request_cancel_upload_sets_flag_when_pod_acquired(test_state) -> None:
    """A cancel after the pod is acquired can't finalize synchronously (the
    runner may be mid-blocking-upload), so it sets `cancel_requested` and leaves
    the reconciler to release the pod + flip to `cancelled`."""
    from state.lora_training_state import TargetHandle

    handler = test_state.lora_training
    created = handler.create_dataset(name="d", trigger_word=None, clips=[_clip()])
    handler.request_upload(created.id)
    handler.set_dataset_target(
        created.id, TargetHandle(provider="runpod", pod_id="p1")
    )

    cancelling = handler.request_cancel_upload(created.id)
    # Still `uploading` so it stays in the upload reconcile list until the
    # runner releases the pod; otherwise we'd orphan a billing pod.
    assert cancelling.status == "uploading"
    assert cancelling.cancel_requested is True


def test_request_cancel_upload_rejected_for_non_uploading(test_state) -> None:
    handler = test_state.lora_training
    created = handler.create_dataset(name="d", trigger_word=None, clips=[_clip()])
    # Draft — not uploading.
    with pytest.raises(LoraTransitionError):
        handler.request_cancel_upload(created.id)


def test_mark_dataset_upload_cancelled_clears_pod(test_state) -> None:
    from state.lora_training_state import TargetHandle

    handler = test_state.lora_training
    created = handler.create_dataset(name="d", trigger_word=None, clips=[_clip()])
    handler.request_upload(created.id)
    handler.set_dataset_target(
        created.id, TargetHandle(provider="runpod", pod_id="p1")
    )
    handler.request_cancel_upload(created.id)

    handler.mark_dataset_upload_cancelled(created.id)
    done = handler.get_dataset(created.id)
    assert done.status == "cancelled"
    assert done.cancel_requested is False
    assert done.error is None
    # The released pod id is cleared so idle-stop/delete don't treat it as live.
    assert done.target is not None and done.target.pod_id is None


def test_request_upload_allowed_from_cancelled(test_state) -> None:
    """After cancelling, the user can re-request upload (retry) — `cancelled`
    is a valid entry status for upload, and the cancel flag must reset."""
    handler = test_state.lora_training
    created = handler.create_dataset(name="d", trigger_word=None, clips=[_clip()])
    handler.request_upload(created.id)
    handler.request_cancel_upload(created.id)
    assert handler.get_dataset(created.id).status == "cancelled"

    uploading = handler.request_upload(created.id)
    assert uploading.status == "uploading"
    assert uploading.cancel_requested is False


# ----------------------------------------------------------------
# Rename (any status)
# ----------------------------------------------------------------


def test_rename_dataset_at_any_status(test_state) -> None:
    handler = test_state.lora_training
    created = handler.create_dataset(name="d", trigger_word=None, clips=[_clip()])
    # Move into uploading — rename must still work.
    handler.request_upload(created.id)
    assert handler.get_dataset(created.id).status == "uploading"

    renamed = handler.rename_dataset(created.id, "  Zeev  ")
    assert renamed.name == "Zeev"
    assert handler.get_dataset(created.id).name == "Zeev"


def test_rename_dataset_rejects_empty(test_state) -> None:
    handler = test_state.lora_training
    created = handler.create_dataset(name="d", trigger_word=None, clips=[_clip()])
    with pytest.raises(LoraTransitionError):
        handler.rename_dataset(created.id, "   ")


def test_rename_dataset_missing_raises(test_state) -> None:
    handler = test_state.lora_training
    with pytest.raises(LoraEntityNotFoundError):
        handler.rename_dataset("nope", "x")


# ----------------------------------------------------------------
# Folders
# ----------------------------------------------------------------


def test_create_folder_nests_under_parent(test_state) -> None:
    handler = test_state.lora_training
    root = handler.create_folder("People", None)
    sub = handler.create_folder("IC set", root.id)
    assert sub.parent_id == root.id


def test_create_folder_rejects_empty_name(test_state) -> None:
    handler = test_state.lora_training
    with pytest.raises(LoraTransitionError):
        handler.create_folder("  ", None)


def test_create_folder_missing_parent_raises(test_state) -> None:
    handler = test_state.lora_training
    with pytest.raises(LoraEntityNotFoundError):
        handler.create_folder("x", "ghost")


def test_rename_folder(test_state) -> None:
    handler = test_state.lora_training
    folder = handler.create_folder("A", None)
    renamed = handler.rename_folder(folder.id, "B")
    assert renamed.name == "B"


def test_move_folder_rejects_cycle_self(test_state) -> None:
    handler = test_state.lora_training
    folder = handler.create_folder("A", None)
    with pytest.raises(LoraTransitionError):
        handler.move_folder(folder.id, folder.id)


def test_move_folder_rejects_cycle_descendant(test_state) -> None:
    handler = test_state.lora_training
    root = handler.create_folder("Root", None)
    child = handler.create_folder("Child", root.id)
    grandchild = handler.create_folder("Grand", child.id)
    # Moving `root` under `grandchild` would create a cycle.
    with pytest.raises(LoraTransitionError):
        handler.move_folder(root.id, grandchild.id)


def test_move_folder_to_root(test_state) -> None:
    handler = test_state.lora_training
    root = handler.create_folder("Root", None)
    child = handler.create_folder("Child", root.id)
    moved = handler.move_folder(child.id, None)
    assert moved.parent_id is None


def test_delete_folder_non_recursive_moves_contents_up(test_state) -> None:
    handler = test_state.lora_training
    root = handler.create_folder("Root", None)
    inner = handler.create_folder("Inner", root.id)
    ds = handler.create_dataset(name="d", trigger_word=None, clips=[_clip()])
    handler.move_dataset(ds.id, inner.id)
    sub = handler.create_folder("Sub", inner.id)

    handler.delete_folder(inner.id, recursive=False)

    # Folder gone; its dataset + subfolder reparented to `root`.
    folders = [f.id for f in handler._datasets.folders]
    assert inner.id not in folders
    assert root.id in folders and sub.id in folders
    moved_sub = next(f for f in handler._datasets.folders if f.id == sub.id)
    assert moved_sub.parent_id == root.id
    assert handler.get_dataset(ds.id).folder_id == root.id


def test_delete_folder_recursive_deletes_contents(test_state) -> None:
    handler = test_state.lora_training
    root = handler.create_folder("Root", None)
    inner = handler.create_folder("Inner", root.id)
    ds = handler.create_dataset(name="d", trigger_word=None, clips=[_clip()])
    handler.move_dataset(ds.id, inner.id)
    sub = handler.create_folder("Sub", inner.id)

    handler.delete_folder(inner.id, recursive=True)

    folders = [f.id for f in handler._datasets.folders]
    assert inner.id not in folders and sub.id not in folders
    # The dataset inside the subtree is deleted too.
    assert handler.get_dataset(ds.id) is None
    # Root (sibling of the deleted subtree) survives.
    assert root.id in folders


def test_delete_folder_top_level_non_recursive_moves_to_root(test_state) -> None:
    handler = test_state.lora_training
    top = handler.create_folder("Top", None)
    ds = handler.create_dataset(name="d", trigger_word=None, clips=[_clip()])
    handler.move_dataset(ds.id, top.id)

    handler.delete_folder(top.id, recursive=False)

    assert top.id not in [f.id for f in handler._datasets.folders]
    # Contained dataset moved to root.
    assert handler.get_dataset(ds.id).folder_id is None


def test_move_dataset_into_folder(test_state) -> None:
    handler = test_state.lora_training
    folder = handler.create_folder("F", None)
    ds = handler.create_dataset(name="d", trigger_word=None, clips=[_clip()])
    moved = handler.move_dataset(ds.id, folder.id)
    assert moved.folder_id == folder.id


def test_move_dataset_to_root(test_state) -> None:
    handler = test_state.lora_training
    folder = handler.create_folder("F", None)
    ds = handler.create_dataset(name="d", trigger_word=None, clips=[_clip()])
    handler.move_dataset(ds.id, folder.id)
    moved = handler.move_dataset(ds.id, None)
    assert moved.folder_id is None


def test_move_dataset_missing_folder_raises(test_state) -> None:
    handler = test_state.lora_training
    ds = handler.create_dataset(name="d", trigger_word=None, clips=[_clip()])
    with pytest.raises(LoraEntityNotFoundError):
        handler.move_dataset(ds.id, "ghost")


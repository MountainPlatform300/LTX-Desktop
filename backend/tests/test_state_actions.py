"""State action and invariant tests for AppState."""

from __future__ import annotations

import pytest

from runtime_config.model_download_specs import (
    DEPTH_PROCESSOR_CP_ID,
    get_latest_ltx_model_id,
    get_ltx_model_spec,
    resolve_model_path,
)
from state.app_settings import UpdateSettingsRequest
from state.app_state_types import CpuSlot, GpuSlot, ICLoraState, RetakePipelineState, VideoPipelineState


def _current_model_spec():
    return get_ltx_model_spec(get_latest_ltx_model_id())


def test_start_generation_requires_gpu(test_state):
    with pytest.raises(RuntimeError, match="No active GPU pipeline"):
        test_state.generation.start_generation("gen-1")


def test_generation_mutex_prevents_second_start(test_state, create_fake_model_files):
    create_fake_model_files()
    test_state.pipelines.load_gpu_pipeline("fast")
    test_state.generation.start_generation("gen-1")

    with pytest.raises(RuntimeError, match="Generation already in progress"):
        test_state.generation.start_generation("gen-2")


def test_download_terminal_state_is_sticky_until_next_session(test_state):
    session_id = test_state.downloads.start_download({"ltx-2.3-22b-distilled"})
    test_state.downloads.start_file("ltx-2.3-22b-distilled", "ltx-2.3-22b-distilled.safetensors")
    test_state.downloads.finish_download()

    progress = test_state.downloads.get_download_progress(session_id)
    assert progress.status == "complete"


def test_handler_attributes_are_wired(test_state):
    assert test_state.settings is not None
    assert test_state.models is not None
    assert test_state.downloads is not None
    assert test_state.text is not None
    assert test_state.pipelines is not None
    assert test_state.generation is not None
    assert test_state.video_generation is not None
    assert test_state.image_generation is not None
    assert test_state.health is not None
    assert test_state.suggest_gap_prompt is not None
    assert test_state.retake is not None
    assert test_state.ic_lora is not None


def test_rlock_allows_nested_handler_calls(test_state):
    test_state.settings.update_settings(UpdateSettingsRequest(useTorchCompile=True))
    assert test_state.state.app_settings.use_torch_compile is True


def test_mps_skips_torch_compile(test_state, fake_services, create_fake_model_files):
    create_fake_model_files()
    test_state.state.app_settings.use_torch_compile = True
    test_state.pipelines._runtime_device = "mps"  # noqa: SLF001 - explicit platform behavior assertion

    pipeline_state = test_state.pipelines.load_gpu_pipeline("fast")
    assert fake_services.fast_video_pipeline.compile_calls == 0
    assert pipeline_state.is_compiled is False


def test_retake_pipeline_eviction(test_state, create_fake_model_files):
    create_fake_model_files()
    test_state.pipelines.load_gpu_pipeline("fast")

    retake_state = test_state.pipelines.load_retake_pipeline(distilled=True)
    assert isinstance(test_state.state.gpu_slot, GpuSlot)
    assert isinstance(test_state.state.gpu_slot.active_pipeline, RetakePipelineState)
    assert test_state.state.gpu_slot.active_pipeline is retake_state

    test_state.pipelines.load_gpu_pipeline("fast")
    assert isinstance(test_state.state.gpu_slot.active_pipeline, VideoPipelineState)


def test_ic_lora_load_includes_depth_resources(test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files):
    create_fake_model_files()
    create_fake_ic_lora_files()
    model_spec = _current_model_spec()
    lora_path = str(resolve_model_path(test_state.config.default_models_dir, model_spec.ic_loras_spec.canny_cp))
    depth_path = str(resolve_model_path(test_state.config.default_models_dir, DEPTH_PROCESSOR_CP_ID))

    ic_state = test_state.pipelines.load_ic_lora(lora_path, depth_path)

    assert isinstance(ic_state, ICLoraState)
    assert ic_state.pipeline is fake_services.ic_lora_pipeline
    assert ic_state.depth_pipeline is fake_services.depth_processor_pipeline
    assert ic_state.lora_path == lora_path
    assert ic_state.depth_model_path == depth_path


def test_ic_lora_unload_clears_preprocessing_resources(test_state, create_fake_model_files, create_fake_ic_lora_files):
    create_fake_model_files()
    create_fake_ic_lora_files()
    model_spec = _current_model_spec()
    lora_path = str(resolve_model_path(test_state.config.default_models_dir, model_spec.ic_loras_spec.canny_cp))
    depth_path = str(resolve_model_path(test_state.config.default_models_dir, DEPTH_PROCESSOR_CP_ID))
    test_state.pipelines.load_ic_lora(lora_path, depth_path)

    assert isinstance(test_state.state.gpu_slot, GpuSlot)
    assert isinstance(test_state.state.gpu_slot.active_pipeline, ICLoraState)

    test_state.pipelines.unload_gpu_pipeline()

    assert test_state.state.gpu_slot is None


def test_standard_lora_swap_rebuilds_gpu_slot(test_state, fake_services, create_fake_model_files):
    """A standard LoRA is part of the fast-pipeline cache key: switching adapters
    rebuilds the GpuSlot instead of silently reusing the base-model pipeline."""
    create_fake_model_files()
    fake = fake_services.fast_video_pipeline
    create_calls_after = lambda: fake.create_calls  # noqa: E731

    base = test_state.pipelines.load_gpu_pipeline("fast")
    assert base.lora_path is None
    assert fake.last_lora_path is None
    creates_after_base = create_calls_after()

    # Same lora_path (None) -> cache hit, no rebuild.
    again = test_state.pipelines.load_gpu_pipeline("fast")
    assert again is base  # same VideoPipelineState wrapper
    assert create_calls_after() == creates_after_base

    # Different adapter -> rebuild: new state wrapper, create() invoked, adapter recorded.
    lora_a = test_state.pipelines.load_gpu_pipeline("fast", lora_path="/loras/a.safetensors", lora_scale=0.8)
    assert lora_a.lora_path == "/loras/a.safetensors"
    assert lora_a.lora_scale == 0.8
    assert fake.last_lora_path == "/loras/a.safetensors"
    assert fake.last_lora_scale == 0.8
    assert lora_a is not base
    creates_after_a = create_calls_after()
    assert creates_after_a == creates_after_base + 1

    # Same adapter again -> cache hit, no rebuild.
    lora_a_again = test_state.pipelines.load_gpu_pipeline("fast", lora_path="/loras/a.safetensors", lora_scale=0.8)
    assert lora_a_again is lora_a
    assert create_calls_after() == creates_after_a

    # Different adapter -> rebuild.
    lora_b = test_state.pipelines.load_gpu_pipeline("fast", lora_path="/loras/b.safetensors")
    assert lora_b.lora_path == "/loras/b.safetensors"
    assert fake.last_lora_path == "/loras/b.safetensors"
    assert lora_b is not lora_a
    creates_after_b = create_calls_after()
    assert creates_after_b == creates_after_a + 1

    # Back to base model -> rebuild (different cache key from any LoRA).
    base_again = test_state.pipelines.load_gpu_pipeline("fast")
    assert base_again.lora_path is None
    assert fake.last_lora_path is None
    assert base_again is not lora_b
    assert create_calls_after() == creates_after_b + 1

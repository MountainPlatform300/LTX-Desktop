"""Tests for checkpoint specs and pure path helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from api_types import ModelCheckpointID
from runtime_config.model_download_specs import (
    ALL_MODEL_CP_IDS,
    ALL_LTX_LOCAL_MODEL_IDS,
    ModelCheckpointSpec,
    get_ic_loras_cp_ids,
    get_latest_ltx_model_id,
    get_ltx_cps,
    get_ltx_model_cp_ids,
    get_ltx_model_spec,
    get_model_cp_spec,
    resolve_downloading_dir,
    resolve_downloading_path,
    resolve_downloading_target_path,
    resolve_model_path,
)


def test_specs_cover_all_checkpoint_ids():
    assert set(ALL_MODEL_CP_IDS) == {cp_id for cp_id in ALL_MODEL_CP_IDS}


def test_primary_ltx_checkpoints_map_1_to_1_with_ltx_models():
    assert len(get_ltx_cps()) == len(ALL_LTX_LOCAL_MODEL_IDS)


def test_latest_ltx_model_is_relevant():
    latest = get_latest_ltx_model_id()
    spec = get_ltx_model_spec(latest)
    assert spec.model_cp in get_ltx_cps()


def test_ic_lora_cp_ids_are_deduped():
    spec = get_ltx_model_spec(get_latest_ltx_model_id())
    assert get_ic_loras_cp_ids(spec.ic_loras_spec) == ("ltx-2.3-22b-ic-lora-union-control-ref0.5",)


def test_ltx_model_cp_ids_include_deduped_ic_loras():
    spec = get_ltx_model_spec(get_latest_ltx_model_id())
    assert get_ltx_model_cp_ids(get_latest_ltx_model_id()) == (
        spec.model_cp,
        spec.upscale_cp,
        spec.text_encoder_cp,
        "ltx-2.3-22b-ic-lora-union-control-ref0.5",
    )


def test_model_path_resolves_from_relative_path(tmp_path):
    cp_id: ModelCheckpointID = "gemma-3-12b-it-qat-q4_0-unquantized"
    spec = get_model_cp_spec(cp_id)
    assert resolve_model_path(tmp_path, cp_id) == tmp_path / spec.relative_path


def test_downloading_path_is_derived_from_spec():
    models_dir = Path("/tmp/models")
    downloading_dir = resolve_downloading_dir(models_dir)

    assert resolve_downloading_path(models_dir, "ltx-2.3-22b-distilled") == downloading_dir
    assert (
        resolve_downloading_path(models_dir, "gemma-3-12b-it-qat-q4_0-unquantized")
        == downloading_dir / "gemma-3-12b-it-qat-q4_0-unquantized"
    )
    assert resolve_downloading_target_path(models_dir, "ltx-2.3-22b-distilled") == downloading_dir / "ltx-2.3-22b-distilled.safetensors"


def test_relative_paths_are_unique():
    relative_paths = {get_model_cp_spec(cp_id).relative_path for cp_id in ALL_MODEL_CP_IDS}
    assert len(relative_paths) == len(ALL_MODEL_CP_IDS)


def test_new_image_model_checkpoints_have_specs_and_unique_folders():
    new_ids: tuple[ModelCheckpointID, ...] = (
        "flux-2-klein-9b",
    )
    for cp_id in new_ids:
        spec = get_model_cp_spec(cp_id)
        assert spec.is_folder is True
        assert spec.repo_id
        assert spec.expected_size_bytes > 0


def test_gated_flag_is_set_on_gated_repos_only():
    gated = {"flux-2-klein-9b"}
    for cp_id in ALL_MODEL_CP_IDS:
        spec = get_model_cp_spec(cp_id)
        if cp_id in gated:
            assert spec.is_gated is True, f"{cp_id} should be gated"
        else:
            assert spec.is_gated is False, f"{cp_id} should not be gated"


def test_new_image_model_paths_are_unique_from_existing():
    existing_paths = {
        get_model_cp_spec(cp_id).relative_path
        for cp_id in ALL_MODEL_CP_IDS
        if cp_id not in {"flux-2-klein-9b"}
    }
    for cp_id in ("flux-2-klein-9b",):
        assert get_model_cp_spec(cp_id).relative_path not in existing_paths


def test_model_path_rejects_parent_traversal(monkeypatch, tmp_path):
    bad_spec = ModelCheckpointSpec(
        relative_path=Path("../escape.safetensors"),
        expected_size_bytes=1,
        is_folder=False,
        repo_id="test/repo",
        description="bad",
    )

    monkeypatch.setattr(
        "runtime_config.model_download_specs.get_model_cp_spec",
        lambda cp_id: bad_spec,
    )

    with pytest.raises(ValueError):
        resolve_model_path(tmp_path, "ltx-2.3-22b-distilled")

"""Tests for the image-model catalog (Z-Image + open-weight additions)."""

from __future__ import annotations

from pathlib import Path

from runtime_config.image_model_specs import (
    IMAGE_MODELS,
    build_image_model_specs_response,
    get_default_image_model_spec,
    get_image_model_spec,
    resolve_image_model_spec,
)
from runtime_config.model_download_specs import get_model_cp_spec, is_cp_downloaded


def test_catalog_first_entry_is_z_image_available():
    spec = IMAGE_MODELS[0]
    assert spec.id == "z-image-turbo"
    assert spec.inference_status == "available"
    assert spec.gated is False


def test_catalog_only_contains_z_image_and_klein():
    ids = {s.id for s in IMAGE_MODELS}
    assert ids == {"z-image-turbo", "flux-2-klein-9b"}


def test_catalog_gated_flags_match_checkpoint_specs():
    for spec in IMAGE_MODELS:
        assert spec.gated == get_model_cp_spec(spec.checkpoint_id).is_gated


def test_catalog_checkpoint_ids_are_valid_and_unique():
    cp_ids = [s.checkpoint_id for s in IMAGE_MODELS]
    assert len(cp_ids) == len(set(cp_ids))
    # Every catalog checkpoint has a download spec (would raise if not mapped).
    for cp_id in cp_ids:
        assert get_model_cp_spec(cp_id).repo_id


def test_resolve_image_model_spec_defaults_to_z_image():
    assert resolve_image_model_spec(None).id == "z-image-turbo"
    assert resolve_image_model_spec("unknown-id").id == "z-image-turbo"
    assert resolve_image_model_spec("flux-2-klein-9b").id == "flux-2-klein-9b"


def test_get_default_image_model_spec_is_z_image():
    assert get_default_image_model_spec().id == "z-image-turbo"


def test_get_image_model_spec_returns_none_for_unknown():
    assert get_image_model_spec("nope") is None


def test_build_response_marks_downloaded_flag_from_disk(tmp_path):
    response = build_image_model_specs_response(tmp_path)
    by_id = {m.id: m for m in response.models}
    # Empty models dir → nothing downloaded.
    assert all(not m.downloaded for m in response.models)

    # Materialize the Z-Image folder so is_cp_downloaded flips for it.
    z_image_path = tmp_path / get_model_cp_spec("z-image-turbo").relative_path
    z_image_path.mkdir(parents=True, exist_ok=True)
    (z_image_path / "model.safetensors").write_bytes(b"\x00")
    assert is_cp_downloaded(tmp_path, "z-image-turbo")

    response = build_image_model_specs_response(tmp_path)
    by_id = {m.id: m for m in response.models}
    assert by_id["z-image-turbo"].downloaded is True
    assert by_id["flux-2-klein-9b"].downloaded is False


def test_build_response_surfaces_repo_id_and_size(tmp_path):
    response = build_image_model_specs_response(tmp_path)
    klein = next(m for m in response.models if m.id == "flux-2-klein-9b")
    assert klein.repo_id == "black-forest-labs/FLUX.2-klein-9B"
    assert klein.size_bytes == get_model_cp_spec("flux-2-klein-9b").expected_size_bytes
    assert klein.default_resolution == (1024, 1024)
    assert (1024, 1024) in klein.supported_resolutions


def test_path_argument_is_used_for_download_check(tmp_path: Path) -> None:
    # Sanity: the response builder uses the passed models_dir, not a global.
    other = tmp_path / "other"
    other.mkdir()
    z_image_path = other / get_model_cp_spec("z-image-turbo").relative_path
    z_image_path.mkdir(parents=True, exist_ok=True)
    (z_image_path / "model.safetensors").write_bytes(b"\x00")

    assert build_image_model_specs_response(other).models[0].downloaded is True
    assert build_image_model_specs_response(tmp_path).models[0].downloaded is False

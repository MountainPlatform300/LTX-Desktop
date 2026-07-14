"""Unit tests for the LoRA key-overlap diagnostic helper.

These craft a real (header-only) safetensors byte payload by hand — the same
bytes `safetensors` would write for a zero-data tensor set — so the parser is
exercised against the actual on-disk format without depending on the
`safetensors` package (not a declared backend dep) or loading any tensors.
"""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path

import pytest

from services.lora_diagnostics import _safetensors_keys, log_lora_diagnostics


def _write_safetensors(path: Path, keys: list[str]) -> None:
    """Write a minimal valid safetensors file whose header lists `keys`.

    Each key maps to a zero-byte F32 tensor (empty shape, data_offsets [0, 0]),
    which is all the diagnostic parser needs — it only reads the header JSON.
    """
    header: dict[str, object] = {
        k: {"dtype": "F32", "shape": [], "data_offsets": [0, 0]} for k in keys
    }
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)


# The base checkpoint is stored with a `model.diffusion_model.` prefix that the
# loader's base rename map strips; the LoRA is stored with `diffusion_model.`
# that the LoRA rename map strips. After both strips the prefixes must line up.
_BASE_KEYS = [
    "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight",
    "model.diffusion_model.transformer_blocks.0.attn1.to_k.weight",
    "model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight",
]


def test_safetensors_keys_reads_only_header(tmp_path: Path) -> None:
    path = tmp_path / "native.safetensors"
    _write_safetensors(path, ["attn1.to_q.lora_A.weight", "attn1.to_q.lora_B.weight"])

    keys = _safetensors_keys(str(path))

    assert keys is not None
    assert set(keys) == {"attn1.to_q.lora_A.weight", "attn1.to_q.lora_B.weight"}


def test_safetensors_keys_returns_none_for_non_safetensors(tmp_path: Path) -> None:
    path = tmp_path / "not_st.safetensors"
    path.write_bytes(b"definitely not a safetensors file")

    assert _safetensors_keys(str(path)) is None


def test_log_diagnostics_warns_on_zero_overlap(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # LoRA targets module names that do NOT exist in the base after both rename
    # maps -> zero overlap -> the real silent no-op case.
    base = tmp_path / "base.safetensors"
    _write_safetensors(base, _BASE_KEYS)
    lora = tmp_path / "mismatched.safetensors"
    _write_safetensors(
        lora,
        [
            "diffusion_model.transformer_blocks.0.some_other_module.to_q.lora_A.weight",
            "diffusion_model.transformer_blocks.0.some_other_module.to_q.lora_B.weight",
            "diffusion_model.transformer_blocks.0.attn1.to_z.lora_A.weight",
            "diffusion_model.transformer_blocks.0.attn1.to_z.lora_B.weight",
        ],
    )

    with caplog.at_level(logging.INFO, logger="services.lora_diagnostics"):
        log_lora_diagnostics(
            lora_path=str(lora),
            label="IC-LoRA",
            base_checkpoint_path=str(base),
        )

    assert any("matches ZERO base modules" in rec.message for rec in caplog.records)
    assert any("matched=0" in rec.message for rec in caplog.records)


def test_log_diagnostics_no_warning_when_prefixes_line_up(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # The real LTX-2.3 IC-LoRA format: diffusion_model.transformer_blocks.N...
    # keys that, after stripping diffusion_model., line up with the base's
    # model.diffusion_model. stripped prefixes -> full overlap, no warning.
    base = tmp_path / "base.safetensors"
    _write_safetensors(base, _BASE_KEYS)
    lora = tmp_path / "aligned.safetensors"
    _write_safetensors(
        lora,
        [
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight",
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight",
            "diffusion_model.transformer_blocks.0.attn1.to_k.lora_A.weight",
            "diffusion_model.transformer_blocks.0.attn1.to_k.lora_B.weight",
        ],
    )

    with caplog.at_level(logging.INFO, logger="services.lora_diagnostics"):
        log_lora_diagnostics(
            lora_path=str(lora),
            label="IC-LoRA",
            base_checkpoint_path=str(base),
        )

    assert not any("silently no-op" in rec.message for rec in caplog.records)
    assert not any("matches ZERO" in rec.message for rec in caplog.records)
    assert any("matched=2" in rec.message for rec in caplog.records)


def test_log_diagnostics_handles_missing_path(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="services.lora_diagnostics"):
        log_lora_diagnostics(
            lora_path="/does/not/exist.safetensors",
            label="distilled t2v",
        )

    assert any("does not exist" in rec.message for rec in caplog.records)


def test_log_diagnostics_base_only_when_no_lora(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="services.lora_diagnostics"):
        log_lora_diagnostics(
            lora_path=None,
            label="distilled t2v",
        )

    assert any("base-only" in rec.message for rec in caplog.records)

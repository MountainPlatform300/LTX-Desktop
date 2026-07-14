"""Diagnostic logging for LoRA adapters loaded for inference.

The LTX loader fuses LoRA weights directly into the base model's state dict:
for each base weight ``<prefix>.weight`` it looks for ``<prefix>.lora_A.weight``
and ``<prefix>.lora_B.weight`` in the (rename-mapped) LoRA state dict. If the
prefixes don't line up, the adapter matches **zero** modules and silently
no-ops — the output is indistinguishable from the base model. That is the
classic "my imported LoRA has no effect" failure mode.

Rather than guess from key-prefix heuristics, this module reads **only the
safetensors JSON header** (the 8-byte length prefix + the header JSON — no
tensors) of both the LoRA and the base checkpoint, applies the same rename
transforms the loader uses, and reports the real overlap:

* ``matched``  — LoRA target prefixes that line up with a base ``.weight`` key.
* ``missed``   — LoRA target prefixes that don't.

``matched == 0`` is the only condition that produces a no-op warning, so the
diagnostic never cries wolf when the adapter is actually fine. It is cheap
(header-only) and safe to run on every load; it never raises.
"""

from __future__ import annotations

import json
import logging
import os
import struct
from typing import cast

logger = logging.getLogger(__name__)

# How many leading keys to print as a sample — enough to eyeball the format
# without flooding the log for a large adapter.
_SAMPLE_KEY_COUNT = 6

# Replicates ``LTXV_LORA_COMFY_RENAMING_MAP`` (with_matching() matches all;
# replace "diffusion_model." -> ""). The map replaces every occurrence, but in
# practice each LTX LoRA key carries at most one such prefix.
def _lora_key_after_rename(key: str) -> str:
    return key.replace("diffusion_model.", "")


# Replicates ``LTXV_MODEL_COMFY_RENAMING_MAP`` (matching prefix
# "model.diffusion_model." then replace "model.diffusion_model." -> "").
def _base_key_after_rename(key: str) -> str:
    if key.startswith("model.diffusion_model."):
        return key[len("model.diffusion_model.") :]
    return key


def _safetensors_keys(path: str) -> list[str] | None:
    """Read just the tensor key names from a safetensors file (no tensors loaded).

    Returns ``None`` if the file can't be parsed as safetensors so the caller can
    log + skip; never raises.
    """
    try:
        file_size = os.path.getsize(path)
        with open(path, "rb") as handle:
            header_len_bytes = handle.read(8)
            if len(header_len_bytes) < 8:
                return None
            (header_len,) = struct.unpack("<Q", header_len_bytes)
            # Bound the header length against the real file size so a non-
            # safetensors file (whose first 8 bytes decode to an absurd u64)
            # is rejected instead of triggering a multi-GB read / MemoryError.
            if header_len > file_size - 8:
                return None
            header_bytes = handle.read(header_len)
        parsed = json.loads(header_bytes)
        if not isinstance(parsed, dict):
            return None
        header = cast("dict[str, object]", parsed)
        keys: list[str] = []
        for raw_key in header.keys():
            if raw_key != "__metadata__":
                keys.append(raw_key)
        return keys
    except (OSError, ValueError, struct.error, json.JSONDecodeError, MemoryError) as exc:
        logger.warning("LoRA diagnostic: could not read safetensors header at %s: %s", path, exc)
        return None


def _lora_target_prefixes(lora_keys: list[str]) -> set[str]:
    """Distinct base-module prefixes this adapter targets, after the LoRA rename."""
    prefixes: set[str] = set()
    for key in lora_keys:
        suffix = ".lora_A.weight"
        if key.endswith(suffix):
            prefixes.add(_lora_key_after_rename(key[: -len(suffix)]))
    return prefixes


def _base_weight_prefixes(base_keys: list[str]) -> set[str]:
    """Distinct base ``.weight`` module prefixes, after the base rename."""
    prefixes: set[str] = set()
    for key in base_keys:
        suffix = ".weight"
        if key.endswith(suffix):
            prefixes.add(_base_key_after_rename(key[: -len(suffix)]))
    return prefixes


def log_lora_diagnostics(
    *,
    lora_path: str | None,
    label: str,
    base_checkpoint_path: str | None = None,
    distilled_lora_path: str | None = None,
    applying_comfy_rename: bool = True,  # noqa: ARG001 — kept for call-site compat
) -> None:
    """Log key-overlap diagnostics for a LoRA about to be loaded for inference.

    ``label`` identifies the pipeline / role (e.g. ``"IC-LoRA"`` or
    ``"distilled t2v"``). Safe to call with a missing or non-safetensors path —
    it logs the failure and returns. Never raises, so it can wrap any load site
    without risking the generation.
    """
    base_name = os.path.basename(base_checkpoint_path) if base_checkpoint_path else "<none>"
    logger.info(
        "LoRA diagnostic [%s]: base_checkpoint=%s distilled_lora_stacked=%s",
        label,
        base_name,
        distilled_lora_path is not None,
    )
    if not lora_path:
        logger.info("LoRA diagnostic [%s]: no adapter path — generating base-only", label)
        return
    if not os.path.exists(lora_path):
        logger.warning(
            "LoRA diagnostic [%s]: adapter path does not exist: %s", label, lora_path
        )
        return
    lora_keys = _safetensors_keys(lora_path)
    if not lora_keys:
        logger.warning(
            "LoRA diagnostic [%s]: no tensor keys read from %s — "
            "is this a .safetensors LoRA?",
            label,
            lora_path,
        )
        return

    lora_prefixes = _lora_target_prefixes(lora_keys)
    sample = ", ".join(lora_keys[:_SAMPLE_KEY_COUNT])
    logger.info(
        "LoRA diagnostic [%s]: file=%s keys=%d target_prefixes=%d sample_keys=[%s]",
        label,
        os.path.basename(lora_path),
        len(lora_keys),
        len(lora_prefixes),
        sample,
    )

    # Compute the real fusion overlap against the base checkpoint when we can.
    base_keys = _safetensors_keys(base_checkpoint_path) if base_checkpoint_path else None
    if not base_keys:
        logger.info(
            "LoRA diagnostic [%s]: base checkpoint unreadable — skipping fusion "
            "overlap check (this is not a warning).",
            label,
        )
        return
    base_prefixes = _base_weight_prefixes(base_keys)
    matched = lora_prefixes & base_prefixes
    missed = lora_prefixes - base_prefixes
    logger.info(
        "LoRA diagnostic [%s]: fusion overlap matched=%d missed=%d "
        "(base_weight_modules=%d)",
        label,
        len(matched),
        len(missed),
        len(base_prefixes),
    )
    if not lora_prefixes:
        logger.warning(
            "LoRA diagnostic [%s]: %s has no lora_A/lora_B keys — the adapter will "
            "do nothing. This is the probable cause of an imported LoRA having no "
            "visible effect.",
            label,
            os.path.basename(lora_path),
        )
    elif not matched:
        sample_missed = ", ".join(sorted(missed)[:3])
        logger.warning(
            "LoRA diagnostic [%s]: %s matches ZERO base modules after the rename "
            "map (missed %d/%d, e.g. %s) — the adapter will silently no-op. This is "
            "the probable cause of an imported LoRA having no visible effect.",
            label,
            os.path.basename(lora_path),
            len(missed),
            len(lora_prefixes),
            sample_missed,
        )

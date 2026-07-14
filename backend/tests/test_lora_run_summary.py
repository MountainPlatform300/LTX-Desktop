"""Tests for the LoRA training run-summary Markdown builder."""

from __future__ import annotations

from handlers.lora_run_summary import build_run_summary_markdown, format_duration
from state.lora_training_state import (
    LoraClip,
    LoraDataset,
    PreprocessedDataset,
    TrainingConfig,
    TrainingJob,
)


def test_format_duration_variants() -> None:
    assert format_duration("2026-06-25T15:00:00", "2026-06-25T15:00:42") == "42s"
    assert format_duration("2026-06-25T15:00:00", "2026-06-25T15:43:12") == "43m 12s"
    assert format_duration("2026-06-25T15:00:00", "2026-06-25T16:18:00") == "1h 18m"
    assert format_duration(None, "2026-06-25T15:00:00") is None
    assert format_duration("2026-06-25T15:00:00", "2026-06-25T14:00:00") is None


def test_summary_includes_key_facts() -> None:
    job = TrainingJob(
        id="abc123",
        preprocessed_id="pre1",
        name="cleanplate",
        created_at="2026-06-25T15:34:00",
        status="completed",
        config=TrainingConfig(steps=2000, rank=32, alpha=32, preset="low_vram"),
        provider="runpod",
        trainer_repo_url="https://github.com/Lightricks/LTX-2.git",
        trainer_repo_ref="abc987",
        total_steps=2000,
        gpu_type="NVIDIA B200",
        gpu_vram_gb=180,
        started_at="2026-06-25T15:35:00",
        first_step_at="2026-06-25T15:45:00",
        completed_at="2026-06-25T16:28:00",
        remote_output_dir="/workspace/outputs/cleanplate-abc123",
    )
    dataset = LoraDataset(
        id="ds1", name="Clean Plate", created_at="2026-06-25T14:00:00",
        status="uploaded", type="ic_lora",
        clips=[LoraClip(id="c1", local_path="/x.mp4")],
    )
    pre = PreprocessedDataset(
        id="pre1", dataset_id="ds1", created_at="2026-06-25T15:20:00",
        status="ready", resolution_buckets="768x448x49", with_audio=False,
    )
    md = build_run_summary_markdown(
        job=job, preprocessed=pre, dataset=dataset,
        local_lora_path="/local/abc123.safetensors",
    )
    assert "# cleanplate" in md
    assert "NVIDIA B200 (180 GB)" in md
    # Total = click -> done (54m); Setup = click -> first step (11m); Training
    # = first step -> done (43m). The silent setup phase is now explicit.
    assert "| Total time | 54m" in md
    assert "| Setup (load + validation) | 11m" in md
    assert "| Training (steps) | 43m" in md
    assert "768x448x49" in md
    assert "32 / 32" in md
    # low_vram preset resolves acceleration when not set explicitly.
    assert "adamw8bit" in md and "int8-quanto" in md
    assert "LTX-2.git@abc987" in md
    assert "/local/abc123.safetensors" in md


def test_summary_without_first_step_omits_breakdown() -> None:
    # A run that failed before any step: no first_step_at -> Setup/Training are
    # "—" but Total still computes from created_at -> completed_at.
    job = TrainingJob(
        id="x", preprocessed_id="p", name="run", created_at="2026-06-25T15:00:00",
        status="failed", config=TrainingConfig(), provider="runpod",
        started_at="2026-06-25T15:01:00", completed_at="2026-06-25T15:12:00",
    )
    md = build_run_summary_markdown(
        job=job, preprocessed=None, dataset=None, local_lora_path="/x.safetensors",
    )
    assert "| Total time | 12m" in md
    assert "| Setup (load + validation) | — |" in md
    assert "| Training (steps) | — |" in md


def test_summary_degrades_without_related_records() -> None:
    job = TrainingJob(
        id="x", preprocessed_id="p", name="run", created_at="2026-06-25T15:00:00",
        status="completed", config=TrainingConfig(), provider="runpod",
    )
    md = build_run_summary_markdown(
        job=job, preprocessed=None, dataset=None, local_lora_path="/x.safetensors",
    )
    assert "# run" in md  # builds without crashing

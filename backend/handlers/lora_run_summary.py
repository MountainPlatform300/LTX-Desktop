"""Build a human-readable Markdown summary of a finished LoRA training run.

Pure: given the job + its preprocessed/dataset records, returns a Markdown
string. The runner writes it next to the downloaded adapter so a run's details
(time taken, GPU, steps, rank, dataset, paths) live on disk — mirroring the
LTX-2 train-model skill's `run-summary.md`.
"""

from __future__ import annotations

from datetime import datetime

from state.lora_training_state import (
    LoraDataset,
    PreprocessedDataset,
    TrainingConfig,
    TrainingJob,
)


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def format_duration(started_at: str | None, completed_at: str | None) -> str | None:
    """Wall-clock run time as e.g. "43m 12s" / "1h 18m", or None if unknown."""
    start = _parse_iso(started_at)
    end = _parse_iso(completed_at)
    if start is None or end is None:
        return None
    total = int((end - start).total_seconds())
    if total < 0:
        return None
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _acceleration(config: TrainingConfig) -> tuple[str, str]:
    """(optimizer, quantization) resolved from preset when not set explicitly."""
    low_vram = config.preset == "low_vram"
    optimizer = config.optimizer_type or ("adamw8bit" if low_vram else "adamw")
    quant = config.quantization or ("int8-quanto" if low_vram else "none")
    return optimizer, quant


def build_run_summary_markdown(
    *,
    job: TrainingJob,
    preprocessed: PreprocessedDataset | None,
    dataset: LoraDataset | None,
    local_lora_path: str,
) -> str:
    """Render the run summary. Missing related records degrade gracefully."""
    cfg = job.config
    optimizer, quant = _acceleration(cfg)
    steps = job.total_steps or cfg.steps
    # Total wall clock is from the click (created_at) to completion. Split it into
    # Setup (pod boot + model load from the network volume + the one-time step-0
    # validation, which produces no step output) and Training (actual stepping),
    # using the first observed step as the boundary. This makes the ~10 min of
    # "dead air" before step 1 explicit instead of hiding it in one number.
    total = format_duration(job.created_at, job.completed_at)
    setup = format_duration(job.created_at, job.first_step_at)
    training = format_duration(job.first_step_at, job.completed_at)
    # Avg step time is measured over the stepping window only (excludes setup).
    avg_step = None
    first_step, end = _parse_iso(job.first_step_at), _parse_iso(job.completed_at)
    if first_step and end and steps:
        secs = (end - first_step).total_seconds()
        if secs > 0:
            avg_step = f"{secs / steps:.1f}s"

    gpu = job.gpu_type or "—"
    if job.gpu_vram_gb:
        gpu = f"{gpu} ({job.gpu_vram_gb} GB)"

    dataset_type = dataset.type if dataset else "—"
    clip_count = len(dataset.clips) if dataset else None
    resolution = preprocessed.resolution_buckets if preprocessed else "—"
    # Surface a bucket collapse: when an IC-LoRA low_vram run had multiple
    # buckets configured, preprocessing trains at the first one only. Show the
    # real trained resolution (and that it was narrowed) instead of the
    # user's uncollapsed list.
    if preprocessed and preprocessed.effective_resolution_buckets:
        resolution = (
            f"{preprocessed.effective_resolution_buckets} "
            f"(narrowed from {preprocessed.resolution_buckets})"
        )
    with_audio = preprocessed.with_audio if preprocessed else False
    captioner = preprocessed.captioner_type if preprocessed else "—"
    trigger = cfg.trigger_word or "—"
    trainer = (
        f"{job.trainer_repo_url}@{job.trainer_repo_ref}"
        if job.trainer_repo_url and job.trainer_repo_ref
        else "—"
    )

    rows: list[tuple[str, str]] = [
        ("Status", job.status),
        ("Total time", total or "—"),
        ("Setup (load + validation)", setup or "—"),
        ("Training (steps)", training or "—"),
        ("Avg step time", avg_step or "—"),
        ("GPU", gpu),
        ("Provider", job.provider),
        ("Trainer revision", trainer),
        ("Steps", str(steps)),
        ("LoRA rank / alpha", f"{cfg.rank} / {cfg.alpha}"),
        ("Preset", cfg.preset),
        ("Learning rate", f"{cfg.learning_rate:g}"),
        ("Optimizer", optimizer),
        ("Quantization", quant),
        ("Target modules", ", ".join(cfg.target_modules) or "—"),
        ("Dataset", dataset.name if dataset else "—"),
        ("Dataset type", dataset_type),
        ("Training clips", str(clip_count) if clip_count is not None else "—"),
        ("Resolution (WxHxF)", resolution),
        ("Audio", "yes" if with_audio else "no"),
        ("Captioner", captioner),
        ("Trigger word", trigger),
        ("Started", job.started_at or "—"),
        ("Completed", job.completed_at or "—"),
    ]
    table = "\n".join(f"| {k} | {v} |" for k, v in rows)

    return (
        f"# {job.name}\n\n"
        f"LoRA training run summary (generated by LTX Desktop).\n\n"
        f"| Field | Value |\n| --- | --- |\n{table}\n\n"
        f"## Output\n\n"
        f"- Trained adapter: `{local_lora_path}`\n"
        f"- Remote output dir: `{job.remote_output_dir or '—'}`\n"
        f"- Run id: `{job.id}`\n"
    )

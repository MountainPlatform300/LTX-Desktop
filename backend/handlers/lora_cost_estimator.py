"""Pure, deliberately coarse LoRA training time/cost estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Confidence = Literal["low", "medium", "high"]
PhaseName = Literal["provision", "upload", "preprocess", "train", "idle"]
_STANDARD_STORAGE_FIRST_TB_GB = 1000
_STANDARD_STORAGE_OVER_TB_RATE = 0.05


@dataclass(frozen=True)
class EstimateInputs:
    steps: int
    clip_count: int
    total_clip_seconds: float
    preprocessed: bool
    resolution_buckets: str
    mode: str
    with_audio: bool
    gpu_price_per_hr: float
    storage_readiness: Literal["ready", "missing", "unknown"]
    estimated_model_download_bytes: int | None
    idle_timeout_minutes: int
    storage_size_gb: int


@dataclass(frozen=True)
class HistoricalTiming:
    steps: int
    setup_seconds: int
    train_seconds: int
    upload_seconds: int | None = None
    preprocess_seconds: int | None = None


@dataclass(frozen=True)
class PhaseEstimate:
    phase: PhaseName
    low_seconds: int
    high_seconds: int


@dataclass(frozen=True)
class CostEstimate:
    low_seconds: int
    high_seconds: int
    low_gpu_cost: float
    high_gpu_cost: float
    phases: tuple[PhaseEstimate, ...]
    confidence: Confidence
    matched_history_count: int
    download_bytes: int | None
    storage_monthly_cost: float


def _minute(seconds: float) -> int:
    return max(0, int((seconds + 59) // 60) * 60)


def storage_monthly_cost(size_gb: int) -> float:
    first_tb = min(size_gb, _STANDARD_STORAGE_FIRST_TB_GB)
    over_tb = max(0, size_gb - _STANDARD_STORAGE_FIRST_TB_GB)
    return round(
        first_tb * 0.07 + over_tb * _STANDARD_STORAGE_OVER_TB_RATE,
        2,
    )


def estimate_cost(
    inputs: EstimateInputs, history: tuple[HistoricalTiming, ...] = ()
) -> CostEstimate:
    comparable = tuple(
        item
        for item in history
        if item.steps > 0 and 0.5 <= inputs.steps / item.steps <= 2.0
    )
    if comparable:
        setup_rates = sorted(item.setup_seconds for item in comparable)
        train_rates = sorted(item.train_seconds / item.steps for item in comparable)
        setup_low = setup_rates[max(0, len(setup_rates) // 4)]
        setup_high = setup_rates[min(len(setup_rates) - 1, 3 * len(setup_rates) // 4)]
        train_low = train_rates[max(0, len(train_rates) // 4)] * inputs.steps
        train_high = train_rates[min(len(train_rates) - 1, 3 * len(train_rates) // 4)] * inputs.steps
        provision = (max(30, setup_low * 0.7), max(120, setup_high * 1.3))
        train = (max(60, train_low * 0.85), max(180, train_high * 1.2))
        confidence: Confidence = "high" if len(comparable) >= 5 else "medium"
    else:
        provision = {
            "ready": (60, 240),
            "missing": (900, 2700),
            "unknown": (300, 2700),
        }[inputs.storage_readiness]
        bucket = inputs.resolution_buckets.split(";")[0].split("x")
        try:
            pixels_frames = int(bucket[0]) * int(bucket[1]) * int(bucket[2])
        except (IndexError, ValueError):
            pixels_frames = 768 * 448 * 49
        complexity = max(0.6, pixels_frames / (768 * 448 * 49))
        mode_factor = 1.25 if inputs.mode == "ic_lora" else 1.0
        audio_factor = 1.15 if inputs.with_audio else 1.0
        per_step = complexity * mode_factor * audio_factor
        train = (inputs.steps * 0.7 * per_step, inputs.steps * 3.0 * per_step)
        confidence = "low"

    upload_history = sorted(
        item.upload_seconds for item in comparable if item.upload_seconds is not None
    )
    upload = (
        (
            upload_history[max(0, len(upload_history) // 4)] * 0.8,
            upload_history[min(len(upload_history) - 1, 3 * len(upload_history) // 4)]
            * 1.25,
        )
        if upload_history
        else (
            30 + inputs.clip_count * 2,
            180 + inputs.clip_count * 15 + inputs.total_clip_seconds * 2,
        )
    )
    preprocess_history = sorted(
        item.preprocess_seconds
        for item in comparable
        if item.preprocess_seconds is not None
    )
    if inputs.preprocessed:
        preprocess = (0, 0)
    elif preprocess_history:
        preprocess = (
            preprocess_history[max(0, len(preprocess_history) // 4)] * 0.8,
            preprocess_history[
                min(len(preprocess_history) - 1, 3 * len(preprocess_history) // 4)
            ]
            * 1.25,
        )
    else:
        preprocess = (
            60 + inputs.total_clip_seconds * 2,
            600 + inputs.total_clip_seconds * (12 if inputs.with_audio else 8),
        )
    raw: tuple[tuple[PhaseName, tuple[float, float]], ...] = (
        ("provision", provision),
        ("upload", upload),
        ("preprocess", preprocess),
        ("train", train),
        ("idle", (0, inputs.idle_timeout_minutes * 60)),
    )
    phases = tuple(
        PhaseEstimate(phase, _minute(bounds[0]), _minute(bounds[1]))
        for phase, bounds in raw
    )
    low_seconds = sum(phase.low_seconds for phase in phases)
    high_seconds = sum(phase.high_seconds for phase in phases)
    return CostEstimate(
        low_seconds=low_seconds,
        high_seconds=high_seconds,
        low_gpu_cost=round(low_seconds / 3600 * inputs.gpu_price_per_hr, 2),
        high_gpu_cost=round(high_seconds / 3600 * inputs.gpu_price_per_hr, 2),
        phases=phases,
        confidence=confidence,
        matched_history_count=len(comparable),
        download_bytes=(
            0
            if inputs.storage_readiness == "ready"
            else inputs.estimated_model_download_bytes
        ),
        storage_monthly_cost=storage_monthly_cost(inputs.storage_size_gb),
    )

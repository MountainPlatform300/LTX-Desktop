"""Canonical state model for backend runtime state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NewType, Protocol

from api_types import ModelCheckpointID
from state.conditioning_cache import ConditioningCache

if TYPE_CHECKING:
    from state.app_settings import AppSettings
    from services.interfaces import (
        A2VPipeline,
        DepthProcessorPipeline,
        FastVideoPipeline,
        ImageEditPipeline,
        ImageGenerationPipeline,
        IcLoraPipeline,
        PoseProcessorPipeline,
        RetakePipeline,
        TextEncoder,
    )
    import torch


# Download session
# ============================================================


DownloadSessionId = NewType("DownloadSessionId", str)


@dataclass(frozen=True)
class DownloadSessionComplete:
    status: str = "complete"


@dataclass(frozen=True)
class DownloadSessionError:
    error_message: str
    status: str = "error"


DownloadSessionResult = DownloadSessionComplete | DownloadSessionError


def _default_completed_download_sessions() -> dict[DownloadSessionId, DownloadSessionResult]:
    return {}


@dataclass
class FileDownloadRunning:
    file_type: ModelCheckpointID
    target_path: str
    downloaded_bytes: int
    speed_bytes_per_sec: float


@dataclass
class DownloadingSession:
    id: DownloadSessionId
    current_running_file: FileDownloadRunning | None
    files_to_download: set[ModelCheckpointID]
    completed_files: set[ModelCheckpointID]
    completed_bytes: int


# ============================================================
# Text encoding
# ============================================================


@dataclass
class TextEncodingResult:
    video_context: torch.Tensor
    audio_context: torch.Tensor | None


class CachedTextEncoder(Protocol):
    def to(self, device: torch.device) -> "CachedTextEncoder":
        ...


def _new_prompt_cache() -> dict[tuple[str, bool], TextEncodingResult]:
    return {}


@dataclass
class TextEncoderState:
    service: TextEncoder
    prompt_cache: dict[tuple[str, bool], TextEncodingResult] = field(default_factory=_new_prompt_cache)
    api_embeddings: TextEncodingResult | None = None
    cached_encoder: CachedTextEncoder | None = None


# ============================================================
# Pipeline state
# ============================================================


@dataclass
class VideoPipelineState:
    pipeline: FastVideoPipeline
    is_compiled: bool
    # The adapter the pipeline was built with (None = base model). Part of the
    # GpuSlot cache key so switching standard LoRAs swaps the pipeline rather
    # than silently reusing the base model.
    lora_path: str | None = None
    lora_scale: float = 1.0


@dataclass
class PoseResources:
    pipeline: PoseProcessorPipeline
    person_detector_model_path: str
    pose_model_path: str


@dataclass
class ICLoraState:
    pipeline: IcLoraPipeline
    lora_path: str
    depth_pipeline: DepthProcessorPipeline
    depth_model_path: str
    lora_scale: float = 1.0
    pose_resources: PoseResources | None = None
    conditioning_cache: ConditioningCache = field(default_factory=ConditioningCache)
    # Base checkpoint the pipeline was built on (distilled, or dev when the
    # opt-in quality base is active). Part of the cache key so toggling the
    # setting evicts and reloads with the new base.
    base_checkpoint_path: str = ""


@dataclass
class A2VPipelineState:
    pipeline: A2VPipeline


@dataclass
class RetakePipelineState:
    pipeline: RetakePipeline
    distilled: bool
    quantized: bool


@dataclass
class KleinPipelineState:
    pipeline: ImageEditPipeline


# ============================================================
# Generation state
# ============================================================


@dataclass
class GenerationProgress:
    phase: str
    progress: int
    current_step: int | None
    total_steps: int | None


@dataclass
class GenerationRunning:
    id: str
    progress: GenerationProgress


@dataclass
class GenerationComplete:
    id: str
    result: str | list[str]


@dataclass
class GenerationError:
    id: str
    error: str


@dataclass
class GenerationCancelled:
    id: str


GenerationState = GenerationRunning | GenerationComplete | GenerationError | GenerationCancelled


@dataclass
class GpuGeneration:
    state: GenerationState


@dataclass
class ApiGeneration:
    state: GenerationState


ActiveGeneration = GpuGeneration | ApiGeneration


# ============================================================
# Device slots
# ============================================================


@dataclass
class GpuSlot:
    active_pipeline: (
        VideoPipelineState
        | ICLoraState
        | A2VPipelineState
        | RetakePipelineState
        | KleinPipelineState
        | ImageGenerationPipeline
    )


@dataclass
class CpuSlot:
    active_pipeline: ImageGenerationPipeline


# HuggingFace auth
# ============================================================


@dataclass(frozen=True)
class HfNotAuthenticated:
    pass


@dataclass(frozen=True)
class HfOAuthPending:
    state: str
    code_verifier: str
    created_at: float


@dataclass(frozen=True)
class HfAuthenticated:
    access_token: str
    expires_at: float


HfAuthState = HfNotAuthenticated | HfOAuthPending | HfAuthenticated


# ============================================================
# Top-level state
# ============================================================


@dataclass
class AppState:
    downloading_session: DownloadingSession | None
    gpu_slot: GpuSlot | None
    active_generation: ActiveGeneration | None
    cpu_slot: CpuSlot | None
    text_encoder: TextEncoderState | None
    app_settings: AppSettings
    completed_download_sessions: dict[DownloadSessionId, DownloadSessionResult] = field(
        default_factory=_default_completed_download_sessions
    )
    hf_auth_state: HfAuthState = field(default_factory=HfNotAuthenticated)

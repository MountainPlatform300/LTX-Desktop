"""Compatibility re-exports for service interfaces."""

from __future__ import annotations

from typing import Literal

from services.a2v_pipeline.a2v_pipeline import A2VPipeline
from services.clip_processor.clip_processor import (
    ClipProbeResult,
    ClipProcessor,
    ClipProcessorError,
    CropSpec,
    EditPlan,
    ScaleSpec,
    SceneSpan,
    TrimSpec,
)
from services.depth_processor_pipeline.depth_processor_pipeline import DepthProcessorPipeline
from services.fast_video_pipeline.fast_video_pipeline import FastVideoPipeline
from services.zit_api_client.zit_api_client import ZitAPIClient
from services.gpu_cleaner.gpu_cleaner import GpuCleaner
from services.gpu_info.gpu_info import GpuInfo, GpuTelemetryPayload
from services.http_client.http_client import HTTPClient, HttpResponseLike, HttpTimeoutError
from services.ic_lora_pipeline.ic_lora_pipeline import IcLoraPipeline
from services.image_editor.image_editor import (
    ImageEditor,
    ImageEditorError,
    NanoBananaModel,
)
from services.image_generation_pipeline.image_edit_pipeline import ImageEditPipeline
from services.image_generation_pipeline.image_generation_pipeline import ImageGenerationPipeline
from services.lora_prompt_profiler.lora_prompt_profiler import (
    LoraPromptProfile,
    LoraPromptProfileResult,
    LoraPromptProfileStatus,
    LoraPromptProfiler,
    NullLoraPromptProfiler,
)
from services.ltx_api_client.ltx_api_client import LTXAPIClient
from services.retake_pipeline.retake_pipeline import RetakePipeline
from services.model_downloader.model_downloader import ModelDownloader
from services.pose_processor_pipeline.pose_processor_pipeline import PoseProcessorPipeline
from services.services_utils import JSONScalar, JSONValue
from services.task_runner.task_runner import TaskRunner
from services.text_encoder.text_encoder import TextEncoder
from services.trainer_target.trainer_target import (
    RemoteCommandState,
    RemoteCommandStatus,
    TrainerCredentials,
    TrainerTarget,
    TrainerTargetError,
)
from services.video_captioner.video_captioner import VideoCaptioner, VideoCaptionerError
from services.video_processor.video_processor import VideoInfoPayload, VideoProcessor
from services.pexels_client.pexels_client import (
    PexelsClient,
    PexelsError,
    PexelsMediaKind,
    PexelsMediaResult,
    PexelsSearchResult,
)
from services.video_restyler.video_restyler import VideoRestyler, VideoRestylerError

VideoPipelineModelType = Literal["fast"]

__all__ = [
    "A2VPipeline",
    "JSONScalar",
    "JSONValue",
    "GpuTelemetryPayload",
    "VideoInfoPayload",
    "HttpTimeoutError",
    "HttpResponseLike",
    "HTTPClient",
    "ModelDownloader",
    "GpuCleaner",
    "GpuInfo",
    "VideoProcessor",
    "DepthProcessorPipeline",
    "PoseProcessorPipeline",
    "TaskRunner",
    "VideoPipelineModelType",
    "FastVideoPipeline",
    "ZitAPIClient",
    "ImageGenerationPipeline",
    "ImageEditPipeline",
    "IcLoraPipeline",
    "LTXAPIClient",
    "LoraPromptProfile",
    "LoraPromptProfileResult",
    "LoraPromptProfileStatus",
    "LoraPromptProfiler",
    "NullLoraPromptProfiler",
    "RetakePipeline",
    "TextEncoder",
    "TrainerTarget",
    "TrainerTargetError",
    "TrainerCredentials",
    "RemoteCommandStatus",
    "RemoteCommandState",
    "VideoCaptioner",
    "VideoCaptionerError",
    "ClipProcessor",
    "ClipProcessorError",
    "ClipProbeResult",
    "TrimSpec",
    "CropSpec",
    "ScaleSpec",
    "EditPlan",
    "SceneSpan",
    "ImageEditor",
    "ImageEditorError",
    "NanoBananaModel",
    "VideoRestyler",
    "VideoRestylerError",
    "PexelsClient",
    "PexelsError",
    "PexelsMediaKind",
    "PexelsMediaResult",
    "PexelsSearchResult",
]

"""Pydantic request/response models and typed aliases for ltx2_server."""

from __future__ import annotations

from typing import Annotated
from typing import Literal, NamedTuple, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmptyPrompt = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

#: Generic placeholder shipped as the default `validationPrompts` value. The
#: runner replaces it (and an empty list) with captions auto-seeded from the
#: dataset at training start, so a run never validates against this string
#: unless the user explicitly keeps it.
DEFAULT_VALIDATION_PROMPT = "A high quality sample from the trained concept."

ModelCheckpointID = Literal[
    "ltx-2.3-22b-distilled",
    "ltx-2.3-22b-dev",
    "ltx-2.3-22b-distilled-lora-384-1.1",
    "ltx-2.3-spatial-upscaler-x2-1.0",
    "ltx-2.3-22b-ic-lora-union-control-ref0.5",
    "dpt-hybrid-midas",
    "yolox-l-torchscript",
    "dw-ll-ucoco-384-bs5",
    "gemma-3-12b-it-qat-q4_0-unquantized",
    "z-image-turbo",
    "flux-2-klein-9b",
]
LTXLocalModelId = Literal["ltx-2.3-22b-distilled"]


class ImageConditioningInput(NamedTuple):
    """Image conditioning triplet used by all video pipelines."""

    path: str
    frame_idx: int
    strength: float


JsonObject: TypeAlias = dict[str, object]
VideoCameraMotion = Literal[
    "none",
    "dolly_in",
    "dolly_out",
    "dolly_left",
    "dolly_right",
    "jib_up",
    "jib_down",
    "static",
    "focus_shift",
]


# ============================================================
# Response Models
# ============================================================


class ModelStatusItem(BaseModel):
    id: str
    name: str
    loaded: bool
    downloaded: bool


class GpuTelemetry(BaseModel):
    name: str
    vram: int
    vramUsed: int


class HealthResponse(BaseModel):
    status: Literal["ok"]
    models_loaded: bool
    active_model: str | None
    gpu_info: GpuTelemetry
    sage_attention: bool
    models_status: list[ModelStatusItem]


class GpuInfoResponse(BaseModel):
    cuda_available: bool
    mps_available: bool = False
    gpu_available: bool = False
    gpu_name: str | None
    vram_gb: int | None
    gpu_info: GpuTelemetry


class RuntimePolicyResponse(BaseModel):
    force_api_generations: bool


class GenerationProgressResponse(BaseModel):
    status: Literal["idle", "running", "complete", "cancelled", "error"]
    phase: str
    progress: int
    currentStep: int | None
    totalSteps: int | None


class DownloadProgressRunningResponse(BaseModel):
    status: Literal["downloading"]
    current_downloading_file: ModelCheckpointID | None
    current_file_progress: float
    total_progress: float
    total_downloaded_bytes: int
    expected_total_bytes: int
    completed_files: set[ModelCheckpointID]
    all_files: set[ModelCheckpointID]
    error: None = None
    speed_bytes_per_sec: float


class DownloadProgressCompleteResponse(BaseModel):
    status: Literal["complete"]


class DownloadProgressErrorResponse(BaseModel):
    status: Literal["error"]
    error: str


DownloadProgressResponse: TypeAlias = (
    DownloadProgressRunningResponse | DownloadProgressCompleteResponse | DownloadProgressErrorResponse
)


class SuggestGapPromptResponse(BaseModel):
    status: Literal["success"] = "success"
    suggested_prompt: str


class GenerateVideoCompleteResponse(BaseModel):
    status: Literal["complete"]
    video_path: str


class GenerateVideoCancelledResponse(BaseModel):
    status: Literal["cancelled"]


GenerateVideoResponse: TypeAlias = GenerateVideoCompleteResponse | GenerateVideoCancelledResponse


class GenerateImageCompleteResponse(BaseModel):
    status: Literal["complete"]
    image_paths: list[str]


class GenerateImageCancelledResponse(BaseModel):
    status: Literal["cancelled"]


GenerateImageResponse: TypeAlias = GenerateImageCompleteResponse | GenerateImageCancelledResponse


class CancelCancellingResponse(BaseModel):
    status: Literal["cancelling"]
    id: str


class CancelNoActiveGenerationResponse(BaseModel):
    status: Literal["no_active_generation"]


CancelResponse: TypeAlias = CancelCancellingResponse | CancelNoActiveGenerationResponse


class RetakeVideoResponse(BaseModel):
    status: Literal["complete"]
    video_path: str


class RetakePayloadResponse(BaseModel):
    status: Literal["complete"]
    result: JsonObject


class RetakeCancelledResponse(BaseModel):
    status: Literal["cancelled"]


RetakeResponse: TypeAlias = RetakeVideoResponse | RetakePayloadResponse | RetakeCancelledResponse


class IcLoraExtractResponse(BaseModel):
    conditioning: str
    original: str
    conditioning_type: ConditioningType
    frame_time: float


class IcLoraGenerateCompleteResponse(BaseModel):
    status: Literal["complete"]
    video_path: str


class IcLoraGenerateCancelledResponse(BaseModel):
    status: Literal["cancelled"]


IcLoraGenerateResponse: TypeAlias = IcLoraGenerateCompleteResponse | IcLoraGenerateCancelledResponse


# ============================================================
# HuggingFace auth
# ============================================================


class HuggingFaceLoginResponse(BaseModel):
    client_id: str
    redirect_uri: str
    scope: str
    state: str
    code_challenge: str
    code_challenge_method: str


class HuggingFaceAuthStatusResponse(BaseModel):
    status: Literal["authenticated", "pending", "not_authenticated"]


class HuggingFaceLogoutResponse(BaseModel):
    status: Literal["logged_out"]


class ModelDownloadStartResponse(BaseModel):
    status: Literal["started"]
    message: str
    sessionId: str


class LtxDownloadRecommendationResponse(BaseModel):
    status: Literal["download"]
    cps_to_download: list[ModelCheckpointID]


class LtxUpgradeRecommendationResponse(BaseModel):
    status: Literal["upgrade"]
    ltx_model_id: LTXLocalModelId
    upgrade_message: str | None = None
    cps_to_download: list[ModelCheckpointID]
    cps_to_delete: list[ModelCheckpointID]


class LtxOkRecommendationResponse(BaseModel):
    status: Literal["ok"]


LtxRecommendationResponse: TypeAlias = (
    LtxDownloadRecommendationResponse | LtxUpgradeRecommendationResponse | LtxOkRecommendationResponse
)


class ImageGenRecommendationResponse(BaseModel):
    cp_to_download: ModelCheckpointID | None


class LtxIcLoraRecommendationResponse(BaseModel):
    cps_to_download: list[ModelCheckpointID]


class TextEncoderRecommendationResponse(BaseModel):
    cp_to_download: ModelCheckpointID | None
    expected_size_bytes: int
    expected_size_gb: float


class StatusResponse(BaseModel):
    status: str


class HTTPErrorResponse(BaseModel):
    code: str
    message: str


class LtxInsufficientFundsErrorResponse(BaseModel):
    code: Literal["LTX_INSUFFICIENT_FUNDS"]
    message: str


# ============================================================
# Request Models
# ============================================================


LTXVideoGenResolution: TypeAlias = Literal["540p", "720p", "1080p", "1440p", "2160p"]
LTXVideoGenDuration: TypeAlias = Literal[5, 6, 8, 10, 12, 14, 16, 18, 20]
LTXVideoGenFps: TypeAlias = Literal[24, 25, 48, 50]
LTXVideoGenPipeline: TypeAlias = Literal["fast", "pro"]


class LTXVideoGenerationResolutionSpec(BaseModel):
    fps_to_durations: dict[LTXVideoGenFps, list[LTXVideoGenDuration]]


class LTXVideoGenerationSpec(BaseModel):
    display_name: str
    supported_resolutions_durations: dict[LTXVideoGenResolution, LTXVideoGenerationResolutionSpec]
    a2v_supported_resolutions_durations: dict[LTXVideoGenResolution, LTXVideoGenerationResolutionSpec] | None = None


class LTXVideoGenerationModelSpecItem(BaseModel):
    pipeline: LTXVideoGenPipeline
    spec: LTXVideoGenerationSpec


class GenerateVideoModelsSpecsResponse(BaseModel):
    local_models: list[LTXVideoGenerationModelSpecItem]
    api_models: list[LTXVideoGenerationModelSpecItem]


# ============================================================
# Image generation model catalog (Z-Image + open-weight models)
# ============================================================
#
# Mirrors the video model-specs pattern: a backend-owned catalog of
# downloadable image models surfaced to the Gen Space image picker. Each entry
# carries enough metadata for the picker to render an honest state —
# downloaded / coming-soon / gated — and an "i" tooltip with license + size.
# `inference_status` separates "selectable + downloadable" from "actually
# runnable this pass": the new open-weight models are downloadable now, with
# inference wired as a follow-up (the handler returns 501 for coming-soon
# models so the UI never silently no-ops).

ImageModelInferenceStatus = Literal["available", "coming_soon"]


class ImageModelSpecApi(BaseModel):
    id: str
    display_name: str
    checkpoint_id: ModelCheckpointID
    # HuggingFace repo id (e.g. "krea/Krea-2-Turbo"). Surfaced so the picker
    # can probe per-repo access and open the repo page for gated models.
    repo_id: str
    description: str
    license: str
    # True when the HuggingFace repo is gated — the picker routes to the HF
    # auth flow before attempting the download.
    gated: bool
    inference_status: ImageModelInferenceStatus
    # Whether the checkpoint is present on disk (computed at request time).
    downloaded: bool
    default_resolution: tuple[int, int]
    supported_resolutions: list[tuple[int, int]]
    # Approximate download size (bytes), from the checkpoint spec.
    size_bytes: int
    # True for instruction-based editing models (e.g. FLUX.2 [klein] 9B) that
    # are served by the /api/generate-image-edit endpoint and accept input
    # reference images. The Gen Space picker shows an input-image affordance
    # for these and routes them to the edit endpoint. Edit models are not
    # required for basic text-to-image, so the first-run recommendation skips
    # them even when `inference_status == "available"`.
    is_edit_model: bool = False


class GenerateImageModelsSpecsResponse(BaseModel):
    models: list[ImageModelSpecApi]


class GenerateVideoRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    prompt: NonEmptyPrompt
    resolution: LTXVideoGenResolution = "1080p"
    model: LTXVideoGenPipeline = "fast"
    cameraMotion: VideoCameraMotion = "none"
    negativePrompt: str = ""
    duration: LTXVideoGenDuration = 5
    fps: LTXVideoGenFps = 24
    audio: bool = False
    imagePath: str | None = None
    audioPath: str | None = None
    aspectRatio: Literal["16:9", "9:16"] = "16:9"


class GenerateImageRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    prompt: NonEmptyPrompt
    width: int = Field(default=1024, ge=16)
    height: int = Field(default=1024, ge=16)
    numSteps: int = Field(default=4, ge=1)
    numImages: int = Field(default=1, ge=1)
    # Image model id from the image-model catalog (see
    # runtime_config/image_model_specs.py). None/unknown falls back to the
    # default image model (Z-Image Turbo). Coming-soon models are rejected by
    # the handler with 501 before any GPU work.
    model: str | None = None


class GenerateImageEditRequest(BaseModel):
    """Instruction-based image editing with FLUX.2 [klein] 9B (local).

    With no ``referenceImages`` the model behaves as text-to-image; with one or
    more reference images it performs single/multi-reference instruction editing
    (up to 4 references). Reference paths are local file paths resolved by the
    backend (same convention as the LoRA frame-edit flow).
    """

    model_config = ConfigDict(strict=True)

    prompt: NonEmptyPrompt
    width: int = Field(default=1024, ge=16)
    height: int = Field(default=1024, ge=16)
    numSteps: int = Field(default=4, ge=1)
    numImages: int = Field(default=1, ge=1)
    referenceImages: list[str] = Field(default_factory=list)


class GenerateImageEditCompleteResponse(BaseModel):
    status: Literal["complete"]
    image_paths: list[str]


class GenerateImageEditCancelledResponse(BaseModel):
    status: Literal["cancelled"]


GenerateImageEditResponse: TypeAlias = GenerateImageEditCompleteResponse | GenerateImageEditCancelledResponse


# ============================================================
# LoRA inference generate (Gen Space "Apply LoRA" → generate)
# ============================================================
#
# Discriminated by `variant` so the handler routes to the right pipeline:
#   - standard            : user-trained t2v/i2v LoRA → fast video pipeline
#                           (lora_path/lora_scale built into the DistilledPipeline)
#   - union_control       : official LTX-2 IC-LoRA union adapter, control-signal
#                           conditioned (canny/depth/pose) → IC-LoRA pipeline
#   - video_input_ic_lora : user-trained IC-LoRA conditioned on a reference video
#                           (no control-signal preprocessing) → IC-LoRA pipeline
# `loraId` resolves through the inference registry to an on-disk adapter path
# (for standard / video_input_ic_lora); union_control reuses the official union
# checkpoint resolved inside the IC-LoRA handler from `conditioning_type`.
# Defined above the queue section so `LoraQueuePayload` is in scope for the
# `QueuePayloadApi` discriminated union (the union RHS evaluates at runtime,
# unlike annotations, so ordering matters).


class LoraStandardGenerateRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    variant: Literal["standard"] = "standard"
    loraId: str
    loraScale: float = Field(default=1.0, ge=0.0, le=2.0)
    # The full video-generation spec (prompt, resolution, duration, fps, image
    # for i2v, audio, aspect ratio, camera motion). Reusing GenerateVideoRequest
    # keeps the LoRA path on the same validation + dimension resolution as a
    # plain Generate — a standard LoRA is just "Generate Video + adapter".
    request: GenerateVideoRequest


class LoraUnionControlGenerateRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    variant: Literal["union_control"] = "union_control"
    loraId: str
    # The IC-LoRA generation spec (reference video, conditioning_type,
    # conditioning_strength, prompt, image inputs). conditioning_type drives
    # which union adapter + processor is loaded (canny/depth/pose).
    request: IcLoraGenerateRequest


class LoraVideoInputIcLoraGenerateRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    variant: Literal["video_input_ic_lora"] = "video_input_ic_lora"
    loraId: str
    loraScale: float = Field(default=1.0, ge=0.0, le=2.0)
    prompt: NonEmptyPrompt
    # The reference video the user-trained IC-LoRA is conditioned on. Unlike
    # union_control, no control-signal preprocessing runs — the reference video
    # feeds the pipeline's video_conditioning channel directly.
    videoPath: str
    conditioningStrength: float = 1.0
    negativePrompt: str = ""
    # Optional output duration override (seconds). See `IcLoraGenerateRequest`.
    duration: int | None = None
    # First-pass (stage-1) resolution bucket. See `IcLoraGenerateRequest`.
    resolution: IcLoraResolutionApi = "540p"
    # When true, mux the reference video's audio onto the output. See
    # `IcLoraGenerateRequest.preserve_audio`.
    preserveAudio: bool = False
    # When true, run the 2x spatial upsample + refine pass (stage 2). Defaults
    # to false (single high-res stage) to preserve identity — see
    # `IcLoraGenerateRequest.refine`.
    refine: bool = False


LoraGenerateRequest: TypeAlias = Annotated[
    LoraStandardGenerateRequest
    | LoraUnionControlGenerateRequest
    | LoraVideoInputIcLoraGenerateRequest,
    Field(discriminator="variant"),
]


class LoraGenerateCompleteResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    status: Literal["complete"] = "complete"
    videoPath: str


class LoraGenerateCancelledResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    status: Literal["cancelled"] = "cancelled"


LoraGenerateResponse: TypeAlias = Annotated[
    LoraGenerateCompleteResponse | LoraGenerateCancelledResponse,
    Field(discriminator="status"),
]


class LoraQueuePayload(BaseModel):
    model_config = ConfigDict(strict=True)

    kind: Literal["lora"] = "lora"
    request: LoraGenerateRequest


# ============================================================
# Generation queue (durable, unified video + image ledger)
# ============================================================
#
# The queue is a persistent batch generation ledger (see
# `state/queue_state.py` + `handlers/queue_handler.py`). Each item carries
# a self-contained, discriminated `payload` snapshot so the runner can
# dispatch to the right generation handler (video or image) on claim
# without re-reading any shared, mutable params. The discriminator is the
# literal `kind` field; pydantic routes parsing off it so an
# image-typed `request` landed under `kind="video"` is rejected at the
# boundary rather than mid-render.

QueueItemKindApi: TypeAlias = Literal["video", "image", "image_edit", "lora"]


class VideoQueuePayload(BaseModel):
    model_config = ConfigDict(strict=True)

    kind: Literal["video"] = "video"
    request: GenerateVideoRequest


class ImageQueuePayload(BaseModel):
    model_config = ConfigDict(strict=True)

    kind: Literal["image"] = "image"
    request: GenerateImageRequest


class ImageEditQueuePayload(BaseModel):
    """Queue payload for FLUX.2 [klein] 9B image editing (local GPU).

    Routes through the same durable queue as video/image generations so a
    Klein edit doesn't block the UI (it shows in the queue panel and the
    runner cooperates with the single-flight GPU slot instead of failing
    "already in progress"). Carries the same `GenerateImageEditRequest`
    the synchronous `/api/generate-image-edit` endpoint takes.
    """

    model_config = ConfigDict(strict=True)

    kind: Literal["image_edit"] = "image_edit"
    request: GenerateImageEditRequest


# Discriminated union: pydantic picks the variant by `kind`, so the
# request body is type-narrowed on the way in and the runner can pattern
# match on the same field on the way out.
QueuePayloadApi: TypeAlias = Annotated[
    VideoQueuePayload | ImageQueuePayload | ImageEditQueuePayload | LoraQueuePayload,
    Field(discriminator="kind"),
]

QueueItemStatusApi: TypeAlias = Literal[
    "pending", "running", "completed", "failed", "cancelled"
]

QueueItemSourceApi: TypeAlias = Literal[
    "genspace", "queue_manual", "gemini_brainstorm"
]


class QueueItemApi(BaseModel):
    model_config = ConfigDict(strict=True)

    id: str
    status: QueueItemStatusApi
    createdAt: str
    startedAt: str | None = None
    completedAt: str | None = None
    originatingProjectId: str | None = None
    payload: QueuePayloadApi
    outputPath: str | None = None
    error: str | None = None
    retryCount: int = 0
    source: QueueItemSourceApi


class QueueStateResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    items: list[QueueItemApi]
    paused: bool
    schemaVersion: int


class EnqueueQueueItemRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    payload: QueuePayloadApi
    originatingProjectId: str | None = None
    source: QueueItemSourceApi = "genspace"


class EnqueueQueueBatchRequest(BaseModel):
    """Batch enqueue body. The queue authoring flows (manual multi-prompt
    entry, brainstorm-with-auto-enqueue) send a single round-trip rather
    than N enqueues; one persistence write per batch keeps `queue.json`
    cheap when landing 20-50 items at once."""

    model_config = ConfigDict(strict=True)

    items: list[EnqueueQueueItemRequest] = Field(min_length=1)


class UpdateQueueItemRequest(BaseModel):
    """Replace a pending item's payload. Only valid while `pending` —
    once the runner has claimed the item the snapshot is in flight and
    editing would be racy, so the backend rejects with 409. The body
    carries the full discriminated payload rather than a patch so we
    don't need a partial-update schema per field."""

    model_config = ConfigDict(strict=True)

    payload: QueuePayloadApi


class ReorderQueueRequest(BaseModel):
    """Full permutation of currently-pending item ids. Permutation
    semantics (rather than swap/move) mean the frontend's drag-and-drop
    produces one new ordering that we validate atomically — no
    intermediate states to reason about."""

    model_config = ConfigDict(strict=True)

    itemIds: list[str] = Field(min_length=0)


class ClearQueueResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    cleared: int


def _default_model_types() -> set[ModelCheckpointID]:
    return set()


class ModelDownloadRequest(BaseModel):
    type: Literal["download", "upgrade"] = "download"
    cp_ids: set[ModelCheckpointID] = Field(default_factory=_default_model_types)


ModelAccessStatus: TypeAlias = Literal["authorized", "not_authorized"]


class CheckModelAccessRequest(BaseModel):
    cp_ids: set[ModelCheckpointID] = Field(default_factory=_default_model_types)


class CheckModelAccessResponse(BaseModel):
    access: dict[str, ModelAccessStatus]


class ModelDeleteRequest(BaseModel):
    cp_ids: set[ModelCheckpointID] = Field(default_factory=_default_model_types)


class CheckpointPathResponse(BaseModel):
    """Resolved on-disk location for a checkpoint (used by "Reveal in Explorer").

    `path` is the absolute path the app expects the checkpoint at; `exists`
    is whether it's present (downloaded/linked) there.
    """

    model_config = ConfigDict(strict=True)

    cp_id: ModelCheckpointID
    path: str
    exists: bool


class LoadModelFromPathRequest(BaseModel):
    """Point the app at an already-downloaded model on disk ("Load from location").

    `source_path` is a user-chosen folder/file containing the checkpoint; the
    backend links (symlink/junction) or copies it into the expected models-dir
    location so `is_cp_downloaded` flips without a re-download.
    """

    model_config = ConfigDict(strict=True)

    cp_id: ModelCheckpointID
    sourcePath: str = Field(min_length=1)


class LoadModelFromPathResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    cp_id: ModelCheckpointID
    path: str
    # "linked" (symlink/junction) or "copied" — copied can be slow for large
    # models, so the UI surfaces it.
    method: Literal["linked", "copied"]


GapPromptMode: TypeAlias = Literal["text-to-video", "image-to-video", "text-to-image"]


class SuggestGapPromptRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    beforePrompt: str = ""
    afterPrompt: str = ""
    beforeFrame: str | None = None
    afterFrame: str | None = None
    gapDuration: float = 5
    mode: GapPromptMode = "text-to-video"
    inputImage: str | None = None

    @model_validator(mode="after")
    def _validate_input_image_mode(self) -> "SuggestGapPromptRequest":
        if self.inputImage is not None and self.mode != "image-to-video":
            raise ValueError("inputImage is only valid for image-to-video mode")
        return self


RetakeMode: TypeAlias = Literal["replace_audio_and_video", "replace_video", "replace_audio"]


class RetakeRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    video_path: str
    start_time: float
    duration: float
    prompt: str = ""
    mode: RetakeMode = "replace_audio_and_video"


ConditioningType: TypeAlias = Literal["canny", "depth", "pose"]

# Legacy alias kept for the LoRA inference registry surface; it now carries the
# same set as `ConditioningType` since the IC-LoRA path supports pose (B2b).
ControlConditioningType: TypeAlias = ConditioningType


# ============================================================
# LoRA inference registry (in-app LoRA testing — Gen Space)
# ============================================================
#
# The registry lists every LoRA usable from Gen Space:
#   - the official LTX-2 IC-LoRA union-control adapter (canny/depth/pose), and
#   - user-trained adapters from completed training jobs (`TrainingJob.local_lora_path`).
# Each entry carries a `variant` that drives the discriminated generate request
# in B2: "union_control" (official, control-signal conditioned), "standard"
# (user-trained text/image-to-video LoRA), or "video_input_ic_lora" (a
# user-trained IC-LoRA conditioned on a reference video rather than a control signal).
LoraInferenceVariantApi: TypeAlias = Literal[
    "standard", "union_control", "video_input_ic_lora"
]
LoraInferenceKindApi: TypeAlias = Literal["official_union", "user_trained", "imported"]

# Variants a user can tag on an *imported* LoRA at import time. The official
# union_control checkpoint is never importable (it's a known LTX-2 adapter
# fetched through the model-download flow), so the picker offers only the two
# variants that route through a user-supplied weights file: a standard
# text/image-to-video style adapter, or a reference-video IC-LoRA.
ImportedLoraVariantApi: TypeAlias = Literal["standard", "video_input_ic_lora"]


class LoraInferenceEntryApi(BaseModel):
    model_config = ConfigDict(strict=True)

    id: str
    kind: LoraInferenceKindApi
    variant: LoraInferenceVariantApi
    name: str
    # Control-signal types this entry accepts. Only `union_control` populates
    # this (canny/depth/pose); `standard` and `video_input_ic_lora` are empty
    # (their conditioning is the prompt / a reference video, not a control signal).
    conditioningTypes: list[ControlConditioningType]
    # Absolute path to the adapter on disk, or None if not yet downloaded.
    # `available` is the flag to gate the UI; `localPath` lets the generate
    # handler skip a re-resolution.
    localPath: str | None = None
    available: bool
    # For `user_trained` entries: the training job id that produced the adapter.
    sourceTrainingId: str | None = None
    description: str | None = None
    # When the entry was created (import time / training start), ISO 8601 UTC.
    # None for the official adapter and legacy records that predate the field.
    createdAt: str | None = None
    # Size of the adapter weights on disk in bytes, when known. Surfaced in the
    # Library so the user can see how big each LoRA is.
    fileSizeBytes: int | None = None
    # HuggingFace model-card URL the LoRA was imported from / profiled against,
    # so the user can get back to the source. None for trained + official.
    huggingfaceUrl: str | None = None
    # Trigger word the adapter was trained with (best-effort default derived
    # from the LoRA name when the user hasn't set one). The auto-prompt
    # assistant relies on this to activate the adapter.
    triggerWord: str | None = None
    # Per-LoRA system prompt the Gemini auto-prompt assistant uses to write a
    # tailored text-to-video prompt from the reference video. The app
    # auto-generates a sensible default per entry; the user can edit it in a
    # per-LoRA modal. None only when a variant has no meaningful template
    # (e.g. a standard style LoRA, which has no reference video to prompt from).
    promptTemplate: str | None = None
    # True only when promptTemplate comes from a persisted user override.
    # Generated defaults can be safely rebuilt when metadata changes.
    promptTemplateCustomized: bool = False
    # An optional user-supplied example (image or video) showing what the LoRA
    # does, rendered as the card thumbnail in the Library (CivitAI-style) and
    # previewed in the detail panel. `exampleMediaType` is the media kind
    # ("image" / "video"); None when no example is attached. The bytes are
    # served via the secure example-media route, so no filesystem path is
    # exposed to the client.
    exampleMediaType: Literal["image", "video"] | None = None


class LoraInferenceRegistryResponseApi(BaseModel):
    model_config = ConfigDict(strict=True)

    entries: list[LoraInferenceEntryApi]


# ------------------------------------------------------------
# Imported LoRA library (user-supplied adapter weights)
# ------------------------------------------------------------
#
# Lets a user bring in a LoRA they got from outside the app (a downloaded
# `.safetensors` / `.pt`) and use it from Gen Space. The backend copies the
# file into app storage so the import survives the source moving/disappearing,
# and tags it with the variant the user picked so it routes through the same
# generate flow as a trained adapter. `sourcePath` is an absolute path the
# Electron file dialog produced; the response is the resulting registry entry.
class ImportLoraRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    sourcePath: str
    name: str
    variant: ImportedLoraVariantApi
    description: str | None = None
    # The trigger word the adapter was trained with. Drives the auto-generated
    # system prompt (so it activates the LoRA). Optional, but never inferred
    # from the display name. If a profile is derived (built-in / HF / example),
    # the profile's verified trigger wins over this field.
    triggerWord: str | None = None
    # Optional HuggingFace model-card URL. When provided, the backend fetches the
    # card markdown and asks Gemini to configure an accurate per-LoRA system
    # prompt + trigger word (so the LoRA actually activates instead of silently
    # no-op'ing from a prompt-structure mismatch). Best-effort; failures fall
    # back to the name-derived default.
    huggingfaceUrl: str | None = None
    # Optional example prompt the LoRA was trained on — the universal fallback
    # for LoRAs with no HuggingFace page (Civitai / direct files / Discord).
    # Fed to the same Gemini meta-prompt to derive the system prompt + trigger.
    examplePrompt: str | None = None


class ImportLoraResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    entry: LoraInferenceEntryApi
    # Outcome of the per-LoRA prompt profiling step (built-in profile / Gemini
    # from the HuggingFace card or example prompt). Surfaced so the import
    # modal can tell the user whether the system prompt was auto-configured or
    # why it wasn't — previously this was entirely silent. "skipped" is normal
    # (no source provided); "failed" carries a user-facing reason in
    # ``profileMessage``.
    profileStatus: Literal["builtin", "configured", "skipped", "failed"] = "skipped"
    profileMessage: str | None = None


class UpdateImportedLoraRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    # Any subset of fields may be patched. `name` is validated non-blank when
    # present; at least one field must be supplied.
    name: str | None = None
    description: str | None = None
    huggingfaceUrl: str | None = None


class UpdateImportedLoraResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    entry: LoraInferenceEntryApi


class UpdateTrainedLoraRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    # Editable metadata for a user-trained LoRA (backed by the training job).
    # `name` is validated non-blank when present; at least one field must be set.
    name: str | None = None
    description: str | None = None


class UpdateTrainedLoraResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    entry: LoraInferenceEntryApi


class ReprofileImportedLoraRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    # Optional HuggingFace card URL / example prompt to (re)derive the per-LoRA
    # system prompt + trigger word for an already-imported LoRA.
    huggingfaceUrl: str | None = None
    examplePrompt: str | None = None


class SetLoraExampleRequest(BaseModel):
    """Attach a user-supplied example image/video to an imported or trained LoRA.

    `sourcePath` is an absolute path from the Electron file dialog; the backend
    copies it into app storage so the example survives the source moving, and
    infers the media kind from the extension. Replaces any existing example.
    """

    model_config = ConfigDict(strict=True)

    sourcePath: str


class SetLoraExampleResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    entry: LoraInferenceEntryApi


class ReprofileImportedLoraResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    entry: LoraInferenceEntryApi
    profileStatus: Literal["builtin", "configured", "skipped", "failed"] = "skipped"
    profileMessage: str | None = None


# ------------------------------------------------------------
# Per-LoRA prompt-writing assistant (Gemini Flash)
# ------------------------------------------------------------
#
# The auto-prompt flow lets the user click a "sparkle" button in the prompt bar
# to have Gemini Flash watch the reference video and write a tailored
# text-to-video prompt using the LoRA's per-LoRA system prompt (`promptTemplate`
# on the registry entry). The template is auto-generated per LoRA and editable
# via the update endpoint; the auto-prompt endpoint is read-only and stateless.
class AutoPromptRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    loraId: str
    videoPath: str


class AutoPromptResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    prompt: str


class UpdateLoraPromptTemplateRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    # Optional — passing null/None resets to the auto-generated default.
    promptTemplate: str | None = None
    triggerWord: str | None = None


class UpdateLoraPromptTemplateResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    entry: LoraInferenceEntryApi


class IcLoraExtractRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    video_path: str
    conditioning_type: ConditioningType = "canny"
    frame_time: float = 0


class IcLoraImageInput(BaseModel):
    model_config = ConfigDict(strict=True)

    path: str
    frame: int = 0
    strength: float = 1.0


# IC-LoRA first-pass (stage-1) resolution. The adapter is trained at the 540p
# bucket (~960x576); 720p / 1080p generate off that distribution (sharper but
# risking identity drift) and cost more VRAM. Buckets are 64-multiples so the
# two-stage patchify splits the latent evenly. `refine` (Stage X2) doubles
# whichever bucket is chosen. Defaults to "540p" (the on-distribution
# bucket) so existing behavior is unchanged.
IcLoraResolutionApi: TypeAlias = Literal["540p", "720p", "1080p"]


def _default_ic_lora_images() -> list[IcLoraImageInput]:
    return []


class IcLoraGenerateRequest(BaseModel):
    model_config = ConfigDict(strict=True)
    video_path: str
    conditioning_type: ConditioningType
    prompt: NonEmptyPrompt
    conditioning_strength: float = 1.0
    num_inference_steps: int = 30
    cfg_guidance_scale: float = 1.0
    negative_prompt: str = ""
    images: list[IcLoraImageInput] = Field(default_factory=_default_ic_lora_images)
    # Optional output duration override (seconds, one of the supported IC-LoRA
    # durations). When omitted the duration is snapped from the reference video.
    # The reference is resampled / freeze-padded to match, so this is a free
    # user choice within the supported set — it is what gives the Gen Space UI
    # length control over an IC-LoRA generation (AR + resolution stay
    # reference-derived to keep the adapter on its training distribution).
    duration: int | None = None
    # First-pass (stage-1) resolution bucket. Defaults to "540p" (the adapter's
    # training bucket). "720p" / "1080p" generate off-distribution — sharper but
    # risking identity drift — and cost more VRAM. `refine` (Stage X2) doubles
    # whichever bucket is chosen. See `IcLoraHandler._canonical_shape`.
    resolution: IcLoraResolutionApi = "540p"
    # When true, the reference video's audio track is muxed onto the generated
    # output (trimmed to the output length). No-op if the reference has no
    # audio. The generation itself is video-only; this is a post step.
    preserve_audio: bool = False
    # When true, run the 2x spatial upsample + refine pass (stage 2) after the
    # initial generation. Defaults to false (single high-res stage) because the
    # refine stage trades identity fidelity for sharpness — matching the
    # ComfyUI flow where stage 2 is bypassed. The pipeline is always called with
    # a 2x target resolution so stage 1 lands at the bucket; this flag decides
    # whether the output is the stage-1 result (false) or the 2x refined result
    # (true). See `IcLoraHandler._canonical_shape`.
    refine: bool = False


# ============================================================
# LoRA trainer (camelCase API surface)
# ============================================================
#
# These models mirror the snake_case persistence shapes in
# `state/lora_training_state.py` but expose camelCase fields to match
# the rest of the HTTP API. Routes convert at the boundary via the
# `*_to_api` helpers in that module. The feature drives the official
# LTX-2 trainer scripts on a remote RunPod GPU,
# modeled as three durable ledgers: datasets, preprocessed datasets,
# and training jobs. See `state/lora_training_state.py` for the state
# machines and `handlers/lora_training_handler.py` for the rules.

LoraDatasetStatusApi: TypeAlias = Literal[
    "draft",
    "uploading",
    "uploaded",
    "upload_failed",
    "cancelled",
    "gpu_selection_required",
]
LoraPreprocessStatusApi: TypeAlias = Literal[
    "pending", "captioning", "preprocessing", "ready", "failed", "cancelled"
]
LoraTrainingStatusApi: TypeAlias = Literal[
    "pending",
    "running",
    "completed",
    "failed",
    "cancelled",
    "gpu_selection_required",
]
LoraProviderApi: TypeAlias = Literal["runpod", "local"]
LoraWorkspacePolicyApi: TypeAlias = Literal[
    "primary_cache", "ephemeral_any_region"
]
LoraCaptionerTypeApi: TypeAlias = Literal["qwen_omni", "gemini_flash"]
LoraTrainingPresetApi: TypeAlias = Literal["standard", "low_vram"]
# Collection type: `standard` LoRA (video modality, optionally + audio) vs
# `ic_lora` (In-Context LoRA: reference -> target transformations).
LoraDatasetTypeApi: TypeAlias = Literal["standard", "ic_lora"]


class LoraTargetHandleApi(BaseModel):
    model_config = ConfigDict(strict=True)

    provider: LoraProviderApi
    podId: str | None = None
    remoteJobId: str | None = None


LoraClipOriginApi: TypeAlias = Literal["imported", "gen_space", "ai_derived"]

# Manual curation triage flag. ``None`` = unreviewed; "keep"/"reject" let the
# user mark picks while scrubbing a fresh import, then filter to kept clips
# before training. Rejected clips are excluded from readiness counts.
# `holdout` reserves a clip for the in-training validation feed (excluded from
# training, reference video staged for IC-LoRA validation).
LoraClipTriageApi: TypeAlias = Literal["keep", "reject", "holdout"]


class LoraTrimApi(BaseModel):
    """Keep only [startSeconds, endSeconds) of the source timeline."""

    model_config = ConfigDict(strict=True)

    startSeconds: float = Field(ge=0.0)
    endSeconds: float = Field(gt=0.0)


class LoraCropApi(BaseModel):
    """Pixel crop rectangle, applied after trim."""

    model_config = ConfigDict(strict=True)

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class LoraScaleApi(BaseModel):
    """Target resolution, applied after crop (bucket-snap / normalize)."""

    model_config = ConfigDict(strict=True)

    width: int = Field(gt=0)
    height: int = Field(gt=0)


class LoraClipEditsApi(BaseModel):
    """Non-destructive edit stack for a clip.

    Applied in order: trim → crop → scale → fps → reverse → speed (audio
    mirrors video unless muted). Stored on the clip so the editor can
    re-open and re-render from the untouched source. A null/false field
    means that step is skipped.
    """

    model_config = ConfigDict(strict=True)

    trim: LoraTrimApi | None = None
    crop: LoraCropApi | None = None
    scale: LoraScaleApi | None = None
    fps: float | None = Field(default=None, gt=0.0)
    speed: float | None = Field(default=None, gt=0.0)
    mute: bool = False
    reverse: bool = False


class LoraClipProbeApi(BaseModel):
    """Measured facts about a clip (from the desktop ffmpeg probe)."""

    model_config = ConfigDict(strict=True)

    durationSeconds: float
    width: int
    height: int
    fps: float
    frameCount: int
    hasAudio: bool
    videoCodec: str | None = None


class LoraDatasetClipApi(BaseModel):
    model_config = ConfigDict(strict=True)

    id: str
    localPath: str
    caption: str = ""
    durationSeconds: float | None = None
    # Primary conditioning reference (single). `referencePaths` carries all
    # references for a manually-grouped IC-LoRA set (one target, N references).
    referencePath: str | None = None
    referencePaths: list[str] = Field(default_factory=list)
    # Provenance of the clip so the UI can badge AI-derived clips and
    # filter project-imported ones. Defaults keep old ledgers valid.
    origin: LoraClipOriginApi = "imported"
    # Cached probe; null until the desktop has measured the clip.
    probe: LoraClipProbeApi | None = None
    # Untouched original (set once the clip has been edited); `localPath`
    # then points at the rendered derivative that gets uploaded.
    sourcePath: str | None = None
    # The applied non-destructive edit stack (null = pristine clip).
    edits: LoraClipEditsApi | None = None
    # Curation preview assets generated by a local clip-job. `posterPath`
    # is a single representative frame; `spritePath` is a horizontal
    # filmstrip montage used for hover-scrub. Null until generated.
    posterPath: str | None = None
    spritePath: str | None = None
    spriteTiles: int | None = None
    # Manual keep/reject curation flag (null = unreviewed).
    triage: LoraClipTriageApi | None = None
    # Soft-delete timestamp (ISO-8601). When set the clip is in the dataset's
    # recycle bin: hidden from the gallery and excluded from pairing, readiness,
    # training and export until restored or permanently deleted.
    deletedAt: str | None = None


class RunpodSelectionApi(BaseModel):
    """Explicit compute/storage choice snapshotted for one RunPod run."""

    model_config = ConfigDict(strict=True)

    gpuType: str = Field(min_length=1)
    gpuVramGb: int = Field(default=0, ge=0)
    datacenter: str = ""
    workspacePolicy: LoraWorkspacePolicyApi = "ephemeral_any_region"
    volumeId: str | None = None

    @model_validator(mode="after")
    def _validate_volume_policy(self) -> "RunpodSelectionApi":
        if self.workspacePolicy == "primary_cache" and not self.volumeId:
            raise ValueError("volumeId is required for primary_cache")
        if self.workspacePolicy == "ephemeral_any_region" and self.volumeId is not None:
            raise ValueError("volumeId is only valid for primary_cache")
        return self


class LoraDatasetApi(BaseModel):
    model_config = ConfigDict(strict=True)

    id: str
    name: str
    createdAt: str
    status: LoraDatasetStatusApi
    type: LoraDatasetTypeApi = "standard"
    workspacePolicy: LoraWorkspacePolicyApi = "primary_cache"
    cacheVolumeId: str | None = None
    triggerWord: str | None = None
    clips: list[LoraDatasetClipApi]
    remoteDatasetDir: str | None = None
    target: LoraTargetHandleApi | None = None
    runpodSelection: RunpodSelectionApi | None = None
    error: str | None = None
    statusDetail: str | None = None
    statusPercent: int | None = None
    statusEtaSeconds: int | None = None
    cancelRequested: bool = False
    folderId: str | None = None
    updatedAt: str | None = None
    # Project the dataset was started from (e.g. clips dragged in from a
    # project's Gen Space). Purely informational — LoRAs stay global and
    # usable across projects — but lets the UI show provenance / jump back.
    originatingProjectId: str | None = None
    # Present while a one-click pipeline is waiting for replacement GPU stock.
    # Lets recovery UI edit the actual persisted intent instead of showing
    # freshly initialized defaults.
    pendingPipeline: LoraPendingPipelineApi | None = None
    archivedAt: str | None = None
    keepAliveUntil: str | None = None
    finalActivityAt: str | None = None
    releaseStatus: Literal["scheduled", "releasing", "released", "failed"] | None = None
    releaseError: str | None = None
    releaseAttemptedAt: str | None = None
    lastPodId: str | None = None


class LoraFolderApi(BaseModel):
    model_config = ConfigDict(strict=True)

    id: str
    name: str
    parentId: str | None = None


class LoraDatasetsResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    datasets: list[LoraDatasetApi]
    folders: list[LoraFolderApi]
    schemaVersion: int


class LoraPreprocessedApi(BaseModel):
    model_config = ConfigDict(strict=True)

    id: str
    datasetId: str
    createdAt: str
    status: LoraPreprocessStatusApi
    resolutionBuckets: str
    effectiveResolutionBuckets: str | None = None
    withAudio: bool
    autoCaption: bool
    captionerType: LoraCaptionerTypeApi
    remotePrecomputedDir: str | None = None
    target: LoraTargetHandleApi | None = None
    error: str | None = None
    startedAt: str | None = None
    completedAt: str | None = None
    cancelRequested: bool = False
    statusDetail: str | None = None


class LoraPreprocessedResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    items: list[LoraPreprocessedApi]
    schemaVersion: int


class LoraTrainingConfigApi(BaseModel):
    model_config = ConfigDict(strict=True)

    preset: LoraTrainingPresetApi = "standard"
    # LoRA adapter
    rank: int = 32
    alpha: int = 32
    dropout: float = 0.0
    targetModules: list[str] = Field(
        default_factory=lambda: ["to_k", "to_q", "to_v", "to_out.0"]
    )
    # Optimization
    learningRate: float = 1e-4
    steps: int = 2000
    batchSize: int = 1
    gradientAccumulationSteps: int = 1
    maxGradNorm: float = 1.0
    optimizerType: str | None = None
    schedulerType: str = "linear"
    enableGradientCheckpointing: bool = True
    # Acceleration (null -> preset-derived)
    mixedPrecisionMode: str = "bf16"
    quantization: str | None = None
    loadTextEncoderIn8bit: bool | None = None
    offloadOptimizerDuringValidation: bool | None = None
    # Data
    numDataloaderWorkers: int = 2
    # Strategy override (null -> dataset-type default)
    firstFrameConditioningP: float | None = None
    # Validation
    validationPrompts: list[str] = Field(default_factory=lambda: [DEFAULT_VALIDATION_PROMPT])
    validationNegativePrompt: str = (
        "worst quality, inconsistent motion, blurry, jittery, distorted"
    )
    validationVideoWidth: int = 576
    validationVideoHeight: int = 576
    validationVideoFrames: int = 49
    validationFrameRate: float = 25.0
    validationInferenceSteps: int = 30
    validationInterval: int = 250
    validationGuidanceScale: float = 4.0
    validationSeed: int = 42
    stgScale: float = 1.0
    stgBlocks: list[int] = Field(default_factory=lambda: [29])
    stgMode: str = "stg_av"
    skipInitialValidation: bool | None = None
    # Checkpoints
    checkpointInterval: int = 250
    checkpointKeepLastN: int = 3
    checkpointPrecision: str = "bfloat16"
    # Flow matching (advanced)
    timestepSamplingMode: str = "shifted_logit_normal"
    # Hub / tracking (advanced)
    pushToHub: bool = False
    hubModelId: str | None = None
    wandbEnabled: bool = False
    # Misc
    seed: int = 42
    loadCheckpoint: str | None = None
    # Dataset/preprocessing-driven (not surfaced in the profile editor)
    withAudio: bool = False
    triggerWord: str | None = None


class LoraPendingPipelineApi(BaseModel):
    model_config = ConfigDict(strict=True)

    name: str
    description: str | None = None
    resolutionBuckets: str
    withAudio: bool
    autoCaption: bool
    captionerType: LoraCaptionerTypeApi
    config: LoraTrainingConfigApi


# LoraDatasetApi is declared before the training config types to keep the
# existing API module organization; resolve its recovery payload now.
LoraDatasetApi.model_rebuild()


class LoraTrainingProfileApi(BaseModel):
    model_config = ConfigDict(strict=True)

    id: str
    name: str
    createdAt: str
    updatedAt: str
    config: LoraTrainingConfigApi
    builtin: bool = False
    description: str = ""
    datasetTypes: list[LoraDatasetTypeApi] = Field(
        default_factory=lambda: ["standard", "ic_lora"], min_length=1
    )
    minVramGb: int | None = Field(default=None, ge=1)
    autoRecommended: bool = False


class LoraTrainingProfilesResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    profiles: list[LoraTrainingProfileApi]
    schemaVersion: int


class LoraValidationFeedItemApi(BaseModel):
    """One generated validation sample in the training-results feed."""

    model_config = ConfigDict(strict=True)

    step: int
    sampleIndex: int
    # Browser-loadable URL of the downloaded sample media (served via the
    # secure feed-media route, not a raw filesystem path).
    mediaUrl: str
    mediaType: Literal["video", "audio"] = "video"
    source: Literal["prompt", "holdout"] = "prompt"
    prompt: str = ""
    # Browser-loadable URL of the held-out reference video (IC-LoRA), when staged.
    referenceMediaUrl: str | None = None
    createdAt: str = ""


class LoraCheckpointArtifactApi(BaseModel):
    """One adapter checkpoint downloaded from the remote run output.

    Paired with validation feed entries by `step`. `localPath` is the
    downloaded file; the frontend's Reveal action opens its folder in the OS
    file manager via `window.electronAPI.showItemInFolder`.
    """

    model_config = ConfigDict(strict=True)

    step: int
    localPath: str
    createdAt: str = ""


class LoraGpuStatusApi(BaseModel):
    """Live GPU telemetry snapshot for the GPU-status panel."""

    model_config = ConfigDict(strict=True)

    name: str
    vramTotalMb: int
    vramUsedMb: int
    gpuUtilPct: int
    memUtilPct: int
    tempC: int | None = None
    updatedAt: str


class LoraTrainingJobApi(BaseModel):
    model_config = ConfigDict(strict=True)

    id: str
    preprocessedId: str
    name: str
    createdAt: str
    status: LoraTrainingStatusApi
    config: LoraTrainingConfigApi
    provider: LoraProviderApi
    # Optional user-edited description shown in the LoRA Library.
    description: str | None = None
    remoteOutputDir: str | None = None
    localLoraPath: str | None = None
    target: LoraTargetHandleApi | None = None
    currentStep: int | None = None
    totalSteps: int | None = None
    etaSeconds: int | None = None
    gpuType: str = ""
    gpuVramGb: int = 0
    runpodSelection: RunpodSelectionApi | None = None
    firstStepAt: str | None = None
    error: str | None = None
    startedAt: str | None = None
    completedAt: str | None = None
    computeRatePerHr: float | None = Field(default=None, ge=0)
    archivedAt: str | None = None
    workloadBillingStartedAt: str | None = None
    workloadBillingEndedAt: str | None = None
    capturedHourlyRate: float | None = Field(default=None, ge=0)
    attributedSeconds: float | None = Field(default=None, ge=0)
    attributedCost: float | None = Field(default=None, ge=0)
    podPreparationStartedAt: str | None = None
    podPreparationEndedAt: str | None = None
    trainingSetupStartedAt: str | None = None
    trainingSetupEndedAt: str | None = None
    trainingStepsStartedAt: str | None = None
    trainingStepsEndedAt: str | None = None
    lastPodId: str | None = None
    cancelRequested: bool = False
    statusDetail: str | None = None
    # Training-results feed (validation samples streamed back as they train)
    # and live GPU telemetry for the run's GPU.
    validationFeed: list[LoraValidationFeedItemApi] = Field(default_factory=list[LoraValidationFeedItemApi])
    # Adapter checkpoints downloaded live, paired with the validation feed by
    # step. Each carries a local path the frontend can reveal in the OS file
    # manager.
    checkpoints: list[LoraCheckpointArtifactApi] = Field(default_factory=list[LoraCheckpointArtifactApi])
    gpuStatus: LoraGpuStatusApi | None = None


class LoraTrainingResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    items: list[LoraTrainingJobApi]
    schemaVersion: int


# ---- Request bodies ----


class LoraClipInput(BaseModel):
    model_config = ConfigDict(strict=True)

    id: str | None = Field(default=None, min_length=1, max_length=128)
    localPath: str = Field(min_length=1)
    caption: str = ""
    durationSeconds: float | None = None
    referencePath: str | None = None
    referencePaths: list[str] = Field(default_factory=list)
    origin: LoraClipOriginApi = "imported"
    probe: LoraClipProbeApi | None = None
    sourcePath: str | None = None
    edits: LoraClipEditsApi | None = None
    posterPath: str | None = None
    spritePath: str | None = None
    spriteTiles: int | None = None
    triage: LoraClipTriageApi | None = None
    deletedAt: str | None = None


class CreateLoraDatasetRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    name: str = Field(min_length=1)
    type: LoraDatasetTypeApi = "standard"
    triggerWord: str | None = None
    clips: list[LoraClipInput] = Field(default_factory=list[LoraClipInput])
    originatingProjectId: str | None = None


class UploadLoraDatasetRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    provider: LoraProviderApi | None = None


LoraExportFormatApi: TypeAlias = Literal["folder", "zip"]


class ExportLoraDatasetRequest(BaseModel):
    """Write a portable, trainer-ready dataset bundle to local disk.

    `destPath` is a directory (folder mode) or the target `.zip` path
    (zip mode). Rejected clips are excluded unless `includeRejected`.
    `profileId`, when set, builds the bundle's `train_config.yaml` from that
    saved training profile's knobs instead of the built-in defaults; null
    keeps the default config.
    """

    model_config = ConfigDict(strict=True)

    destPath: str = Field(min_length=1)
    format: LoraExportFormatApi = "folder"
    includeRejected: bool = False
    profileId: str | None = None
    # IC-LoRA training-ready normalization knobs (ignored for standard LoRA).
    # Both clips of every pair are re-encoded to `icLoraFps` / short-side
    # `icLoraShortSide` and trimmed to exactly `icLoraBucketFrames` frames so
    # target and reference align; pairs that can't comply are dropped.
    icLoraFps: float = Field(default=25.0, gt=0, le=120)
    icLoraShortSide: int = Field(default=576, ge=32, le=2160)
    icLoraBucketFrames: int = Field(default=49, ge=1, le=2049)
    icLoraMaxDurationSeconds: float | None = Field(default=None, gt=0)
    # Caption words to reject in target captions (the trigger word is always
    # rejected). E.g. ["beard","stubble"] for a beard-removal LoRA.
    forbiddenCaptionWords: list[str] = Field(default_factory=list)
    # Which supplementary artifacts to write next to the core dataset (clips +
    # dataset.json, always included). The export modal exposes these as toggles.
    includeConfig: bool = True  # train_config.yaml
    includeReadme: bool = True  # README.md (trainer instructions)
    includeManifest: bool = True  # ltxdesktop.json (re-import manifest)
    includeModelCard: bool = True  # MODEL_CARD.md (Hugging Face model card)


class ExportLoraDatasetResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    exportPath: str
    clipCount: int
    # Pairs/clips excluded during a training-ready IC-LoRA export, each as a
    # human-readable "name: reason" line for the UI summary (empty for standard).
    droppedPairs: list[str] = Field(default_factory=list)


class ImportLoraDatasetRequest(BaseModel):
    """Re-create a dataset from a bundle folder or `.zip` exported by
    another LTX Desktop (must contain `ltxdesktop.json`)."""

    model_config = ConfigDict(strict=True)

    sourcePath: str = Field(min_length=1)


PublishPlatformApi: TypeAlias = Literal["huggingface", "civitai", "portable"]


class PublicationMetaApi(BaseModel):
    """User-editable model-card fields (pre-seeded server-side from the run)."""

    model_config = ConfigDict(strict=True)

    title: str = Field(min_length=1)
    summary: str = ""
    description: str = ""
    author: str = ""
    license: str = "other"
    tags: list[str] = Field(default_factory=list)
    baseModel: str = "Lightricks/LTX-Video"


class PublicationExampleApi(BaseModel):
    """A showcase clip the user chose from the LoRA's dataset."""

    model_config = ConfigDict(strict=True)

    mediaPath: str = Field(min_length=1)
    caption: str = ""


class PublishLoraPreviewRequest(BaseModel):
    """Render (without writing) the card for each platform. `meta` is omitted on
    the first call so the server returns its suggested fields to prefill."""

    model_config = ConfigDict(strict=True)

    platforms: list[PublishPlatformApi] = Field(min_length=1)
    meta: PublicationMetaApi | None = None
    examples: list[PublicationExampleApi] = Field(default_factory=list[PublicationExampleApi])


class PublishLoraPreviewResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    meta: PublicationMetaApi
    # platform -> rendered Markdown card.
    cards: dict[str, str]


class PublishLoraExportRequest(BaseModel):
    """Write the publication bundle (card(s) + examples + weights) under
    `destPath`."""

    model_config = ConfigDict(strict=True)

    destPath: str = Field(min_length=1)
    platforms: list[PublishPlatformApi] = Field(min_length=1)
    meta: PublicationMetaApi
    examples: list[PublicationExampleApi] = Field(default_factory=list[PublicationExampleApi])


class PublishLoraExportResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    publicationPath: str
    exampleCount: int
    files: list[str]
    weightsFile: str | None = None


class UpdateLoraDatasetRequest(BaseModel):
    """Edit a dataset's name, trigger word, and clip captions.

    Only valid while the dataset is in `draft` / `upload_failed`; once
    uploaded the remote copy is in flight and edits are rejected (409)
    so a re-upload is the explicit path.
    """

    model_config = ConfigDict(strict=True)

    name: str | None = None
    type: LoraDatasetTypeApi | None = None
    triggerWord: str | None = None
    clips: list[LoraClipInput] | None = None


class RenameLoraDatasetRequest(BaseModel):
    """Rename a dataset at any status (display-only; the remote dataset dir is
    already recorded and isn't recomputed from the name)."""

    model_config = ConfigDict(strict=True)

    name: str


class CreateLoraFolderRequest(BaseModel):
    """Create a folder, optionally nested under `parentId` (null = root)."""

    model_config = ConfigDict(strict=True)

    name: str
    parentId: str | None = None


class RenameLoraFolderRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    name: str


class MoveLoraDatasetRequest(BaseModel):
    """Move a dataset into a folder. `folderId` null = root."""

    model_config = ConfigDict(strict=True)

    folderId: str | None = None


class MoveLoraFolderRequest(BaseModel):
    """Reparent a folder. `parentId` null = root; must not create a cycle."""

    model_config = ConfigDict(strict=True)

    parentId: str | None = None


class RunPreprocessingRequest(BaseModel):
    """Create + enqueue a preprocessed dataset from an uploaded dataset.

    `resolutionBuckets` follows the trainer's "WxHxF" form; the backend
    validates spatial%32==0 and frames%8==1 before enqueueing.
    """

    model_config = ConfigDict(strict=True)

    datasetId: str = Field(min_length=1)
    resolutionBuckets: str = Field(default="768x448x49")
    withAudio: bool = False
    autoCaption: bool = True
    captionerType: LoraCaptionerTypeApi = "gemini_flash"


class StartTrainingRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    preprocessedId: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str | None = Field(default=None, max_length=500)
    # Preferred path: reference a saved profile whose config is snapshotted
    # onto the run. `config` remains for back-compat / per-run customization
    # (used when `profileId` is null). `triggerWordOverride`, when set,
    # overrides the resolved config's trigger word for this run only.
    profileId: str | None = None
    config: LoraTrainingConfigApi | None = None
    triggerWordOverride: str | None = None
    # Optional override applied after the profile/config is resolved, so the
    # user can edit/approve validation prompts from the training modal without
    # forking the whole config. Null/omitted keeps the resolved config's
    # prompts (which the runner auto-seeds from captions when left default).
    validationPrompts: list[str] | None = None
    # Explicit acknowledgement for a profile/config outside the conservative
    # envelope of the selected GPU. Auto never needs this flag.
    allowUnsafeOverride: bool = False
    # Backend that runs the job. Defaults to RunPod so existing clients (which
    # omit it) keep training remotely with zero behavior change; "local" routes
    # the run to the WSL2 trainer instead.
    provider: LoraProviderApi = "runpod"
    runpodSelection: RunpodSelectionApi | None = None


class CreateTrainingProfileRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    name: str = Field(min_length=1)
    config: LoraTrainingConfigApi = Field(default_factory=LoraTrainingConfigApi)
    description: str = ""
    datasetTypes: list[LoraDatasetTypeApi] = Field(
        default_factory=lambda: ["standard", "ic_lora"]
    )


class UpdateTrainingProfileRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    name: str | None = None
    config: LoraTrainingConfigApi | None = None
    description: str | None = None
    datasetTypes: list[LoraDatasetTypeApi] | None = Field(default=None, min_length=1)


class StartTrainingPipelineRequest(BaseModel):
    """One-click pipeline: upload → preprocess → train in one action.

    Bundles the preprocess params (resolution/audio/caption) with the training
    config so the backend can auto-advance through every stage. `config`/
    `profileId` resolve the training config exactly like `StartTrainingRequest`.
    """

    model_config = ConfigDict(strict=True)

    datasetId: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str | None = Field(default=None, max_length=500)
    resolutionBuckets: str = Field(default="768x448x49")
    withAudio: bool = False
    autoCaption: bool = True
    captionerType: LoraCaptionerTypeApi = "gemini_flash"
    profileId: str | None = None
    config: LoraTrainingConfigApi | None = None
    triggerWordOverride: str | None = None
    # Optional override applied after the profile/config is resolved, so the
    # user can edit/approve validation prompts from the training modal without
    # forking the whole config. Null/omitted keeps the resolved config's
    # prompts (which the runner auto-seeds from captions when left default).
    validationPrompts: list[str] | None = None
    allowUnsafeOverride: bool = False
    # Backend that runs the whole pipeline. Defaults to RunPod so existing
    # clients keep behaving exactly as before; "local" routes upload →
    # preprocess → train to the WSL2 trainer.
    provider: LoraProviderApi = "runpod"
    # Optional one-run override. The resolved policy and selected volume id are
    # persisted before upload and never changed after remote work begins.
    workspacePolicy: LoraWorkspacePolicyApi | None = None
    runpodSelection: RunpodSelectionApi | None = None


class ReselectRunpodRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    selection: RunpodSelectionApi


class LoraTestConnectionRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    # RunPod is the only backend; kept optional so an empty body validates.
    provider: LoraProviderApi = "runpod"


class LoraTestConnectionResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    ok: bool
    message: str


class LoraGpuOfferApi(BaseModel):
    """A GPU type the connected RunPod account can allocate."""

    model_config = ConfigDict(strict=True)

    id: str
    label: str
    memoryGb: int
    pricePerHr: float | None = None
    available: bool = True
    activeRegionAvailable: bool | None = None
    availableElsewhere: bool | None = None
    bestAvailableRegion: str | None = None
    recommended: bool = False


class LoraNetworkVolumeApi(BaseModel):
    model_config = ConfigDict(strict=True)

    id: str
    name: str
    sizeGb: int
    datacenterId: str = ""
    createdByApp: bool = False
    active: bool = False
    regionHealth: Literal["healthy", "no_stock", "unknown"] = "unknown"
    qualifyingGpuAvailable: bool | None = None
    availableGpuIds: list[str] = Field(default_factory=list[str])
    savedModelReadiness: Literal["ready", "missing", "unknown"] = "unknown"


class LoraRegionHealthApi(BaseModel):
    model_config = ConfigDict(strict=True)

    datacenterId: str
    status: Literal["healthy", "no_stock", "unknown"]
    qualifyingGpuAvailable: bool
    availableGpuIds: list[str] = Field(default_factory=list[str])


class LoraCreateNetworkVolumeRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    datacenterId: str | None = None
    sizeGb: int | None = Field(default=None, ge=250, le=4000)


class LoraSelectNetworkVolumeRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    volumeId: str = Field(min_length=1)


class LoraRelocateNetworkVolumeRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    datacenterId: str = Field(min_length=1)
    sizeGb: int | None = Field(default=None, ge=250, le=4000)


class LoraNetworkVolumeActionResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    ok: bool
    message: str
    volume: LoraNetworkVolumeApi | None = None
    previousVolumeId: str | None = None
    provisioningRequired: bool = False


class LoraPodInfoApi(BaseModel):
    """A pod already on the connected RunPod account."""

    model_config = ConfigDict(strict=True)

    id: str
    name: str
    gpu: str
    status: str
    costPerHr: float | None = None
    createdByApp: bool = False
    # Normalized lifecycle so the UI picks the right action without parsing the
    # raw `status`: the provider's canonical `desiredStatus` and a boolean that's
    # True only while the pod is actually consuming GPU (billable).
    desiredStatus: str = ""
    running: bool = False
    uptimeSeconds: int | None = Field(default=None, ge=0)
    lastStartedAt: str | None = None


class LocalTrainerEligibilityResponse(BaseModel):
    """Whether local (WSL2) LoRA training is possible on this machine.

    Mirrors `LocalTrainerEligibility`; the frontend polls this to decide
    whether to offer the "train locally" provider. `eligible=False` always
    carries a non-empty `reason`; the granular flags let the UI tailor its
    setup guidance (install WSL vs. fix CUDA vs. GPU too small).
    """

    model_config = ConfigDict(strict=True)

    eligible: bool
    reason: str
    wslInstalled: bool
    cudaInWsl: bool
    gpuName: str | None = None
    vramGb: int | None = None


class LoraConnectRunpodResponse(BaseModel):
    """Result of the one-click RunPod connect probe.

    On success the UI populates the GPU picker, explicitly selected volume, and
    existing pods. Connecting never creates paid storage.
    """

    model_config = ConfigDict(strict=True)

    ok: bool
    message: str
    gpus: list[LoraGpuOfferApi] = Field(default_factory=list[LoraGpuOfferApi])
    volumes: list[LoraNetworkVolumeApi] = Field(default_factory=list[LoraNetworkVolumeApi])
    pods: list[LoraPodInfoApi] = Field(default_factory=list[LoraPodInfoApi])
    activeVolumeId: str | None = None
    # Region GPU availability was checked against (the volume's datacenter).
    datacenter: str = ""
    cacheEnabled: bool = False
    requiresVolumeSelection: bool = False
    regionHealth: list[LoraRegionHealthApi] = Field(
        default_factory=list[LoraRegionHealthApi]
    )
    savedModelReadiness: Literal["ready", "missing", "unknown"] = "unknown"
    estimatedModelDownloadBytes: int | None = None


class LoraEstimatePhaseApi(BaseModel):
    model_config = ConfigDict(strict=True)

    phase: Literal["provision", "upload", "preprocess", "train", "idle"]
    lowSeconds: int
    highSeconds: int


class LoraCostEstimateRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    config: LoraTrainingConfigApi = Field(default_factory=LoraTrainingConfigApi)
    clipCount: int = Field(default=1, ge=0)
    totalClipSeconds: float = Field(default=0.0, ge=0.0)
    preprocessed: bool = False
    resolutionBuckets: str = "768x448x49"
    mode: LoraDatasetTypeApi = "standard"
    withAudio: bool = False
    gpuType: str = ""
    gpuVramGb: int = Field(default=0, ge=0)
    gpuPricePerHr: float = Field(default=0.0, ge=0.0)
    storageReadiness: Literal["ready", "missing", "unknown"] = "unknown"
    estimatedModelDownloadBytes: int | None = Field(default=None, ge=0)
    idleTimeoutMinutes: int = Field(default=10, ge=0, le=240)
    storageSizeGb: int = Field(default=250, ge=0)


class LoraCostEstimateResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    lowSeconds: int
    highSeconds: int
    lowGpuCost: float
    highGpuCost: float
    phases: list[LoraEstimatePhaseApi]
    confidence: Literal["low", "medium", "high"]
    matchedHistoryCount: int
    downloadBytes: int | None
    storageMonthlyCost: float


class LoraTerminatePodResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    ok: bool
    message: str


class LoraPodActionResponse(BaseModel):
    """Result of a lifecycle action on a RunPod pod (stop/resume).

    Same shape as `LoraTerminatePodResponse`; a separate type so the OpenAPI
    schema documents stop/resume endpoints distinctly from terminate. On
    failure `ok` is False and `message` explains why (e.g. bad key).
    """

    model_config = ConfigDict(strict=True)

    ok: bool
    message: str


class LoraKeepAliveRequest(BaseModel):
    """Extend one app-owned workspace without changing the global idle policy."""

    model_config = ConfigDict(strict=True)

    minutes: int = Field(default=30, ge=1, le=240)


class LoraJobLogsResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    lines: list[str]


class LoraCaptionClipRequest(BaseModel):
    """Auto-caption one local clip via the desktop-side vision model.

    Decoupled from any dataset: the frontend captions clips one at a time
    (showing progress) and writes the result back through the dataset edit
    endpoint, so the user can review/tweak before upload.
    """

    model_config = ConfigDict(strict=True)

    videoPath: str = Field(min_length=1)
    withAudio: bool = False


class LoraCaptionClipResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    caption: str


class LoraProbeClipRequest(BaseModel):
    """Measure one local clip before it's added to a dataset.

    Like captioning, this is decoupled from any dataset: the frontend
    probes a clip as the user adds it, shows the badges/warnings, and
    persists the result through the dataset create/edit endpoints.
    """

    model_config = ConfigDict(strict=True)

    videoPath: str = Field(min_length=1)


class LoraProbeClipResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    probe: LoraClipProbeApi


class LoraApplyEditsRequest(BaseModel):
    """Render a trimmed/cropped derivative of `sourcePath`.

    Stateless like probing/captioning: the backend writes the derived
    clip to its managed temp area and returns the path + a fresh probe;
    the frontend persists `localPath = derivedPath`, `sourcePath`, and
    `edits` on the clip via the dataset edit endpoint.
    """

    model_config = ConfigDict(strict=True)

    sourcePath: str = Field(min_length=1)
    edits: LoraClipEditsApi


class LoraApplyEditsResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    derivedPath: str
    probe: LoraClipProbeApi


class LoraSceneSplitRequest(BaseModel):
    """Detect scene cuts in a long clip and render each segment.

    `threshold` is the ffmpeg scene score (0.1–0.9); lower = more cuts.
    """

    model_config = ConfigDict(strict=True)

    sourcePath: str = Field(min_length=1)
    threshold: float = Field(default=0.4, ge=0.1, le=0.9)


class LoraSceneApi(BaseModel):
    model_config = ConfigDict(strict=True)

    localPath: str
    startSeconds: float
    endSeconds: float
    probe: LoraClipProbeApi


class LoraSceneSplitResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    scenes: list[LoraSceneApi]


# --- AI dataset prep (Fal, BYOK) -------------------------------------------

LoraNanoBananaModelApi = Literal["nano-banana", "nano-banana-2", "nano-banana-pro"]

# Which local/remote engine performs a LoRA frame edit (edit-frame modal and
# the generate-example pipeline's edit stage). "fal" => Nano Banana (remote,
# needs a Fal API key). "klein" => FLUX.2 [klein] 9B (local, needs the gated
# HuggingFace checkpoint downloaded). Klein is instruction-based editing, so
# the same prompt that drives Nano Banana is reused.
LoraFrameEditEngineApi = Literal["fal", "klein"]


class LoraEditFrameRequest(BaseModel):
    """Edit a frame of a clip with Nano Banana (Fal) or FLUX.2 [klein] 9B (local).

    Extracts the frame at `timeSeconds`, applies the prompt, and returns
    the edited still's path so the frontend can preview it or feed it to
    image-to-video. `engine` selects the editor: "fal" (default, Nano Banana)
    uses `model` (falls back to the saved Nano Banana setting when omitted);
    "klein" runs the local FLUX.2 [klein] 9B edit pipeline and ignores `model`.
    """

    model_config = ConfigDict(strict=True)

    sourcePath: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    timeSeconds: float = Field(default=0.0, ge=0.0)
    model: LoraNanoBananaModelApi | None = None
    engine: LoraFrameEditEngineApi = "fal"


class LoraEditFrameResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    framePath: str


class LoraAnimateFrameRequest(BaseModel):
    """Image-to-video: turn a (usually edited) still into a clip."""

    model_config = ConfigDict(strict=True)

    imagePath: str = Field(min_length=1)
    prompt: str = Field(min_length=1)


class LoraRestyleClipRequest(BaseModel):
    """Video-to-video: re-render an existing clip under a text prompt."""

    model_config = ConfigDict(strict=True)

    sourcePath: str = Field(min_length=1)
    prompt: str = Field(min_length=1)


LoraMotionEngineApi = Literal["ltx_v2v", "kling_motion", "kling_o3"]


class LoraMotionEditRequest(BaseModel):
    """Motion-locked paired generation for edit datasets.

    Uses the original clip as the motion driver and an edited still
    (e.g. a Nano-Banana-edited first frame) as the content anchor, so the
    result keeps the original's motion but shows the edited content —
    yielding aligned (original ↔ edited) pairs for control LoRA training.

    `engine` picks the Fal backend:
      - "ltx_v2v": LTX-2 video-to-video with the edited frame as the first
        frame; `videoStrength` trades motion/structure fidelity (high) for
        freedom to adopt the edited content (low).
      - "kling_motion": Kling motion-control — transfer the original's
        motion onto the edited character image; `characterOrientation`
        selects "video" (full body + camera) or "image" (preserve framing).
      - "kling_o3": Kling O3 video-to-video edit — re-render the source clip
        under `prompt`, using the edited still as an appearance/style
        reference (@Image1). `keepAudio` preserves the source audio.
    """

    model_config = ConfigDict(strict=True)

    sourcePath: str = Field(min_length=1)
    referenceImagePath: str = Field(min_length=1)
    prompt: str = ""
    engine: LoraMotionEngineApi = "ltx_v2v"
    videoStrength: float = Field(default=0.5, ge=0.0, le=1.0)
    characterOrientation: Literal["video", "image"] = "video"
    keepAudio: bool = True


class LoraDerivedClipResponse(BaseModel):
    """Result of an AI op that produces a new clip (animate / restyle)."""

    model_config = ConfigDict(strict=True)

    derivedPath: str
    probe: LoraClipProbeApi


# --- Pexels stock-media browser (BYOK) -------------------------------------

PexelsMediaKindApi = Literal["video", "photo"]
PexelsOrientationApi = Literal["", "landscape", "portrait", "square"]


class PexelsSearchRequest(BaseModel):
    """Search Pexels for stock media to add to a LoRA collection.

    An empty `query` returns the curated/popular feed so the browser opens
    with content. `media` selects the photo or video endpoint.
    """

    model_config = ConfigDict(strict=True)

    query: str = ""
    media: PexelsMediaKindApi = "video"
    page: int = Field(default=1, ge=1)
    perPage: int = Field(default=24, ge=1, le=80)
    orientation: PexelsOrientationApi = ""


class PexelsMediaItemApi(BaseModel):
    """One Pexels search hit, normalized across photos + videos."""

    model_config = ConfigDict(strict=True)

    id: str
    kind: PexelsMediaKindApi
    width: int
    height: int
    durationSeconds: float | None = None
    previewUrl: str
    downloadUrl: str
    downloadExt: str
    pexelsUrl: str
    author: str
    authorUrl: str
    alt: str


class PexelsSearchResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    items: list[PexelsMediaItemApi]
    page: int
    perPage: int
    totalResults: int
    hasNext: bool


class PexelsDownloadRequest(BaseModel):
    """Download a chosen Pexels asset into app storage. `url` must be a file
    URL returned by `PexelsSearchResponse`.
    """

    model_config = ConfigDict(strict=True)

    url: str = Field(min_length=1)
    kind: PexelsMediaKindApi
    ext: str = ""


class PexelsDownloadResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    localPath: str
    probe: LoraClipProbeApi | None = None


# --- Local clip-prep jobs (sprite/filmstrip generation) --------------------

LoraClipJobKindApi = Literal["sprite"]
LoraClipJobStatusApi = Literal["pending", "running", "completed", "failed"]


class LoraClipJobApi(BaseModel):
    model_config = ConfigDict(strict=True)

    id: str
    kind: LoraClipJobKindApi
    sourcePath: str
    status: LoraClipJobStatusApi
    posterPath: str | None = None
    spritePath: str | None = None
    spriteTiles: int | None = None
    error: str | None = None


class LoraEnqueueClipJobsRequest(BaseModel):
    """Enqueue background prep jobs for a batch of source clips.

    Stateless w.r.t. datasets: jobs are keyed by source path so the
    curation gallery can analyze clips before any dataset is saved. The
    frontend polls `GET /api/lora/clip-jobs` and merges results onto its
    in-memory clips.
    """

    model_config = ConfigDict(strict=True)

    sourcePaths: list[str] = Field(min_length=1)
    kind: LoraClipJobKindApi = "sprite"


class LoraClipJobsResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    jobs: list[LoraClipJobApi]


# --- AI target/variant derivation jobs (background pipeline) ---------------

# Stage-3 backend that drives the edited still into motion.
#   - "ltx_local": local GPU IC-LoRA control (depth/canny). The driver video
#     supplies the control signal; the edited still anchors frame 0. Runs on
#     the single-flight GPU, so the runner waits for a free slot.
#   - "kling": remote Fal Kling motion-control (no local GPU).
#   - "kling_o3": remote Fal Kling O3 video-to-video edit — re-render the driver
#     clip under `scenePrompt`, with the edited still as an appearance reference
#     (@Image1). `keepAudio` preserves the source audio. No local GPU.
LoraDerivationEngineApi = Literal["ltx_local", "kling", "kling_o3"]
# How the AI result is organized once it lands. "target": the generated clip is
# the IC-LoRA target and its driver is the reference (start from a reference,
# generate the target). "reference": the generated clip is a new reference for
# the *source* clip, which becomes the target (start from a target, generate a
# reference). "variant": a standalone augmentation — added to the dataset on its
# own, never grouped into an example (the only meaningful mode for standard
# LoRA, and an explicit "just make a variation" option in IC-LoRA). "frame_edit":
# an edit-only job from the frame-edit modal — no animate step; the completed
# edited still is added to the gallery as a still entry that remembers its
# driver, so it can be motion-locked into a training example later.
LoraDerivationDirectionApi = Literal["target", "reference", "variant", "frame_edit"]
LoraDerivationStatusApi = Literal[
    "pending",
    "editing",
    # Paused after the Nano Banana edit, awaiting the user's go-ahead before
    # the (expensive) motion drive — see `requireReview`.
    "review",
    # User approved the reviewed edit; queued for the motion drive only.
    "approved",
    "generating",
    "completed",
    "failed",
    "cancelled",
]


class CreateLoraDerivationJobRequest(BaseModel):
    """Enqueue a background 'generate target / variant' pipeline.

    The unified staged pipeline:
      1. Source frame: use `framePath` if given (still entry), else extract
         the frame at `frameTimeSeconds` from `driverPath`.
      2. Edit (optional): if `editPrompt` is non-empty, edit that frame with
         Nano Banana to produce the content anchor.
      3. Drive into motion using `driverPath` as the motion source:
         - engine "ltx_local": local IC-LoRA control (`conditioningType`
           depth/canny/pose, `conditioningStrength`), edited still anchors frame 0.
         - engine "kling": remote Kling motion-control (`characterOrientation`).
         - engine "kling_o3": remote Kling O3 video-to-video edit — re-render
           the driver under `scenePrompt`, edited still as appearance reference
           (`keepAudio` preserves the source audio).

    `referencePath` set => result is a paired *target* for IC-LoRA training
    (its reference is the driver). Null => standalone *variant* (standard LoRA).
    """

    model_config = ConfigDict(strict=True)

    driverPath: str = Field(min_length=1)
    framePath: str | None = None
    referencePath: str | None = None
    datasetId: str | None = None
    sourceClipId: str | None = None
    frameTimeSeconds: float = Field(default=0.0, ge=0.0)
    editPrompt: str = ""
    nanoBananaModel: LoraNanoBananaModelApi | None = None
    # Engine for the optional edit stage (frame edit). "fal" (default) =>
    # Nano Banana; "klein" => local FLUX.2 [klein] 9B. The motion-drive
    # `engine` below is independent of this edit-stage engine.
    editEngine: LoraFrameEditEngineApi = "fal"
    scenePrompt: str = ""
    engine: LoraDerivationEngineApi = "ltx_local"
    # See `LoraDerivationDirectionApi`. Default "target" keeps the legacy
    # "generate the target" behavior (result is the IC-LoRA target).
    direction: LoraDerivationDirectionApi = "target"
    conditioningType: ConditioningType = "depth"
    conditioningStrength: float = Field(default=1.0, ge=0.0, le=1.0)
    characterOrientation: Literal["video", "image"] = "video"
    # Kling O3 ("kling_o3") only: keep the source clip's original audio.
    keepAudio: bool = True
    # Whether the resolved still is (or will be) a Nano-Banana-edited frame
    # rather than a verbatim source frame — i.e. an `editPrompt` will run or a
    # committed foreground preview is supplied via `framePath`. Kling O3 uses
    # this to decide whether to pass the still as an appearance reference
    # (@Image1); without an edit it does a pure video + prompt re-render.
    frameEdited: bool = False
    caption: str = ""
    label: str = ""
    # When true (and an edit will run), the job pauses in `review` after the
    # Nano Banana edit so the user can approve/regenerate the still before the
    # motion drive burns time/tokens. Used for bulk and un-previewed edits.
    requireReview: bool = False


class RegenerateLoraDerivationEditRequest(BaseModel):
    """Re-run the Nano Banana edit for a job paused in `review`.

    Optionally swap in a new `editPrompt`; omit/blank to re-roll the same one.
    """

    model_config = ConfigDict(strict=True)

    editPrompt: str | None = None


class CancelAllLoraDerivationsRequest(BaseModel):
    """Abort a whole bulk Fal generation run in one call.

    Cancels every active derivation job, optionally scoped to a single
    dataset (`datasetId`) so unrelated collections keep running. Omit to
    cancel across all datasets.
    """

    model_config = ConfigDict(strict=True)

    datasetId: str | None = None


class LoraDerivationJobApi(BaseModel):
    model_config = ConfigDict(strict=True)

    id: str
    status: LoraDerivationStatusApi
    engine: LoraDerivationEngineApi
    direction: LoraDerivationDirectionApi = "target"
    label: str = ""
    driverPath: str
    referencePath: str | None = None
    datasetId: str | None = None
    sourceClipId: str | None = None
    caption: str = ""
    requireReview: bool = False
    # Engine used for the edit stage (Nano Banana vs local FLUX.2 [klein] 9B).
    editEngine: LoraFrameEditEngineApi = "fal"
    # The exact still fed to the editor (extracted source frame, or a
    # pre-existing still). Lets the review UI show a true "before" that
    # matches the edited "after" instead of the clip's poster thumbnail.
    sourceFramePath: str | None = None
    editedFramePath: str | None = None
    derivedPath: str | None = None
    probe: LoraClipProbeApi | None = None
    error: str | None = None
    createdAt: str
    updatedAt: str | None = None


class LoraDerivationJobsResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    jobs: list[LoraDerivationJobApi]


# ============================================================
# Media extraction (ffmpeg-backed frame/audio helpers)
# Used by the LoRA dataset studio's frame-edit flow (extract-frame).
# ============================================================


class ExtractFrameRequest(BaseModel):
    """Pull a single video frame to disk and return its absolute path."""

    model_config = ConfigDict(strict=True)

    sourcePath: str = Field(min_length=1)
    timeSeconds: float = Field(default=0.0, ge=0.0)


class ExtractFrameResponse(BaseModel):
    """Absolute path to the extracted PNG frame."""

    model_config = ConfigDict(strict=True)

    path: str


class ExtractAudioRequest(BaseModel):
    """Pull a slice of audio to disk and return its absolute path."""

    model_config = ConfigDict(strict=True)

    sourcePath: str = Field(min_length=1)
    startSeconds: float = Field(default=0.0, ge=0.0)
    durationSeconds: float = Field(default=0.0, ge=0.0, le=300.0)


class ExtractAudioResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    path: str

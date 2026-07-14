"""Persistent LoRA-trainer state schemas.

The LoRA trainer drives the official LTX-2 trainer scripts
(`caption_videos.py`, `process_dataset.py`, `train.py`) on a remote
RunPod GPU. All GPU work happens remotely; the desktop app is the durable
control plane.

The feature is modeled as three independent, durable ledgers — one
JSON file each under ``APP_DATA_DIR`` — because the high-value UX is
"preprocess once, train many times":

  - ``lora_datasets.json``   -> `LoraDatasetsState`   (local clips + captions)
  - ``lora_preprocessed.json`` -> `PreprocessedState` (remote `.precomputed` latents)
  - ``lora_training.json``   -> `TrainingState`       (remote training run + LoRA)

Each entity stores a `TargetHandle` snapshot (provider + remote ids)
so the background reconciler can re-poll a remote job after an app
restart instead of orphaning it. Crash recovery on load mirrors the
queue ledger (`state/queue_state.py`): an in-flight item with no live
handle is reset to its pending state; one with a handle is kept so the
reconciler re-polls.

Everything here is JSON-serializable so the on-disk format matches the
in-memory model 1:1, exactly like the queue ledger.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from api_types import (
    DEFAULT_VALIDATION_PROMPT,
    LoraClipEditsApi,
    LoraClipProbeApi,
    LoraCropApi,
    LoraScaleApi,
    LoraDatasetApi,
    LoraDatasetClipApi,
    LoraDatasetsResponse,
    LoraFolderApi,
    LoraGpuStatusApi,
    LoraTrimApi,
    LoraPreprocessedApi,
    LoraPreprocessedResponse,
    LoraPendingPipelineApi,
    LoraTargetHandleApi,
    LoraTrainingConfigApi,
    LoraTrainingJobApi,
    LoraTrainingProfileApi,
    LoraTrainingProfilesResponse,
    LoraTrainingResponse,
    LoraValidationFeedItemApi,
    LoraCheckpointArtifactApi,
    RunpodSelectionApi,
)

# ----------------------------------------------------------------
# Status unions (state machines)
# ----------------------------------------------------------------

# A dataset starts as a local-only `draft`. The reconciler uploads its
# clips to the remote target (`uploading` -> `uploaded`); a transport
# failure lands in `upload_failed` (retryable by re-requesting upload);
# a user cancel during upload lands in `cancelled` (the pod is released and
# the run stops before preprocessing — re-request upload to try again).
LoraDatasetStatus = Literal[
    "draft",
    "uploading",
    "uploaded",
    "upload_failed",
    "cancelled",
    "gpu_selection_required",
]

# A preprocessed dataset runs two remote scripts in sequence:
# `caption_videos.py` (only when the user asked for auto-captioning)
# then `process_dataset.py`. `ready` means the remote `.precomputed`
# dir exists and can feed any number of training runs.
PreprocessStatus = Literal[
    "pending", "captioning", "preprocessing", "ready", "failed", "cancelled"
]

# A training job runs `train.py` and, on success, the resulting
# `lora_weights.safetensors` is downloaded to the local LoRA dir.
TrainingStatus = Literal[
    "pending",
    "running",
    "completed",
    "failed",
    "cancelled",
    "gpu_selection_required",
]

# Which remote execution backend owns the job. `runpod` is BYOK on-demand
# GPU pods; `local` runs the same Linux trainer commands inside WSL2 on the
# user's own Windows machine (no remote compute).
TrainerProvider = Literal["runpod", "local"]

# RunPod workspace placement is snapshotted on the dataset before upload.  The
# dataset owns every later preprocessing/training artifact, so changing this
# after the first remote write would strand those artifacts in another region.
WorkspacePolicy = Literal["primary_cache", "ephemeral_any_region"]

# Captioning backend exposed by `caption_videos.py`. `gemini_flash`
# reuses the app's existing Gemini API key and avoids loading the
# heavy local Qwen-Omni model on the remote GPU.
CaptionerType = Literal["qwen_omni", "gemini_flash"]

# Maps 1:1 to the two shipped LTX-trainer example configs.
TrainingPreset = Literal["standard", "low_vram"]

# What kind of LoRA a collection trains, which selects the `training_strategy`
# shape under the official `flexible` strategy:
#   - `standard`  -> `video` (+ optional `audio`) modality block: learns from
#                    individual clips (look/subject/style). Optional joint audio
#                    via `with_audio`.
#   - `ic_lora`   -> `video` with a `reference` + `first_frame` condition
#                    (In-Context LoRA): learns a transformation from paired
#                    reference -> target clips.
# Defaults to `standard` so existing datasets keep their behaviour.
LoraDatasetType = Literal["standard", "ic_lora"]

# Where a clip came from. `imported` = user-picked file; `gen_space` =
# sent over from a project's Gen Space; `ai_derived` = produced by an
# in-app edit/restyle (Phase 3). Provenance only — never gates training.
ClipOrigin = Literal["imported", "gen_space", "ai_derived"]

# Manual curation flag. `None` = unreviewed. `keep`/`reject` gate training
# eligibility; `holdout` reserves a clip for the in-training validation feed
# (excluded from training, reference video staged for IC-LoRA validation —
# see `lora_training_runner._start_training`). Forward-compatible: old
# ledgers without the value deserialize as None.
ClipTriage = Literal["keep", "reject", "holdout"]


class ClipProbe(BaseModel):
    """Cached measurement of a clip (duration / resolution / fps / audio).

    Stored on the clip so the curation UI can render badges + quality
    warnings without re-probing on every render. Populated by the
    desktop ffmpeg probe at add-time.
    """

    model_config = ConfigDict(strict=True)

    duration_seconds: float
    width: int
    height: int
    fps: float
    frame_count: int
    has_audio: bool
    video_codec: str | None = None


class ClipTrim(BaseModel):
    model_config = ConfigDict(strict=True)

    start_seconds: float
    end_seconds: float


class ClipCrop(BaseModel):
    model_config = ConfigDict(strict=True)

    x: int
    y: int
    width: int
    height: int


class ClipScale(BaseModel):
    model_config = ConfigDict(strict=True)

    width: int
    height: int


class ClipEdits(BaseModel):
    """Non-destructive edit stack (trim → crop → scale → fps → reverse →
    speed). Persisted so the editor can re-render from the untouched
    source when a value changes."""

    model_config = ConfigDict(strict=True)

    trim: ClipTrim | None = None
    crop: ClipCrop | None = None
    scale: ClipScale | None = None
    fps: float | None = None
    speed: float | None = None
    mute: bool = False
    reverse: bool = False


class TargetHandle(BaseModel):
    """Opaque-ish snapshot identifying a remote job/pod.

    Persisted on the owning entity so the reconciler can re-poll the
    remote provider across app restarts. `remote_job_id` is the remote
    process/exec id; `pod_id` is the RunPod pod the job runs on.
    """

    model_config = ConfigDict(strict=True)

    provider: TrainerProvider = "runpod"
    pod_id: str | None = None
    remote_job_id: str | None = None

    @field_validator("provider", mode="before")
    @classmethod
    def _coerce_legacy_provider(cls, value: object) -> object:
        # Old persisted handles may carry a now-removed provider (e.g. "coder").
        # Coerce anything unrecognized to the default backend so loading never
        # crashes on legacy state; known providers pass through.
        return value if value in ("runpod", "local") else "runpod"


class RunpodSelection(BaseModel):
    """Explicit immutable compute/storage selection for one RunPod run."""

    model_config = ConfigDict(strict=True)

    gpu_type: str
    gpu_vram_gb: int = 0
    datacenter: str = ""
    workspace_policy: WorkspacePolicy = "ephemeral_any_region"
    volume_id: str | None = None

    @field_validator("volume_id")
    @classmethod
    def _validate_volume_id(cls, value: str | None) -> str | None:
        return value or None


# ----------------------------------------------------------------
# Dataset entity
# ----------------------------------------------------------------


class LoraClip(BaseModel):
    """A single local training clip plus its (editable) caption.

    `caption` is what gets written into the remote `dataset.json`. The
    optional `reference_path` is reserved for future IC-LoRA paired
    datasets; single-clip LoRA leaves it ``None`` and the field keeps
    the on-disk schema forward-compatible without a migration.
    """

    model_config = ConfigDict(strict=True)

    id: str
    local_path: str
    caption: str = ""
    duration_seconds: float | None = None
    # `reference_path` is the PRIMARY conditioning reference (kept for the
    # single-reference trainer export and back-compat). `reference_paths`
    # holds ALL references for a manually-grouped IC-LoRA set (one target,
    # one-or-more references). Empty list = fall back to `reference_path`.
    reference_path: str | None = None
    reference_paths: list[str] = Field(default_factory=list)
    origin: ClipOrigin = "imported"
    probe: ClipProbe | None = None
    # Untouched original; `local_path` points at the rendered derivative
    # once the clip is edited. Null = `local_path` is itself the original.
    source_path: str | None = None
    edits: ClipEdits | None = None
    # Curation preview assets (generated by a local clip-job). `poster_path`
    # is a representative frame; `sprite_path` is a horizontal filmstrip
    # montage for hover-scrub; `sprite_tiles` is the frame count in it.
    poster_path: str | None = None
    sprite_path: str | None = None
    sprite_tiles: int | None = None
    # Manual keep/reject curation flag (None = unreviewed). Forward-compatible:
    # old ledgers without the field deserialize as None.
    triage: ClipTriage | None = None
    # Soft-delete timestamp (ISO-8601). When set the clip is in the recycle bin:
    # hidden from the gallery and excluded from pairing, readiness, training and
    # export until restored or permanently deleted. None = live clip.
    deleted_at: str | None = None


class LoraDataset(BaseModel):
    model_config = ConfigDict(strict=True)

    id: str
    name: str
    created_at: str
    status: LoraDatasetStatus
    # Selects the `training_strategy` shape under `flexible` (standard vs IC-LoRA).
    # Defaults to `standard` so older persisted datasets load unchanged.
    type: LoraDatasetType = "standard"
    # Trigger token passed to `process_dataset.py --lora-trigger`. The
    # trainer prepends it to every caption at preprocessing time, so we
    # store it once on the dataset rather than editing each caption.
    trigger_word: str | None = None
    clips: list[LoraClip] = Field(default_factory=list[LoraClip])
    # Backend chosen for this dataset's run, persisted at upload-start so the
    # reconciler's FIRST stage (upload provisions the workspace) talks to the
    # right target — `target` (which carries the provider thereafter) doesn't
    # exist yet at that point. Defaults to RunPod so legacy datasets and the
    # plain upload path are unchanged.
    provider: TrainerProvider = "runpod"
    # Immutable for the lifetime of a started pipeline. `cache_volume_id`
    # snapshots the selected primary volume so a later settings change or cache
    # relocation cannot silently move an in-flight/recoverable job.
    workspace_policy: WorkspacePolicy = "primary_cache"
    cache_volume_id: str | None = None
    runpod_selection: RunpodSelection | None = None
    # Remote directory the clips + dataset.json were uploaded to. Set
    # once `status == uploaded`; consumed by preprocessing.
    remote_dataset_dir: str | None = None
    target: TargetHandle | None = None
    error: str | None = None
    updated_at: str | None = None
    # Human-readable sub-stage shown in the UI while `status == uploading`
    # (e.g. "Creating GPU pod…", "Installing trainer & downloading model…",
    # "Uploading clips…"). The reconciler sets it before each phase so the
    # card reflects what's actually happening, not a flat "Uploading".
    status_detail: str | None = None
    # Optional structured progress for the side panel: 0-100 percent and an
    # ETA in seconds, parsed from the remote setup log (the download bar).
    # None when the current phase has no measurable progress.
    status_percent: int | None = None
    status_eta_seconds: int | None = None
    upload_started_at: str | None = None
    upload_completed_at: str | None = None
    # Transient-failure counter for the reconciler's outer guard (upload phase).
    # `status_detail` carries the retry message; once `consecutive_failures`
    # reaches `_TRANSIENT_FAILURE_BUDGET` the handler escalates to
    # `upload_failed`. Cleared on the next clean upload tick.
    consecutive_failures: int = 0
    # User requested cancel during `uploading`. The reconciler checks this at
    # each upload sub-phase boundary (after pod acquire, after provisioning,
    # after staging, after the transfer): when set, it releases the pod and
    # finalizes the cancel via `mark_dataset_upload_cancelled`. Set by
    # `request_cancel_upload`; cleared on finalize or on a re-requested upload.
    cancel_requested: bool = False
    # ISO timestamp of the last time the reconciler used this dataset's
    # remote pod (upload, or an active preprocess/training tick). Drives
    # RunPod idle auto-stop: once no work touches the pod for the
    # configured window, the reconciler terminates it to cap billing.
    last_active_at: str | None = None
    # Provenance only: the project this dataset was started from (clips
    # imported from its Gen Space). LoRAs remain global across projects.
    originating_project_id: str | None = None
    # One-click pipeline intent: when set, the reconciler auto-advances this
    # dataset through preprocess → train (with these params) the moment upload
    # finishes, instead of stopping for a manual step. Cleared once consumed.
    auto_pipeline: AutoPipelineSpec | None = None
    # IC-LoRA media is normalized before upload. Record that staging envelope
    # so later preprocessing cannot request more pixels/frames than were shipped.
    ic_staged_short_side: int = 576
    ic_staged_bucket_frames: int = 49
    # Folder the dataset belongs to (null = root). Purely an organizational
    # hint for the sidebar tree; never read by the training pipeline. Defaults
    # to None so legacy ledgers load unchanged.
    folder_id: str | None = None
    # Reversible organization only. Archived datasets remain resolvable by id
    # for runs/library joins but are hidden from default list responses and
    # excluded from reconciler work.
    archived_at: str | None = None
    # Per-workspace post-run lifecycle state. These fields live on the dataset
    # because every derived preprocess/run shares its workspace handle.
    keep_alive_until: str | None = None
    final_activity_at: str | None = None
    release_status: Literal["scheduled", "releasing", "released", "failed"] | None = None
    release_error: str | None = None
    release_attempted_at: str | None = None
    last_pod_id: str | None = None


# ----------------------------------------------------------------
# Collection folder (organizational grouping for datasets)
# ----------------------------------------------------------------


class LoraFolder(BaseModel):
    """A user-created folder for organizing datasets in the sidebar tree.

    Nesting is expressed via `parent_id` (null = root). Folders are display
    containers only — they never participate in training. Deleting a
    non-recursive folder moves its contents up to `parent_id`; a recursive
    delete removes subfolders and their datasets.
    """

    model_config = ConfigDict(strict=True)

    id: str
    name: str
    parent_id: str | None = None
    created_at: str


# ----------------------------------------------------------------
# Preprocessed dataset entity
# ----------------------------------------------------------------


class PreprocessedDataset(BaseModel):
    model_config = ConfigDict(strict=True)

    id: str
    dataset_id: str
    created_at: str
    status: PreprocessStatus
    # Resolution bucket string in the trainer's "WxHxF" form (e.g.
    # "768x448x49"). Spatial dims must be multiples of 32 and frames
    # must satisfy frames % 8 == 1 — validated at the API boundary.
    resolution_buckets: str
    # The bucket string preprocessing actually trained with, when it differs
    # from `resolution_buckets`. Set when an IC-LoRA low_vram run had multiple
    # buckets configured: `process_dataset.py` rejects multi-bucket + reference
    # downscaling, so the runner collapses to the first bucket and records it
    # here so the UI/run-summary can show the real trained resolution instead
    # of the user's original (uncollapsed) list. None when no collapse occurred.
    effective_resolution_buckets: str | None = None
    with_audio: bool = False
    auto_caption: bool = True
    captioner_type: CaptionerType = "gemini_flash"
    # Training preset in effect when this dataset was preprocessed — drives the
    # text-encoder precision (low_vram -> 8-bit, standard -> bf16) so the
    # preprocess stage matches the training stage and the official
    # t2v_lora_low_vram.yaml (Gemma3 12B is 23 GB in bf16 and OOMs a 32 GB GPU
    # under WSL2; 8-bit halves it). GPU-adjusted at the route boundary via
    # _apply_gpu_preset, so this is the final preset, not the user's raw pick.
    preset: TrainingPreset = "standard"
    # Immutable trainer source contract used for this preprocessing snapshot.
    trainer_repo_url: str | None = None
    trainer_repo_ref: str | None = None
    # Remote `.precomputed` dir produced by process_dataset.py. Set
    # once `status == ready`; consumed by every training run bound to
    # this preprocessed dataset.
    remote_precomputed_dir: str | None = None
    target: TargetHandle | None = None
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    # Whether the auto-captioning sub-stage already completed for this item.
    # Set when captioning succeeds and the item advances to `preprocessing`
    # (process_dataset). Resume uses it to skip re-captioning and go straight
    # to re-running `process_dataset.py` (the heavy step that typically fails),
    # so a resumed run doesn't burn time re-captioning clips it already
    # captioned. Reset clears it so a full reset re-captions from scratch.
    captioning_completed: bool = False
    # Set by `request_preprocess_reset`: the reconciler observes it on the next
    # tick, wipes the remote `.precomputed` latent cache (and re-captions, if
    # auto-caption is on) before re-running, then clears it. Distinguishes
    # "reset from scratch" from "resume reusing cached state".
    reset_requested: bool = False
    # User requested cancellation. The reconciler observes this on its next
    # tick, terminates the in-flight remote caption/`process_dataset` job, and
    # then moves to `cancelled` (see `mark_preprocess_cancelled`). Unlike the
    # old behavior — which flipped an in-flight item to `cancelled` on the spot
    # and so orphaned the still-running remote job — the status stays
    # `captioning`/`preprocessing` (so `list_active_preprocessed` keeps
    # surfacing it) until the remote job is actually gone.
    cancel_requested: bool = False
    # Transient-failure bookkeeping for the reconciler's outer guard. When an
    # unexpected (retryable) exception escapes `_reconcile_preprocess`, the
    # runner records it here instead of silently swallowing: `status_detail`
    # surfaces "Retrying after error: …" on the card while `consecutive_failures`
    # counts tries; once it reaches `_TRANSIENT_FAILURE_BUDGET` the handler
    # escalates to `failed`. Cleared on the next clean tick.
    status_detail: str | None = None
    consecutive_failures: int = 0
    # One-click pipeline carry-forward: when this preprocessed dataset was
    # created by an auto-pipeline, this holds the training run to launch the
    # moment it reaches `ready`. Cleared once the run is started.
    auto_training: PendingTraining | None = None


# ----------------------------------------------------------------
# Training job entity
# ----------------------------------------------------------------


def _default_target_modules() -> list[str]:
    return ["to_k", "to_q", "to_v", "to_out.0"]


def _default_validation_prompts() -> list[str]:
    return [DEFAULT_VALIDATION_PROMPT]


def _default_stg_blocks() -> list[int]:
    return [29]


class TrainingConfig(BaseModel):
    """The full tunable LTX-2 trainer surface, snapshotted per run.

    Every field maps to the trainer YAML built by `lora_config_builder`.
    Defaults reproduce the previously-hardcoded document, so old persisted
    runs (which only stored a 7-field subset) deserialize unchanged and
    still build identical YAML. `preset` is a starting template only —
    when an acceleration field is left ``None`` the builder fills it from
    the preset, so the two shipped presets keep their exact behaviour.

    `with_audio` and `trigger_word` are dataset/preprocessing-driven (the
    handler sets `with_audio` from the preprocessed item); they live here
    for back-compat and the builder, not the profile editor.
    """

    model_config = ConfigDict(strict=True)

    preset: TrainingPreset = "standard"

    # ---- LoRA adapter ----
    rank: int = Field(default=32, ge=1, le=256)
    alpha: int = Field(default=32, ge=1, le=256)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    target_modules: list[str] = Field(default_factory=_default_target_modules)

    # ---- Optimization ----
    learning_rate: float = Field(default=1e-4, gt=0.0, le=1.0)
    steps: int = Field(default=2000, ge=1, le=100_000)
    batch_size: int = Field(default=1, ge=1, le=64)
    gradient_accumulation_steps: int = Field(default=1, ge=1, le=256)
    max_grad_norm: float = Field(default=1.0, ge=0.0, le=100.0)
    # None -> preset-derived (adamw / adamw8bit).
    optimizer_type: str | None = None
    scheduler_type: str = "linear"
    enable_gradient_checkpointing: bool = True

    # ---- Acceleration (None -> preset-derived) ----
    mixed_precision_mode: str = "bf16"
    quantization: str | None = None
    load_text_encoder_in_8bit: bool | None = None
    offload_optimizer_during_validation: bool | None = None

    # ---- Data ----
    num_dataloader_workers: int = Field(default=2, ge=0, le=16)

    # ---- Training strategy override (None -> dataset-type default) ----
    first_frame_conditioning_p: float | None = Field(default=None, ge=0.0, le=1.0)

    # ---- Validation ----
    validation_prompts: list[str] = Field(default_factory=_default_validation_prompts)
    validation_negative_prompt: str = (
        "worst quality, inconsistent motion, blurry, jittery, distorted"
    )
    validation_video_width: int = Field(default=576, ge=32)
    validation_video_height: int = Field(default=576, ge=32)
    validation_video_frames: int = Field(default=49, ge=1)
    validation_frame_rate: float = Field(default=25.0, gt=0.0)
    validation_inference_steps: int = Field(default=30, ge=1, le=500)
    validation_interval: int = Field(default=250, ge=1)
    validation_guidance_scale: float = Field(default=4.0, ge=0.0, le=30.0)
    validation_seed: int = Field(default=42, ge=0)
    # Spatiotemporal guidance (advanced).
    stg_scale: float = Field(default=1.0, ge=0.0, le=30.0)
    stg_blocks: list[int] = Field(default_factory=_default_stg_blocks)
    stg_mode: str = "stg_av"
    # None -> dataset-type default (IC-LoRA skips validation).
    skip_initial_validation: bool | None = None

    # ---- Checkpoints ----
    checkpoint_interval: int = Field(default=250, ge=1)
    checkpoint_keep_last_n: int = Field(default=3, ge=1, le=100)
    checkpoint_precision: str = "bfloat16"

    # ---- Flow matching (advanced) ----
    timestep_sampling_mode: str = "shifted_logit_normal"

    # ---- Hub / tracking (advanced) ----
    push_to_hub: bool = False
    hub_model_id: str | None = None
    wandb_enabled: bool = False

    # ---- Misc ----
    seed: int = Field(default=42, ge=0)
    # Resume from a remote checkpoint path (advanced); None = train fresh.
    load_checkpoint: str | None = None

    # ---- Dataset/preprocessing-driven (not surfaced in the profile UI) ----
    with_audio: bool = False
    trigger_word: str | None = None


class PendingTraining(BaseModel):
    """Training run to launch once a preprocessed dataset is ready.

    Carried by the one-click pipeline from preprocess → train so the reconciler
    can start the run with the user's chosen config without a second click.
    """

    model_config = ConfigDict(strict=True)

    config: TrainingConfig
    name: str
    # User-authored behavior summary for library metadata and generated
    # auto-prompt instructions. Deliberately separate from the display name.
    description: str | None = None
    gpu_type: str = ""
    gpu_vram_gb: int = 0
    # Backend to launch the run on once the preprocessed dataset is ready.
    # Carried from the one-click pipeline so the auto-started training job
    # inherits the user's chosen provider. Defaults to RunPod.
    provider: TrainerProvider = "runpod"
    runpod_selection: RunpodSelection | None = None
    # One-click pipelines begin attribution when upload first acquires/uses a
    # RunPod pod. These values travel upload -> preprocess -> TrainingJob.
    workload_billing_started_at: str | None = None
    captured_hourly_rate: float | None = None


class AutoPipelineSpec(BaseModel):
    """Full one-click intent stored on a dataset: how to preprocess, then train.

    Set when the user starts a one-click run; the reconciler reads it on
    upload-complete to auto-create the preprocess (carrying `training` forward).
    """

    model_config = ConfigDict(strict=True)

    resolution_buckets: str
    with_audio: bool = False
    auto_caption: bool = True
    captioner_type: CaptionerType = "gemini_flash"
    training: PendingTraining
    runpod_selection: RunpodSelection | None = None


# Resolve the forward references on the dataset/preprocessed models now that the
# pipeline carry models above exist (they're declared earlier in the file).
LoraDataset.model_rebuild()
PreprocessedDataset.model_rebuild()


# Source of a validation sample in the training-results feed: a free-form user
# prompt, or a held-out clip from the dataset (caption, + reference video for
# IC-LoRA).
ValidationSampleSource = Literal["prompt", "holdout"]


class ValidationSampleRef(BaseModel):
    """One validation sample configured for a run, for the results feed.

    Stored on the `TrainingJob` at start time so the reconciler can map a
    downloaded validation artifact (identified by its 1-based sample index in
    the trainer's ``step_NNNNNNN_i.ext`` filename) back to the prompt/source
    it was generated from — without recomputing the sample list each poll.
    """

    model_config = ConfigDict(strict=True)

    prompt: str
    source: ValidationSampleSource = "prompt"
    # Local path of the held-out clip's reference video, downloaded once so the
    # feed can show the original alongside the generated output (IC-LoRA).
    # None for prompt samples and until the reference is fetched.
    reference_local_path: str | None = None


class ValidationFeedItem(BaseModel):
    """One generated validation sample downloaded into the feed.

    `step` is the training step at which it was generated, `sample_index` is
    the 1-based index into the run's validation sample list, and `local_path`
    is the downloaded media file the frontend renders.
    """

    model_config = ConfigDict(strict=True)

    step: int
    sample_index: int
    local_path: str
    extension: str = "mp4"
    source: ValidationSampleSource = "prompt"
    prompt: str = ""
    reference_local_path: str | None = None
    created_at: str = ""


class CheckpointArtifact(BaseModel):
    """One adapter checkpoint downloaded from the remote run output.

    The trainer writes ``lora_weights_step_NNNNN.safetensors`` under
    ``{remote_output_dir}/checkpoints/`` at each checkpoint interval; the
    reconciler downloads each as it appears so the user can reveal the
    checkpoint file alongside its validation samples. `step` pairs it with
    the validation feed entries generated at the same step; `local_path` is
    the downloaded file (the frontend's Reveal action opens its folder).
    """

    model_config = ConfigDict(strict=True)

    step: int
    remote_path: str
    local_path: str
    created_at: str = ""


class GpuStatus(BaseModel):
    """Live GPU telemetry snapshot for the run, for the GPU-status panel.

    Mirrors `services.trainer_target.GpuTelemetry` as a JSON-serializable
    pydantic model so it persists on the `TrainingJob` and survives app
    restarts. `updated_at` is the ISO timestamp of the last successful poll.
    """

    model_config = ConfigDict(strict=True)

    name: str
    vram_total_mb: int
    vram_used_mb: int
    gpu_util_pct: int
    mem_util_pct: int
    temp_c: int | None = None
    updated_at: str


# Maximum feed items retained per run. The feed is a rolling window of recent
# validation samples; older ones are trimmed to keep the job record bounded.
VALIDATION_FEED_MAX_ITEMS = 50


class TrainingJob(BaseModel):
    model_config = ConfigDict(strict=True)

    id: str
    preprocessed_id: str
    name: str
    created_at: str
    status: TrainingStatus
    config: TrainingConfig
    provider: TrainerProvider = "runpod"
    trainer_repo_url: str | None = None
    trainer_repo_ref: str | None = None
    # Remote output dir (holds lora_weights.safetensors, validation
    # samples, training_config.yaml) and the resolved local path the
    # finished LoRA was downloaded to.
    remote_output_dir: str | None = None
    local_lora_path: str | None = None
    target: TargetHandle | None = None
    # GPU the run was launched on, captured at start so the run summary reflects
    # what actually trained (not whatever the setting says later). Blank/0 for
    # older runs or non-RunPod providers.
    gpu_type: str = ""
    gpu_vram_gb: int = 0
    runpod_selection: RunpodSelection | None = None
    # Optional behavior description captured at training setup (and editable
    # later in the LoRA Library).
    description: str | None = None
    # Optional user-supplied example image/video showing what the trained LoRA
    # does (CivitAI-style library preview). Stored under the app data dir and
    # served via the example-media route; None until the user attaches one.
    example_path: str | None = None
    # Best-effort progress mirror parsed from remote training logs.
    current_step: int | None = None
    total_steps: int | None = None
    # Highest `lora_weights_step_N` the trainer has saved so far (parsed from
    # the training log during polling and persisted here so it survives a
    # reconciler restart). The download/redownload path prefers the actual
    # remote checkpoint listing (`list_checkpoints`) but falls back to this when
    # the listing is unavailable — and it's what makes a redownload to a fresh
    # pod pick a real checkpoint instead of guessing the configured final step.
    latest_checkpoint_step: int | None = None
    # Estimated seconds remaining, derived from the recent step rate. None until
    # enough steps have elapsed to estimate (and across a reconciler restart).
    eta_seconds: int | None = None
    # When the first training step was observed in the logs. The gap between
    # `started_at` (train.py launched) and this marks the silent setup phase
    # (model load from the network volume + the one-time step-0 validation).
    first_step_at: str | None = None
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    # RunPod's hourly rate captured from the live pod. The completed-run UI
    # combines it with the job interval for an explicitly estimated cost.
    compute_rate_per_hr: float | None = None
    archived_at: str | None = None
    # Workload-scoped RunPod attribution. Unlike pod uptime, this interval
    # excludes unrelated pre-run reuse and post-run idle time.
    workload_billing_started_at: str | None = None
    workload_billing_ended_at: str | None = None
    captured_hourly_rate: float | None = None
    attributed_seconds: float | None = None
    attributed_cost: float | None = None
    pod_preparation_started_at: str | None = None
    pod_preparation_ended_at: str | None = None
    training_setup_started_at: str | None = None
    training_setup_ended_at: str | None = None
    training_steps_started_at: str | None = None
    training_steps_ended_at: str | None = None
    # Retained after the live handle is cleared so a completed run can still
    # associate its banner with a stopped cached pod returned by RunPod.
    last_pod_id: str | None = None
    # User requested cancellation; the reconciler observes this on its
    # next tick, terminates the remote job, and moves to `cancelled`.
    cancel_requested: bool = False
    # Transient-failure bookkeeping for the reconciler's outer guard (mirrors
    # `PreprocessedDataset.status_detail` / `consecutive_failures`): a retryable
    # exception escaping `_reconcile_training` is surfaced on the card via
    # `status_detail` and retried up to `_TRANSIENT_FAILURE_BUDGET` ticks before
    # the handler escalates to `failed`. Cleared on the next clean tick.
    status_detail: str | None = None
    consecutive_failures: int = 0
    # User asked to re-download the trained adapter for a run that finished
    # training but failed at the download step (weights persist on the network
    # volume). The reconciler picks this up and fetches the existing artifact
    # without re-training.
    redownload_requested: bool = False
    # User asked to reset a finished run and re-train from scratch. The
    # reconciler wipes the remote output dir (checkpoints + samples) and the
    # local run folder, then re-runs `train.py` from step 0. Cleared once the
    # wipe is done. Distinguishes "reset from scratch" from "resume from the
    # last checkpoint" (which keeps the existing checkpoints).
    reset_requested: bool = False
    # Training-results feed: the validation samples configured for this run
    # (mapped to downloaded artifacts by 1-based index) and the rolling window
    # of generated samples pulled back as they train. Bounded by
    # `VALIDATION_FEED_MAX_ITEMS`.
    validation_sample_refs: list[ValidationSampleRef] = Field(default_factory=list[ValidationSampleRef])
    validation_feed: list[ValidationFeedItem] = Field(default_factory=list[ValidationFeedItem])
    # Adapter checkpoints downloaded live as they're saved (paired with the
    # validation feed by step). Bounded by `config.checkpoint_keep_last_n` so
    # local disk mirrors the remote retention the user configured.
    checkpoints: list[CheckpointArtifact] = Field(default_factory=list[CheckpointArtifact])
    # Live GPU telemetry for the run, refreshed each poll while training. None
    # until the first successful `query_gpu` (or for runs started before the
    # GPU-status feature existed).
    gpu_status: GpuStatus | None = None

    @field_validator("provider", mode="before")
    @classmethod
    def _coerce_legacy_provider(cls, value: object) -> object:
        # Coerce a now-removed provider (e.g. "coder") on legacy persisted jobs
        # to the default backend so loading never crashes; known providers pass
        # through unchanged.
        return value if value in ("runpod", "local") else "runpod"


# ----------------------------------------------------------------
# Top-level ledgers (one JSON file each)
# ----------------------------------------------------------------


class LoraDatasetsState(BaseModel):
    model_config = ConfigDict(strict=True)

    datasets: list[LoraDataset] = Field(default_factory=list[LoraDataset])
    folders: list[LoraFolder] = Field(default_factory=list[LoraFolder])
    # v1 -> v2: introduced `folders` + `dataset.folder_id`. Both default safely
    # (empty list / None), so the migration is a no-op bump — old ledgers load
    # without transformation. Kept as a version for future structural changes.
    schema_version: int = 3


class PreprocessedState(BaseModel):
    model_config = ConfigDict(strict=True)

    items: list[PreprocessedDataset] = Field(default_factory=list[PreprocessedDataset])
    schema_version: int = 1


class TrainingState(BaseModel):
    model_config = ConfigDict(strict=True)

    items: list[TrainingJob] = Field(default_factory=list[TrainingJob])
    schema_version: int = 2


SavedModelReadiness = Literal["ready", "missing", "unknown"]


class SavedModelVolumeMetadata(BaseModel):
    model_config = ConfigDict(strict=True)

    volume_id: str
    fingerprint: str
    status: SavedModelReadiness = "unknown"
    estimated_download_bytes: int | None = None
    updated_at: str


class SavedModelState(BaseModel):
    model_config = ConfigDict(strict=True)

    volumes: list[SavedModelVolumeMetadata] = Field(
        default_factory=list[SavedModelVolumeMetadata]
    )
    schema_version: int = 1


class LoraTrainingProfile(BaseModel):
    """A named, reusable bundle of `TrainingConfig` knobs.

    Profiles decouple "how to train" from "what to train": a user tunes a
    profile once and picks it when starting any run. The run snapshots the
    profile's `config`, so later edits never rewrite finished runs.
    Curated built-ins are immutable templates; users duplicate one before
    editing. Compatibility metadata keeps standard and paired IC-LoRA recipes
    from being presented as interchangeable.
    """

    model_config = ConfigDict(strict=True)

    id: str
    name: str
    created_at: str
    updated_at: str
    config: TrainingConfig
    builtin: bool = False
    description: str = ""
    dataset_types: list[LoraDatasetType] = Field(
        default_factory=lambda: ["standard", "ic_lora"]
    )
    min_vram_gb: int | None = Field(default=None, ge=1)
    auto_recommended: bool = False


class LoraTrainingProfilesState(BaseModel):
    model_config = ConfigDict(strict=True)

    profiles: list[LoraTrainingProfile] = Field(default_factory=list[LoraTrainingProfile])
    # v5 exposes only profiles documented by the official LTX-2 trainer.
    schema_version: int = 5


# Legacy built-in ids are retained solely for the v5 migration.
BUILTIN_STANDARD_ID = "builtin-standard"
BUILTIN_LOW_VRAM_ID = "builtin-low-vram"
BUILTIN_DETAILED_RANK64_ID = "builtin-detailed-rank64"
BUILTIN_LOW_VRAM_INT4_ID = "builtin-low-vram-int4"

BUILTIN_IC_LORA_ID = "builtin-ic-lora-official"


def _video_target_modules() -> list[str]:
    return [
        "attn1.to_k",
        "attn1.to_q",
        "attn1.to_v",
        "attn1.to_out.0",
        "attn2.to_k",
        "attn2.to_q",
        "attn2.to_v",
        "attn2.to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
    ]


def legacy_detailed_rank64_config() -> TrainingConfig:
    """Knobs from a hand-tuned reference/v2v run (the downloaded
    `training_config.yaml`): a rank-64 adapter over the *full* attention +
    feed-forward modules, a higher LR on a shorter schedule, frequent
    checkpoints, and 960x544 STG-v validation.

    Note: this captures the values a profile can express. The trainer
    *schema* bits in that file (the `flexible` strategy block and the
    reference-based `validation.samples`) are emitted by the config builder,
    not a profile — so selecting this profile matches the numbers but still
    renders the builder's current strategy shape.
    """
    return TrainingConfig(
        preset="standard",
        rank=64,
        alpha=64,
        target_modules=_video_target_modules(),
        learning_rate=0.00042,
        steps=400,
        stg_mode="stg_v",
        stg_blocks=[28],
        validation_video_width=960,
        validation_video_height=544,
        validation_video_frames=49,
        validation_interval=50,
        checkpoint_interval=50,
        checkpoint_keep_last_n=5,
    )


def default_training_profiles(now_iso: str) -> list[LoraTrainingProfile]:
    """Immutable recipes documented by the official LTX-2 trainer."""
    return [
        LoraTrainingProfile(
            id=BUILTIN_STANDARD_ID,
            name="Standard LoRA",
            created_at=now_iso,
            updated_at=now_iso,
            config=TrainingConfig(preset="standard"),
            builtin=True,
            description="Official LTX-2 baseline for standard LoRA training.",
            dataset_types=["standard"],
            auto_recommended=True,
        ),
        LoraTrainingProfile(
            id=BUILTIN_LOW_VRAM_ID,
            name="Low VRAM",
            created_at=now_iso,
            updated_at=now_iso,
            config=TrainingConfig(
                preset="low_vram",
                rank=16,
                alpha=16,
            ),
            builtin=True,
            description="Official memory-efficient LTX-2 LoRA recipe.",
            dataset_types=["standard"],
        ),
        LoraTrainingProfile(
            id=BUILTIN_IC_LORA_ID,
            name="IC-LoRA",
            created_at=now_iso,
            updated_at=now_iso,
            config=TrainingConfig(
                preset="standard",
                rank=32,
                alpha=32,
                target_modules=_video_target_modules(),
                learning_rate=2e-4,
                steps=3000,
                scheduler_type="linear",
                first_frame_conditioning_p=0.2,
                validation_interval=100,
            ),
            builtin=True,
            description="Official LTX-2 paired video-to-video IC-LoRA recipe.",
            dataset_types=["ic_lora"],
            auto_recommended=True,
        ),
    ]


# ----------------------------------------------------------------
# Persistence <-> API boundary converters
# ----------------------------------------------------------------
# snake_case here for on-disk + Python internals, camelCase in
# `api_types` for the HTTP surface. Converting at the route layer keeps
# either side a one-file rename, matching the queue ledger pattern.


def _handle_to_api(handle: TargetHandle | None) -> LoraTargetHandleApi | None:
    if handle is None:
        return None
    return LoraTargetHandleApi(
        provider=handle.provider,
        podId=handle.pod_id,
        remoteJobId=handle.remote_job_id,
    )


def _probe_to_api(probe: ClipProbe | None) -> LoraClipProbeApi | None:
    if probe is None:
        return None
    return LoraClipProbeApi(
        durationSeconds=probe.duration_seconds,
        width=probe.width,
        height=probe.height,
        fps=probe.fps,
        frameCount=probe.frame_count,
        hasAudio=probe.has_audio,
        videoCodec=probe.video_codec,
    )


def _edits_to_api(edits: ClipEdits | None) -> LoraClipEditsApi | None:
    if edits is None:
        return None
    return LoraClipEditsApi(
        trim=None if edits.trim is None else LoraTrimApi(
            startSeconds=edits.trim.start_seconds,
            endSeconds=edits.trim.end_seconds,
        ),
        crop=None if edits.crop is None else LoraCropApi(
            x=edits.crop.x,
            y=edits.crop.y,
            width=edits.crop.width,
            height=edits.crop.height,
        ),
        scale=None if edits.scale is None else LoraScaleApi(
            width=edits.scale.width,
            height=edits.scale.height,
        ),
        fps=edits.fps,
        speed=edits.speed,
        mute=edits.mute,
        reverse=edits.reverse,
    )


def clip_to_api(clip: LoraClip) -> LoraDatasetClipApi:
    return LoraDatasetClipApi(
        id=clip.id,
        localPath=clip.local_path,
        caption=clip.caption,
        durationSeconds=clip.duration_seconds,
        referencePath=clip.reference_path,
        referencePaths=clip.reference_paths,
        origin=clip.origin,
        probe=_probe_to_api(clip.probe),
        sourcePath=clip.source_path,
        edits=_edits_to_api(clip.edits),
        posterPath=clip.poster_path,
        spritePath=clip.sprite_path,
        spriteTiles=clip.sprite_tiles,
        triage=clip.triage,
        deletedAt=clip.deleted_at,
    )


def dataset_to_api(dataset: LoraDataset) -> LoraDatasetApi:
    return LoraDatasetApi(
        id=dataset.id,
        name=dataset.name,
        createdAt=dataset.created_at,
        status=dataset.status,
        type=dataset.type,
        workspacePolicy=dataset.workspace_policy,
        cacheVolumeId=dataset.cache_volume_id,
        triggerWord=dataset.trigger_word,
        clips=[clip_to_api(clip) for clip in dataset.clips],
        remoteDatasetDir=dataset.remote_dataset_dir,
        target=_handle_to_api(dataset.target),
        runpodSelection=runpod_selection_to_api(dataset.runpod_selection),
        error=dataset.error,
        statusDetail=dataset.status_detail,
        statusPercent=dataset.status_percent,
        statusEtaSeconds=dataset.status_eta_seconds,
        cancelRequested=dataset.cancel_requested,
        folderId=dataset.folder_id,
        updatedAt=dataset.updated_at,
        originatingProjectId=dataset.originating_project_id,
        pendingPipeline=(
            LoraPendingPipelineApi(
                name=dataset.auto_pipeline.training.name,
                description=dataset.auto_pipeline.training.description,
                resolutionBuckets=dataset.auto_pipeline.resolution_buckets,
                withAudio=dataset.auto_pipeline.with_audio,
                autoCaption=dataset.auto_pipeline.auto_caption,
                captionerType=dataset.auto_pipeline.captioner_type,
                config=training_config_to_api(dataset.auto_pipeline.training.config),
            )
            if dataset.auto_pipeline is not None
            else None
        ),
        archivedAt=dataset.archived_at,
        keepAliveUntil=dataset.keep_alive_until,
        finalActivityAt=dataset.final_activity_at,
        releaseStatus=dataset.release_status,
        releaseError=dataset.release_error,
        releaseAttemptedAt=dataset.release_attempted_at,
        lastPodId=dataset.last_pod_id,
    )


def folder_to_api(folder: LoraFolder) -> LoraFolderApi:
    return LoraFolderApi(id=folder.id, name=folder.name, parentId=folder.parent_id)


def datasets_state_to_api(state: LoraDatasetsState) -> LoraDatasetsResponse:
    return LoraDatasetsResponse(
        datasets=[dataset_to_api(d) for d in state.datasets],
        folders=[folder_to_api(f) for f in state.folders],
        schemaVersion=state.schema_version,
    )


def preprocessed_to_api(item: PreprocessedDataset) -> LoraPreprocessedApi:
    return LoraPreprocessedApi(
        id=item.id,
        datasetId=item.dataset_id,
        createdAt=item.created_at,
        status=item.status,
        resolutionBuckets=item.resolution_buckets,
        effectiveResolutionBuckets=item.effective_resolution_buckets,
        withAudio=item.with_audio,
        autoCaption=item.auto_caption,
        captionerType=item.captioner_type,
        remotePrecomputedDir=item.remote_precomputed_dir,
        target=_handle_to_api(item.target),
        error=item.error,
        startedAt=item.started_at,
        completedAt=item.completed_at,
        cancelRequested=item.cancel_requested,
        statusDetail=item.status_detail,
    )


def preprocessed_state_to_api(state: PreprocessedState) -> LoraPreprocessedResponse:
    return LoraPreprocessedResponse(
        items=[preprocessed_to_api(i) for i in state.items],
        schemaVersion=state.schema_version,
    )


def training_config_to_api(config: TrainingConfig) -> LoraTrainingConfigApi:
    return LoraTrainingConfigApi(
        preset=config.preset,
        rank=config.rank,
        alpha=config.alpha,
        dropout=config.dropout,
        targetModules=list(config.target_modules),
        learningRate=config.learning_rate,
        steps=config.steps,
        batchSize=config.batch_size,
        gradientAccumulationSteps=config.gradient_accumulation_steps,
        maxGradNorm=config.max_grad_norm,
        optimizerType=config.optimizer_type,
        schedulerType=config.scheduler_type,
        enableGradientCheckpointing=config.enable_gradient_checkpointing,
        mixedPrecisionMode=config.mixed_precision_mode,
        quantization=config.quantization,
        loadTextEncoderIn8bit=config.load_text_encoder_in_8bit,
        offloadOptimizerDuringValidation=config.offload_optimizer_during_validation,
        numDataloaderWorkers=config.num_dataloader_workers,
        firstFrameConditioningP=config.first_frame_conditioning_p,
        validationPrompts=list(config.validation_prompts),
        validationNegativePrompt=config.validation_negative_prompt,
        validationVideoWidth=config.validation_video_width,
        validationVideoHeight=config.validation_video_height,
        validationVideoFrames=config.validation_video_frames,
        validationFrameRate=config.validation_frame_rate,
        validationInferenceSteps=config.validation_inference_steps,
        validationInterval=config.validation_interval,
        validationGuidanceScale=config.validation_guidance_scale,
        validationSeed=config.validation_seed,
        stgScale=config.stg_scale,
        stgBlocks=list(config.stg_blocks),
        stgMode=config.stg_mode,
        skipInitialValidation=config.skip_initial_validation,
        checkpointInterval=config.checkpoint_interval,
        checkpointKeepLastN=config.checkpoint_keep_last_n,
        checkpointPrecision=config.checkpoint_precision,
        timestepSamplingMode=config.timestep_sampling_mode,
        pushToHub=config.push_to_hub,
        hubModelId=config.hub_model_id,
        wandbEnabled=config.wandb_enabled,
        seed=config.seed,
        loadCheckpoint=config.load_checkpoint,
        withAudio=config.with_audio,
        triggerWord=config.trigger_word,
    )


def training_config_from_api(api: LoraTrainingConfigApi) -> TrainingConfig:
    """Build a `TrainingConfig` from its API DTO (camel -> snake).

    Centralizes the field mapping so routes stay thin. `with_audio` and
    `trigger_word` are accepted but the handler may override `with_audio`
    from the preprocessed dataset.
    """
    return TrainingConfig(
        preset="low_vram" if api.preset == "low_vram" else "standard",
        rank=api.rank,
        alpha=api.alpha,
        dropout=api.dropout,
        target_modules=list(api.targetModules),
        learning_rate=api.learningRate,
        steps=api.steps,
        batch_size=api.batchSize,
        gradient_accumulation_steps=api.gradientAccumulationSteps,
        max_grad_norm=api.maxGradNorm,
        optimizer_type=api.optimizerType,
        scheduler_type=api.schedulerType,
        enable_gradient_checkpointing=api.enableGradientCheckpointing,
        mixed_precision_mode=api.mixedPrecisionMode,
        quantization=api.quantization,
        load_text_encoder_in_8bit=api.loadTextEncoderIn8bit,
        offload_optimizer_during_validation=api.offloadOptimizerDuringValidation,
        num_dataloader_workers=api.numDataloaderWorkers,
        first_frame_conditioning_p=api.firstFrameConditioningP,
        validation_prompts=list(api.validationPrompts),
        validation_negative_prompt=api.validationNegativePrompt,
        validation_video_width=api.validationVideoWidth,
        validation_video_height=api.validationVideoHeight,
        validation_video_frames=api.validationVideoFrames,
        validation_frame_rate=api.validationFrameRate,
        validation_inference_steps=api.validationInferenceSteps,
        validation_interval=api.validationInterval,
        validation_guidance_scale=api.validationGuidanceScale,
        validation_seed=api.validationSeed,
        stg_scale=api.stgScale,
        stg_blocks=list(api.stgBlocks),
        stg_mode=api.stgMode,
        skip_initial_validation=api.skipInitialValidation,
        checkpoint_interval=api.checkpointInterval,
        checkpoint_keep_last_n=api.checkpointKeepLastN,
        checkpoint_precision=api.checkpointPrecision,
        timestep_sampling_mode=api.timestepSamplingMode,
        push_to_hub=api.pushToHub,
        hub_model_id=api.hubModelId,
        wandb_enabled=api.wandbEnabled,
        seed=api.seed,
        load_checkpoint=api.loadCheckpoint,
        with_audio=api.withAudio,
        trigger_word=api.triggerWord,
    )


def profile_to_api(profile: LoraTrainingProfile) -> LoraTrainingProfileApi:
    return LoraTrainingProfileApi(
        id=profile.id,
        name=profile.name,
        createdAt=profile.created_at,
        updatedAt=profile.updated_at,
        config=training_config_to_api(profile.config),
        builtin=profile.builtin,
        description=profile.description,
        datasetTypes=list(profile.dataset_types),
        minVramGb=profile.min_vram_gb,
        autoRecommended=profile.auto_recommended,
    )


def profiles_state_to_api(
    state: LoraTrainingProfilesState,
) -> LoraTrainingProfilesResponse:
    return LoraTrainingProfilesResponse(
        profiles=[profile_to_api(p) for p in state.profiles],
        schemaVersion=state.schema_version,
    )


def _validation_media_url(
    job_id: str, step: int, sample_index: int, extension: str
) -> str:
    """Browser-loadable URL for a feed item's media, served by the secure
    feed-media route (the route looks up the path server-side from the job's
    feed, so no filesystem path is exposed to the client)."""
    return (
        f"/api/lora/training/{job_id}/validation-media"
        f"?step={step}&sampleIndex={sample_index}&extension={extension}"
    )


def _validation_feed_item_to_api(
    job_id: str, item: ValidationFeedItem
) -> LoraValidationFeedItemApi:
    return LoraValidationFeedItemApi(
        step=item.step,
        sampleIndex=item.sample_index,
        mediaUrl=_validation_media_url(
            job_id, item.step, item.sample_index, item.extension
        ),
        mediaType="audio" if item.extension == "wav" else "video",
        source=item.source,
        prompt=item.prompt,
        referenceMediaUrl=(
            None if item.reference_local_path is None else "staged"
        ),
        createdAt=item.created_at,
    )


def _checkpoint_to_api(item: CheckpointArtifact) -> LoraCheckpointArtifactApi:
    return LoraCheckpointArtifactApi(
        step=item.step,
        localPath=item.local_path,
        createdAt=item.created_at,
    )


def _gpu_status_to_api(status: GpuStatus) -> LoraGpuStatusApi:
    return LoraGpuStatusApi(
        name=status.name,
        vramTotalMb=status.vram_total_mb,
        vramUsedMb=status.vram_used_mb,
        gpuUtilPct=status.gpu_util_pct,
        memUtilPct=status.mem_util_pct,
        tempC=status.temp_c,
        updatedAt=status.updated_at,
    )


def training_job_to_api(job: TrainingJob) -> LoraTrainingJobApi:
    return LoraTrainingJobApi(
        id=job.id,
        preprocessedId=job.preprocessed_id,
        name=job.name,
        createdAt=job.created_at,
        status=job.status,
        config=training_config_to_api(job.config),
        provider=job.provider,
        description=job.description,
        remoteOutputDir=job.remote_output_dir,
        localLoraPath=job.local_lora_path,
        target=_handle_to_api(job.target),
        currentStep=job.current_step,
        totalSteps=job.total_steps,
        etaSeconds=job.eta_seconds,
        gpuType=job.gpu_type,
        gpuVramGb=job.gpu_vram_gb,
        runpodSelection=runpod_selection_to_api(job.runpod_selection),
        firstStepAt=job.first_step_at,
        error=job.error,
        startedAt=job.started_at,
        completedAt=job.completed_at,
        computeRatePerHr=job.compute_rate_per_hr,
        archivedAt=job.archived_at,
        workloadBillingStartedAt=job.workload_billing_started_at,
        workloadBillingEndedAt=job.workload_billing_ended_at,
        capturedHourlyRate=job.captured_hourly_rate,
        attributedSeconds=job.attributed_seconds,
        attributedCost=job.attributed_cost,
        podPreparationStartedAt=job.pod_preparation_started_at,
        podPreparationEndedAt=job.pod_preparation_ended_at,
        trainingSetupStartedAt=job.training_setup_started_at,
        trainingSetupEndedAt=job.training_setup_ended_at,
        trainingStepsStartedAt=job.training_steps_started_at,
        trainingStepsEndedAt=job.training_steps_ended_at,
        lastPodId=job.last_pod_id,
        cancelRequested=job.cancel_requested,
        statusDetail=job.status_detail,
        validationFeed=[
            _validation_feed_item_to_api(job.id, i) for i in job.validation_feed
        ],
        checkpoints=[_checkpoint_to_api(c) for c in job.checkpoints],
        gpuStatus=_gpu_status_to_api(job.gpu_status) if job.gpu_status else None,
    )


def runpod_selection_to_api(
    selection: RunpodSelection | None,
) -> RunpodSelectionApi | None:
    if selection is None:
        return None
    return RunpodSelectionApi(
        gpuType=selection.gpu_type,
        gpuVramGb=selection.gpu_vram_gb,
        datacenter=selection.datacenter,
        workspacePolicy=selection.workspace_policy,
        volumeId=selection.volume_id,
    )


def runpod_selection_from_api(api: RunpodSelectionApi) -> RunpodSelection:
    return RunpodSelection(
        gpu_type=api.gpuType,
        gpu_vram_gb=api.gpuVramGb,
        datacenter=api.datacenter,
        workspace_policy=api.workspacePolicy,
        volume_id=api.volumeId,
    )


def training_state_to_api(state: TrainingState) -> LoraTrainingResponse:
    return LoraTrainingResponse(
        items=[training_job_to_api(i) for i in state.items],
        schemaVersion=state.schema_version,
    )

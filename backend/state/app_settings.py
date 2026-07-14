"""Canonical app settings schema and patch models."""

from __future__ import annotations

from typing import Any, TypeGuard, TypeVar, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator, model_validator

from api_types import LoraProviderApi

OFFICIAL_LORA_TRAINER_REPO = "https://github.com/Lightricks/LTX-2.git"
PINNED_LORA_TRAINER_REVISION = "9377758131b1ffde4b7f766804590a6617bf2ab9"


def _to_camel_case(field_name: str) -> str:
    special_aliases = {
        "prompt_enhancer_enabled_t2v": "promptEnhancerEnabledT2V",
        "prompt_enhancer_enabled_i2v": "promptEnhancerEnabledI2V",
    }
    if field_name in special_aliases:
        return special_aliases[field_name]

    head, *tail = field_name.split("_")
    return head + "".join(part.title() for part in tail)


def _clamp_int(value: Any, minimum: int, maximum: int, default: int) -> int:
    if value is None:
        return default

    parsed = int(value)
    return max(minimum, min(maximum, parsed))


class SettingsBaseModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel_case,
        populate_by_name=True,
        validate_assignment=True,
        extra="ignore",
    )


class SettingsPatchModel(SettingsBaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel_case,
        populate_by_name=True,
        validate_assignment=True,
        extra="forbid",
    )


class AppSettings(SettingsBaseModel):
    use_torch_compile: bool = False
    ltx_api_key: str = ""
    user_prefers_ltx_api_video_generations: bool = False
    fal_api_key: str = ""
    use_local_text_encoder: bool = False
    # Opt-in higher-quality IC-LoRA base: when true AND the dev checkpoint +
    # distilled v1.1 LoRA are downloaded, IC-LoRA generations load
    # `ltx-2.3-22b-dev` with the distilled LoRA stacked @0.5 instead of the
    # distilled checkpoint — matching the ComfyUI dev + distilled-LoRA flow.
    # Adds ~54 GB of optional downloads; off by default.
    use_dev_quality_base: bool = False
    prompt_cache_size: int = 100
    prompt_enhancer_enabled_t2v: bool = True
    prompt_enhancer_enabled_i2v: bool = False
    gemini_api_key: str = ""
    # Pexels API key (BYOK) for the LoRA trainer's stock-media browser.
    # Masked in SettingsResponse like the other secrets.
    pexels_api_key: str = ""
    seed_locked: bool = False
    locked_seed: int = 42
    models_dir: str = ""

    # ---- LoRA trainer (BYOK remote GPU) ----
    # Secret (masked in SettingsResponse, never echoed back): the RunPod REST
    # API key. Secrets remain in memory while the backend runs but are persisted
    # only in the authenticated credential vault, never settings.json.
    runpod_api_key: str = ""
    # Manual HuggingFace token (BYOK) for downloading gated weights — chiefly
    # the Google Gemma text encoder, which requires license approval. The app's
    # HF OAuth is disabled in this build (HF_GATING_ENABLED=false), so this is
    # the path to authenticated gated downloads. Masked in SettingsResponse.
    hf_token: str = ""
    # Non-secret config: the remote paths the LTX-2 checkpoint and Gemma encoder
    # live at on the GPU host (both must be local paths on the remote — the
    # trainer rejects URLs). `lora_remote_workspace_dir` is the base dir remote
    # jobs use for datasets/outputs (e.g. /workspace). `runpod_gpu_type` is the
    # RunPod GPU type id requested when creating an on-demand pod.
    # Remote checkpoint/encoder paths. Left blank by default: the runner
    # derives them from `lora_remote_workspace_dir` (see
    # lora_command_builder.default_model_path / default_text_encoder_path)
    # so a connected RunPod user never has to enter them. Non-empty values
    # act as an Advanced override (e.g. a pre-baked image with weights at a
    # fixed path).
    lora_remote_model_path: str = ""
    lora_remote_text_encoder_path: str = ""
    lora_remote_workspace_dir: str = "/workspace"
    runpod_gpu_type: str = ""
    # VRAM (GB) of the selected `runpod_gpu_type`, recorded when the user
    # picks a GPU so training can auto-match the preset to the hardware
    # (block <32 GB, low-VRAM preset below 80 GB) without a network lookup.
    # 0 means unknown (skip auto-match).
    runpod_gpu_vram_gb: int = 0
    # Nano Banana tier used by the dataset-prep frame editor (Fal, BYOK
    # via `fal_api_key`). Default balances speed/cost; the UI exposes a
    # dropdown to switch to the higher-quality "pro" tier.
    lora_nano_banana_model: str = "nano-banana-2"
    # Max simultaneous in-flight Fal requests for the LoRA trainer's
    # background derivation pipeline (bulk Nano Banana / Kling generation).
    # Higher = faster bulk runs but more likely to hit Fal's per-account
    # rate limits (transient failures auto-retry). Local GPU drives stay
    # single-flight regardless of this value. Clamped 1..20.
    lora_fal_concurrency: int = 20
    # ---- RunPod auto-provisioning ----
    # When True the reconciler bootstraps a freshly-created RunPod pod
    # before its first upload: it installs the LTX-2 trainer and (if HF
    # repos are configured below) downloads the base model + text encoder.
    # The bootstrap is gated by an idempotent on-disk marker so a reused
    # pod — or a network volume — only pays the install/download cost
    # once. False keeps the prior behaviour (pod assumed pre-baked).
    lora_auto_provision: bool = True
    # Git source for the unified LTX-2 trainer cloned during provisioning.
    # Release builds execute only this audited repository at the immutable
    # revision below; settings cannot redirect provisioning to arbitrary code.
    lora_trainer_repo_url: str = OFFICIAL_LORA_TRAINER_REPO
    # Immutable upstream revision whose trainer CLI/config contract this app
    # has been verified against. Do not point release builds at moving `main`.
    lora_trainer_repo_ref: str = PINNED_LORA_TRAINER_REVISION
    # HuggingFace repo ids for the base checkpoint + Gemma text encoder,
    # defaulted to the canonical LTX-2 weights so a connected user never
    # has to find them. Both repos are gated (LTX-2 model license +
    # Google Gemma license): provisioning passes the app's HuggingFace
    # OAuth token, so the user must be logged in to HF and have accepted
    # both licenses. Provisioning downloads only `lora_model_checkpoint_file`
    # from the model repo (the repo is ~314 GB; the trainer needs one file)
    # and the whole encoder repo.
    # The 22B checkpoint lives in the LTX-2.3 repo (the LTX-2 repo holds the
    # 19B line). Repo is public/un-gated, so no token needed for the model.
    lora_model_hf_repo: str = "Lightricks/LTX-2.3"
    lora_model_checkpoint_file: str = "ltx-2.3-22b-dev.safetensors"
    lora_text_encoder_hf_repo: str = "google/gemma-3-12b-it-qat-q4_0-unquantized"
    # Optional RunPod overrides: a custom container image (empty = the
    # target's built-in default) and a persistent network volume id mounted
    # at the workspace dir so the trainer install + model cache survive pod
    # teardown — the key to making repeat runs cheap.
    runpod_image: str = ""
    runpod_network_volume_id: str = ""
    # Default OFF: a network volume pins the datacenter, and GPU stock varies
    # by region, so caching can strand the user in a region with no available
    # GPU. Off = ephemeral pods that launch in ANY region with the cheapest
    # in-stock GPU (re-downloads the weights per cold pod). Power users can
    # turn it on for fast repeat runs once they're in a GPU-rich region.
    runpod_keep_model_cached: bool = False
    # Size of the auto-created network volume. Budget: ~44GB checkpoint +
    # ~24GB encoder + ~15GB venv ≈ 84GB of fixed cost, leaving the rest for
    # datasets, precomputed latents (the variable cost), and output LoRAs.
    # 250GB default: RunPod's minimum is enough for the ~84GB fixed payload and
    # typical datasets while keeping standing storage cost down. Users can pick
    # 500GB+ for many datasets/checkpoints. Clamped 250..4000. Only used when a
    # fresh volume is created; existing volumes are not resized automatically.
    runpod_volume_size_gb: int = 250
    # Auto-stop an idle training pod this many minutes after its last job
    # completes, to cap per-minute billing. 0 disables idle auto-stop
    # (pod runs until the dataset is deleted). Clamped 0..240.
    runpod_idle_stop_minutes: int = 10
    # Backend that new training runs use. "runpod" (default) keeps the cloud
    # GPU flow unchanged for everyone; "local" routes runs to the WSL2 trainer.
    # Persisted so the trainer's top-right provider pill remembers the choice
    # and every run inherits it. Same Literal as the per-request `provider`.
    lora_provider: LoraProviderApi = "runpod"

    @field_validator("prompt_cache_size", mode="before")
    @classmethod
    def _clamp_prompt_cache_size(cls, value: Any) -> int:
        return _clamp_int(value, minimum=0, maximum=1000, default=100)

    @field_validator("locked_seed", mode="before")
    @classmethod
    def _clamp_locked_seed(cls, value: Any) -> int:
        return _clamp_int(value, minimum=0, maximum=2_147_483_647, default=42)

    @field_validator("lora_fal_concurrency", mode="before")
    @classmethod
    def _clamp_lora_fal_concurrency(cls, value: Any) -> int:
        return _clamp_int(value, minimum=1, maximum=20, default=20)

    @field_validator("runpod_idle_stop_minutes", mode="before")
    @classmethod
    def _clamp_runpod_idle_stop_minutes(cls, value: Any) -> int:
        return _clamp_int(value, minimum=0, maximum=240, default=10)

    @field_validator("runpod_volume_size_gb", mode="before")
    @classmethod
    def _clamp_runpod_volume_size_gb(cls, value: Any) -> int:
        # Floor at 80GB (models alone need ~68GB); generous ceiling.
        # Floor at 250GB: below that the model+encoder+venv+temp don't reliably
        # fit, so an old persisted 150 gets clamped up rather than failing.
        return _clamp_int(value, minimum=250, maximum=4000, default=500)

    @field_validator("lora_trainer_repo_url", mode="before")
    @classmethod
    def _migrate_legacy_trainer_repo(cls, value: Any) -> Any:
        # An early build of the LoRA trainer defaulted to the old LTX-Video
        # trainer repo, which has the wrong layout for LTX-2 (no
        # packages/ltx-trainer). Coerce any persisted value pointing at it
        # back to the LTX-2 monorepo so legacy settings files self-heal on
        # load instead of cloning a repo that can't provision.
        if isinstance(value, str) and "LTX-Video-Trainer" in value:
            return OFFICIAL_LORA_TRAINER_REPO
        return value

    @model_validator(mode="after")
    def _pin_official_trainer_revision(self) -> "AppSettings":
        object.__setattr__(self, "lora_trainer_repo_url", OFFICIAL_LORA_TRAINER_REPO)
        object.__setattr__(
            self, "lora_trainer_repo_ref", PINNED_LORA_TRAINER_REVISION
        )
        return self

    @field_validator("lora_model_hf_repo", mode="before")
    @classmethod
    def _migrate_model_repo(cls, value: Any) -> Any:
        # The 22B checkpoint lives in LTX-2.3; an earlier build defaulted to
        # the LTX-2 (19B) repo, which doesn't contain `ltx-2.3-22b-dev`. Coerce
        # that specific stale value so persisted settings self-heal.
        if isinstance(value, str) and value.strip() == "Lightricks/LTX-2":
            return "Lightricks/LTX-2.3"
        return value


SettingsModelT = TypeVar("SettingsModelT", bound=SettingsBaseModel)
_PARTIAL_MODEL_CACHE: dict[type[SettingsBaseModel], type[SettingsPatchModel]] = {}


def _wrap_optional(annotation: Any) -> Any:
    if type(None) in get_args(annotation):
        return annotation
    return annotation | None


def _to_partial_annotation(annotation: Any) -> Any:
    if _is_settings_model_annotation(annotation):
        return make_partial_model(annotation)
    return annotation


def make_partial_model(model: type[SettingsModelT]) -> type[SettingsPatchModel]:
    cached = _PARTIAL_MODEL_CACHE.get(model)
    if cached is not None:
        return cached

    fields: dict[str, tuple[Any, Any]] = {}
    for field_name, field_info in model.model_fields.items():
        partial_annotation = _wrap_optional(_to_partial_annotation(field_info.annotation))
        fields[field_name] = (partial_annotation, Field(default=None))

    partial_model = create_model(
        f"{model.__name__}Patch",
        __base__=SettingsPatchModel,
        **cast(Any, fields),
    )

    _PARTIAL_MODEL_CACHE[model] = partial_model
    return partial_model


def _is_settings_model_annotation(annotation: object) -> TypeGuard[type[SettingsBaseModel]]:
    return isinstance(annotation, type) and issubclass(annotation, SettingsBaseModel)


AppSettingsPatch = make_partial_model(AppSettings)
UpdateSettingsRequest = AppSettingsPatch


class SettingsResponse(SettingsBaseModel):
    use_torch_compile: bool = False
    has_ltx_api_key: bool = False
    user_prefers_ltx_api_video_generations: bool = False
    has_fal_api_key: bool = False
    use_local_text_encoder: bool = False
    use_dev_quality_base: bool = False
    prompt_cache_size: int = 100
    prompt_enhancer_enabled_t2v: bool = True
    prompt_enhancer_enabled_i2v: bool = False
    has_gemini_api_key: bool = False
    has_pexels_api_key: bool = False
    seed_locked: bool = False
    locked_seed: int = 42
    models_dir: str = ""
    # LoRA trainer: secrets are surfaced as has_* booleans only; the
    # non-secret config fields pass through verbatim for the UI to edit.
    has_runpod_api_key: bool = False
    has_hf_token: bool = False
    lora_remote_model_path: str = ""
    lora_remote_text_encoder_path: str = ""
    lora_remote_workspace_dir: str = "/workspace"
    runpod_gpu_type: str = ""
    runpod_gpu_vram_gb: int = 0
    lora_nano_banana_model: str = "nano-banana-2"
    lora_fal_concurrency: int = 20
    lora_auto_provision: bool = True
    lora_trainer_repo_url: str = OFFICIAL_LORA_TRAINER_REPO
    lora_trainer_repo_ref: str = PINNED_LORA_TRAINER_REVISION
    lora_model_hf_repo: str = "Lightricks/LTX-2.3"
    lora_model_checkpoint_file: str = "ltx-2.3-22b-dev.safetensors"
    lora_text_encoder_hf_repo: str = "google/gemma-3-12b-it-qat-q4_0-unquantized"
    runpod_image: str = ""
    runpod_network_volume_id: str = ""
    runpod_keep_model_cached: bool = False
    runpod_volume_size_gb: int = 250
    runpod_idle_stop_minutes: int = 10
    lora_provider: LoraProviderApi = "runpod"


def to_settings_response(settings: AppSettings) -> SettingsResponse:
    data = settings.model_dump(by_alias=False)
    ltx_key = data.pop("ltx_api_key", "")
    fal_key = data.pop("fal_api_key", "")
    gemini_key = data.pop("gemini_api_key", "")
    pexels_key = data.pop("pexels_api_key", "")
    runpod_key = data.pop("runpod_api_key", "")
    hf_token = data.pop("hf_token", "")
    data["has_ltx_api_key"] = bool(ltx_key)
    data["has_fal_api_key"] = bool(fal_key)
    data["has_gemini_api_key"] = bool(gemini_key)
    data["has_pexels_api_key"] = bool(pexels_key)
    data["has_runpod_api_key"] = bool(runpod_key)
    data["has_hf_token"] = bool(hf_token)
    # models_dir + lora_* config fields pass through as-is (not secret)
    return SettingsResponse.model_validate(data)


def should_video_generate_with_ltx_api(*, force_api_generations: bool, settings: AppSettings) -> bool:
    has_ltx_api_key = bool(settings.ltx_api_key.strip())
    return force_api_generations or (
        settings.user_prefers_ltx_api_video_generations and has_ltx_api_key
    )

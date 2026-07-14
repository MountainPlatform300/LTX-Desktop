"""Build an LTX-2 trainer YAML config from the app's small knob set.

The trainer consumes a structured `LtxTrainerConfig` YAML (see the
configuration-reference docs and the shipped `ltx2_av_lora.yaml` /
`ltx2_av_lora_low_vram.yaml` examples). We don't expose that whole
surface to non-technical users — instead we pick a preset base
(`standard` rank 32, or `low_vram` rank 16 + int8-quanto + adamw8bit)
and override only the few values the UI collects.

This module is intentionally pure: it returns a YAML string given
fully-resolved remote paths, so it's trivial to unit-test and the
runner stays a thin orchestrator. We emit YAML by hand (rather than
pulling in PyYAML) because the document is small, fixed-shape, and the
values are simple scalars/strings — and the backend already avoids
extra deps for one-off serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from state.lora_training_state import (
    LoraDatasetType,
    TrainingConfig,
    TrainingPreset,
    ValidationSampleRef,
    ValidationSampleSource,
)


class ValidationConditionDict(TypedDict):
    """One condition on a `validation.samples` entry (today: IC-LoRA reference)."""

    type: str
    video: str
    downscale_factor: int
    temporal_scale_factor: int
    include_in_output: bool


class ValidationSampleDict(TypedDict):
    """A `validation.samples` entry: a prompt plus zero or more conditions."""

    prompt: str
    conditions: list[ValidationConditionDict]


# Default LoRA target modules — short patterns matching ALL attention (video,
# audio, and cross-modal). The right default for joint audio-video standard
# LoRAs (matches configs/t2v_lora_low_vram.yaml). Kept as a tuple to compare
# against a config's list regardless of ordering identity.
_DEFAULT_TARGET_MODULES: tuple[str, ...] = ("to_k", "to_q", "to_v", "to_out.0")

# IC-LoRA is video-only, so it targets the explicit VIDEO transformer modules
# (self-attention, cross-attention to text, and feed-forward) rather than the
# short patterns that also hit the audio + audio-video cross-attention modules.
# Matches configs/v2v_ic_lora.yaml exactly; substituted only when the user
# hasn't customized `target_modules` (a custom list is honored as-is).
_IC_LORA_TARGET_MODULES: tuple[str, ...] = (
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
)


def _yaml_str(value: str) -> str:
    """Quote a scalar for YAML, escaping the few chars that matter.

    Trigger words and paths are simple, but a path with a colon or a
    trigger with a quote would break a bare scalar. Double-quote and
    backslash-escape to stay safe without a YAML lib.
    """
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _yb(value: bool) -> str:
    return "true" if value else "false"


def _yaml_str_block(values: list[str], indent: str) -> str:
    """Render a YAML block sequence of quoted strings (one per line)."""
    return "".join(f"{indent}- {_yaml_str(v)}\n" for v in values)


def _yaml_int_inline(values: list[int]) -> str:
    """Render an inline YAML flow sequence of ints, e.g. ``[29, 30]``."""
    return "[" + ", ".join(str(v) for v in values) + "]"


def _yaml_samples_block(samples: list[ValidationSampleDict]) -> str:
    """Render the `validation.samples` YAML block (trainer's new format).

    Each sample is ``{prompt, conditions: [...]}``; a reference condition is
    ``{type, video, downscale_factor, temporal_scale_factor, include_in_output}``.
    An empty list renders as ``samples: []`` so the trainer's deprecated-
    format converter no-ops and the prompt-embedding cache load is skipped.
    """
    if not samples:
        return "  samples: []\n"
    lines: list[str] = ["  samples:\n"]
    for sample in samples:
        lines.append(f"    - prompt: {_yaml_str(sample['prompt'])}\n")
        conditions = sample["conditions"]
        if not conditions:
            lines.append("      conditions: []\n")
            continue
        lines.append("      conditions:\n")
        for cond in conditions:
            lines.append(f"        - type: {_yaml_str(cond['type'])}\n")
            lines.append(f"          video: {_yaml_str(cond['video'])}\n")
            lines.append(f"          downscale_factor: {cond['downscale_factor']}\n")
            lines.append(
                f"          temporal_scale_factor: {cond['temporal_scale_factor']}\n"
            )
            lines.append(
                f"          include_in_output: {_yb(cond['include_in_output'])}\n"
            )
    return "".join(lines)


def _preset_acceleration(preset: TrainingPreset) -> tuple[str, str, bool, bool]:
    """(quantization, optimizer_type, load_te_8bit, offload_optimizer).

    Mirrors the official `configs/t2v_lora_low_vram.yaml`, which is the
    LTX-2 trainer's recommended config for a 32 GB GPU (it literally says
    "Recommended for GPUs with 32GB VRAM (e.g., RTX 5090)"): int8-quanto
    base + an 8-bit bitsandbytes text encoder + an 8-bit optimizer, with
    optimizer offload during validation. int8-quanto works on Blackwell
    (sm_120) once its `quanto_cuda` extension is built — local provisioning
    installs gcc-14 for that build (WSL's gcc-15 is rejected by nvcc) and
    pins torch to the cu128 index for sm_120 kernels.
    """
    if preset == "low_vram":
        return ("int8-quanto", "adamw8bit", True, True)
    return ("null", "adamw", False, False)


def preset_loads_text_encoder_in_8bit(preset: TrainingPreset) -> bool:
    """Whether the preset loads the text encoder in 8-bit (bitsandbytes).

    Single source of truth for the text-encoder precision implied by a preset,
    shared by the training YAML (`build_training_yaml`) and the preprocess
    command (`process_dataset_command`) so both stages agree. The 23 GB Gemma3
    text encoder OOMs a 32 GB GPU in bf16, so `low_vram` loads it in 8-bit.
    """
    return _preset_acceleration(preset)[2]


# Spatial downscale applied to IC-LoRA reference latents at preprocess time for
# the `low_vram` preset. IC-LoRA concatenates clean reference + noised target
# tokens into one sequence, so self-attention is O((ref+target)^2): full-size
# references (~2x target tokens) blow up the backward recompute on a 32 GB card
# carrying ~22 GB of int8 weights. Halving reference spatial resolution cuts
# reference tokens ~4x (combined -> ~1.25x target), which is what lets IC-LoRA
# fit. The official `process_dataset.py --reference-downscale-factor` is the
# supported lever, and the `flexible` strategy's `reference` condition infers
# the factor from the ref/target dims at train time — so this is preprocess-only;
# no training-YAML change. `standard` (80 GB+ cards) keeps full-size references,
# matching the
# official `configs/v2v_ic_lora.yaml` (downscale_factor: 1).
IC_LORA_LOW_VRAM_REFERENCE_DOWNSCALE_FACTOR = 2


def preset_reference_downscale_factor(preset: TrainingPreset) -> int:
    """Spatial downscale factor for IC-LoRA reference latents for a preset.

    Returns 2 for `low_vram` (halve reference resolution so IC-LoRA's doubled
    sequence fits a 32 GB GPU) and 1 for `standard` (full-size references, per
    the official v2v config). Callers must gate this on the dataset being an
    IC-LoRA — text-to-video has no reference latents, so the flag is irrelevant.
    """
    if preset == "low_vram":
        return IC_LORA_LOW_VRAM_REFERENCE_DOWNSCALE_FACTOR
    return 1


@dataclass(frozen=True)
class ValidationSampleSpec:
    """One validation sample the trainer should generate during training.

    `prompt` is the caption/text prompt; `reference_video_path` is the absolute
    remote path of a reference video for IC-LoRA (``None`` for a text-to-video
    sample). The runner builds these from the user's validation prompts and the
    dataset's held-out clips; `build_validation_sample_dicts` turns them into
    the trainer's `validation.samples` YAML entries.
    """

    prompt: str
    reference_video_path: str | None = None


@dataclass(frozen=True)
class _ResolvedValidationSample:
    """Internal: one resolved validation sample before splitting into dict/ref.

    `build_validation_sample_dicts` emits the YAML dicts; the runner stores the
    `ValidationSampleRef` view (prompt + source) on the job so it can map a
    downloaded artifact back to its source. Keeping both views off one
    resolution guarantees the 1-based sample index the trainer uses lines up
    across the YAML and the feed.
    """

    prompt: str
    conditions: list[ValidationConditionDict]
    source: ValidationSampleSource
    reference_video_path: str | None


def _resolve_validation_samples(
    *,
    prompt_samples: list[str],
    holdout: list[ValidationSampleSpec],
    dataset_type: LoraDatasetType,
    reference_downscale_factor: int,
) -> list[_ResolvedValidationSample]:
    """Shared resolution behind `build_validation_sample_dicts` / `_refs`.

    Text-to-video: one prompt-only sample per user prompt + one per held-out
    clip's caption. IC-LoRA: one reference-conditioned sample per held-out clip
    (prompt = caption, reference = the clip's reference video); bare user
    prompts are dropped because IC-LoRA validation requires a reference video.
    The reference condition's `downscale_factor` mirrors the preprocess factor
    (the official v2v config requires this match).
    """
    resolved: list[_ResolvedValidationSample] = []
    is_ic_lora = dataset_type == "ic_lora"

    if not is_ic_lora:
        for prompt in prompt_samples:
            resolved.append(
                _ResolvedValidationSample(
                    prompt=prompt,
                    conditions=[],
                    source="prompt",
                    reference_video_path=None,
                )
            )

    for spec in holdout:
        if not spec.prompt.strip():
            continue
        if is_ic_lora:
            if not spec.reference_video_path:
                continue
            resolved.append(
                _ResolvedValidationSample(
                    prompt=spec.prompt,
                    conditions=[
                        {
                            "type": "reference",
                            "video": spec.reference_video_path,
                            "downscale_factor": reference_downscale_factor,
                            "temporal_scale_factor": 1,
                            "include_in_output": True,
                        }
                    ],
                    source="holdout",
                    reference_video_path=spec.reference_video_path,
                )
            )
        else:
            resolved.append(
                _ResolvedValidationSample(
                    prompt=spec.prompt,
                    conditions=[],
                    source="holdout",
                    reference_video_path=spec.reference_video_path,
                )
            )
    return resolved


def build_validation_sample_dicts(
    *,
    prompt_samples: list[str],
    holdout: list[ValidationSampleSpec],
    dataset_type: LoraDatasetType,
    reference_downscale_factor: int,
) -> list[ValidationSampleDict]:
    """Build the trainer's `validation.samples` entries (new flexible format).

    See `_resolve_validation_samples` for the per-strategy rules. Returned
    dicts are YAML-serializable and emitted verbatim by `build_training_yaml`'s
    `validation_samples` path.
    """
    return [
        {"prompt": s.prompt, "conditions": s.conditions}
        for s in _resolve_validation_samples(
            prompt_samples=prompt_samples,
            holdout=holdout,
            dataset_type=dataset_type,
            reference_downscale_factor=reference_downscale_factor,
        )
    ]


def build_validation_sample_refs(
    *,
    prompt_samples: list[str],
    holdout: list[ValidationSampleSpec],
    dataset_type: LoraDatasetType,
    reference_downscale_factor: int,
) -> list[ValidationSampleRef]:
    """Build the per-run feed refs that map downloaded artifacts to their source.

    Order matches `build_validation_sample_dicts` exactly (same inputs feed the
    same `_resolve_validation_samples`), so artifact ``sample_index`` (1-based)
    maps to ``refs[index - 1]``. `reference_local_path` is left None here — the
    runner fills it in 2b once the held-out reference video is downloaded.
    """
    return [
        ValidationSampleRef(
            prompt=s.prompt,
            source=s.source,
            reference_local_path=None,
        )
        for s in _resolve_validation_samples(
            prompt_samples=prompt_samples,
            holdout=holdout,
            dataset_type=dataset_type,
            reference_downscale_factor=reference_downscale_factor,
        )
    ]


# Hardware tiers for matching the training preset to the GPU (see the LTX-2
# train-model skill): below 32 GB the trainer can't fit even the quantized
# low-VRAM path; the full-precision `standard` preset needs an 80 GB-class card,
# so anything in between runs the quantized `low_vram` preset.
MIN_TRAINING_VRAM_GB = 32
STANDARD_PRESET_MIN_VRAM_GB = 80


def recommended_preset_for_vram(vram_gb: int) -> TrainingPreset | None:
    """Preset that fits a GPU with `vram_gb` of VRAM, or None if unknown (0).

    Returns `standard` for 80 GB+ cards and `low_vram` for 32–79 GB. Callers
    must reject `vram_gb < MIN_TRAINING_VRAM_GB` *before* training (it has no
    viable preset); 0 means the VRAM wasn't recorded, so skip auto-matching.
    """
    if vram_gb <= 0:
        return None
    if vram_gb >= STANDARD_PRESET_MIN_VRAM_GB:
        return "standard"
    return "low_vram"


def _training_strategy_block(
    dataset_type: LoraDatasetType,
    *,
    with_audio: bool,
    first_frame_conditioning_p: float | None,
) -> str:
    """Render the `training_strategy` block using the official `flexible` strategy.

    `flexible` is the LTX-2 trainer's unified conditioning framework: a `video`
    modality block (always generated here) plus an optional `audio` modality
    block (generated only for joint audio-video standard LoRAs), with per-
    modality `conditions` for reference / first-frame conditioning. Mirrors the
    official configs exactly:
      - t2v (`configs/t2v_lora_low_vram.yaml`): `video.is_generated: true` and,
        when `with_audio`, `audio.is_generated: true`; NO conditions (the
        legacy `first_frame_conditioning_p` is dropped for t2v to match the
        official config).
      - IC-LoRA (`configs/v2v_ic_lora.yaml`): `video.is_generated: true` with a
        `reference` condition (`latents_dir: "reference_latents"`,
        `probability: 1.0`) plus a `first_frame` condition whose probability is
        `first_frame_conditioning_p` (default 0.2, the official value). No audio
        block — IC-LoRA is video-only.

    See:
    https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-trainer/docs/training-modes.md
    """
    lines: list[str] = [
        "training_strategy:\n",
        '  name: "flexible"\n',
        "  video:\n",
        "    is_generated: true\n",
        '    latents_dir: "latents"\n',
    ]
    if dataset_type == "ic_lora":
        ffc = 0.2 if first_frame_conditioning_p is None else first_frame_conditioning_p
        lines += [
            "    conditions:\n",
            "      - type: reference\n",
            '        latents_dir: "reference_latents"\n',
            "        probability: 1.0\n",
            "      - type: first_frame\n",
            f"        probability: {ffc}\n",
        ]
    elif with_audio:
        lines += [
            "  audio:\n",
            "    is_generated: true\n",
            '    latents_dir: "audio_latents"\n',
        ]
    return "".join(lines)


def build_training_yaml(
    *,
    config: TrainingConfig,
    dataset_type: LoraDatasetType = "standard",
    model_path: str,
    text_encoder_path: str,
    preprocessed_data_root: str,
    output_dir: str,
    validation_samples: list[ValidationSampleDict] | None = None,
) -> str:
    """Render the trainer YAML for a single LoRA training run.

    `dataset_type` selects the `training_strategy` shape under the official
    `flexible` strategy: a standard LoRA emits a `video` (+ optional `audio`)
    modality block, while an IC-LoRA adds a `reference` + `first_frame`
    condition on the video modality. `with_audio` only applies to standard
    LoRAs. All paths are remote (on the GPU host); the trainer requires
    `model_path` / `text_encoder_path` to be local files there.

    `validation_samples`, when provided, switches the validation block to the
    trainer's new `validation.samples` format (prompt + reference conditions),
    enabling the in-app training-results feed for both t2v and IC-LoRA. When
    ``None`` (export path, legacy callers) the deprecated `validation.prompts`
    format is emitted instead, preserving prior behavior. An empty list
    disables validation (interval null) so the prompt-embedding cache load —
    which OOMs memory-tight boxes — is skipped.
    """
    # Acceleration: an unset (None) field falls back to the preset, so the
    # two shipped presets — and old runs that stored no explicit value —
    # keep their exact behaviour. A profile that sets a value overrides it.
    preset_quant, preset_opt, preset_te8, preset_offload = _preset_acceleration(
        config.preset
    )
    quantization = preset_quant if config.quantization is None else config.quantization
    optimizer_type = (
        preset_opt if config.optimizer_type is None else config.optimizer_type
    )
    load_te_8bit = (
        preset_te8
        if config.load_text_encoder_in_8bit is None
        else config.load_text_encoder_in_8bit
    )
    offload_optimizer = (
        preset_offload
        if config.offload_optimizer_during_validation is None
        else config.offload_optimizer_during_validation
    )

    # IC-LoRA (video_to_video) never trains audio.
    is_ic_lora = dataset_type == "ic_lora"
    with_audio = config.with_audio and not is_ic_lora
    strategy_block = _training_strategy_block(
        dataset_type,
        with_audio=with_audio,
        first_frame_conditioning_p=config.first_frame_conditioning_p,
    )
    # IC-LoRA is video-only: when the user hasn't customized `target_modules`
    # (it still equals the default short-pattern set), substitute the explicit
    # video-module set from the official v2v config so LoRA doesn't also target
    # the unused audio / audio-video cross-attention modules. A custom list is
    # honored as-is.
    effective_target_modules = (
        list(_IC_LORA_TARGET_MODULES)
        if is_ic_lora and tuple(config.target_modules) == _DEFAULT_TARGET_MODULES
        else config.target_modules
    )
    # IC-LoRA (video_to_video) validation requires paired reference videos we
    # don't stage yet. The trainer *rejects* a video_to_video config whose
    # validation interval is truthy without reference conditions, so we disable
    # validation by setting the interval to null (None) — pushing it past the run
    # isn't enough, since any non-zero interval trips that validator.
    # Otherwise honor the configured interval / skip flag.
    use_samples = validation_samples is not None
    if use_samples:
        # New `validation.samples` path: enable validation only when there are
        # samples to generate. IC-LoRA samples carry reference conditions, so
        # the interval can be honored (no validator trip); an empty list keeps
        # validation off (interval null) so the prompt-embedding cache load is
        # skipped on memory-tight boxes.
        has_samples = len(validation_samples) > 0
        validation_interval = config.validation_interval if has_samples else None
        skip_initial_validation = (
            bool(config.skip_initial_validation) if has_samples else True
        )
        validation_block = _yaml_samples_block(validation_samples)
    else:
        validation_interval = None if is_ic_lora else config.validation_interval
        skip_initial_validation = (
            True if is_ic_lora else bool(config.skip_initial_validation)
        )
        # When validation is disabled (interval null — IC-LoRA, or a user who
        # turned it off), emit NO validation prompts. The trainer caches prompt
        # embeddings at startup *regardless of the interval* — loading the text
        # encoder AND a copy of the base model (validation_runner
        # `_cache_prompt_embeddings`) — and only an empty prompt list skips it
        # (its `if not prompts: return []` guard, plus the config's deprecated
        # prompts->samples converter, which no-ops on an empty list). On a
        # memory-tight box (local WSL) that load is what OOMs the run, so a
        # disabled-validation run should skip it rather than pay for a load it
        # never uses. `prompts: []` (explicit empty list) — not a bare `prompts:`,
        # which YAML reads as null and the trainer's `list[str]` field rejects.
        emit_validation_prompts = (
            [] if validation_interval is None else config.validation_prompts
        )
        validation_block = (
            "  prompts:\n" + _yaml_str_block(emit_validation_prompts, "    ")
            if emit_validation_prompts
            else "  prompts: []\n"
        )
    hub_model_id_line = (
        "  hub_model_id: null\n"
        if config.hub_model_id is None
        else f"  hub_model_id: {_yaml_str(config.hub_model_id)}\n"
    )
    load_checkpoint_line = (
        "  load_checkpoint: null\n"
        if config.load_checkpoint is None
        else f"  load_checkpoint: {_yaml_str(config.load_checkpoint)}\n"
    )
    video_dims = _yaml_int_inline(
        [
            config.validation_video_width,
            config.validation_video_height,
            config.validation_video_frames,
        ]
    )

    # Validation audio generation: a joint audio-video LoRA *can* validate with
    # audio (`generate_audio: true`) on a big card, but the validation forward
    # runs the full denoising loop WITHOUT gradient checkpointing's peak-memory
    # benefit, and the audio tokens add to the sequence — on a 32 GB card
    # carrying ~22 GB of int8 weights that forward OOMs around block 43 (the
    # `qbytes_mm` of the quantized base). The official `configs/a2v_lora.yaml`
    # sets `generate_audio: false` for the same reason, even on audio configs.
    # So on `low_vram` we validate video-only (the LoRA still trains audio —
    # this only affects what validation *outputs*, not what's learned); on
    # `standard` (80 GB+) we keep audio validation on.
    validation_generate_audio = with_audio and config.preset != "low_vram"

    return (
        "# Generated by the LTX Desktop LoRA trainer. Do not edit by hand.\n"
        "model:\n"
        f"  model_path: {_yaml_str(model_path)}\n"
        f"  text_encoder_path: {_yaml_str(text_encoder_path)}\n"
        '  training_mode: "lora"\n'
        f"{load_checkpoint_line}"
        "lora:\n"
        f"  rank: {config.rank}\n"
        f"  alpha: {config.alpha}\n"
        f"  dropout: {config.dropout}\n"
        "  target_modules:\n"
        f"{_yaml_str_block(effective_target_modules, '    ')}"
        f"{strategy_block}"
        "optimization:\n"
        f"  learning_rate: {config.learning_rate}\n"
        f"  steps: {config.steps}\n"
        f"  batch_size: {config.batch_size}\n"
        f"  gradient_accumulation_steps: {config.gradient_accumulation_steps}\n"
        f"  max_grad_norm: {config.max_grad_norm}\n"
        f"  optimizer_type: {_yaml_str(optimizer_type)}\n"
        f"  scheduler_type: {_yaml_str(config.scheduler_type)}\n"
        "  scheduler_params: {}\n"
        f"  enable_gradient_checkpointing: {_yb(config.enable_gradient_checkpointing)}\n"
        "acceleration:\n"
        f"  mixed_precision_mode: {_yaml_str(config.mixed_precision_mode)}\n"
        f"  quantization: {quantization}\n"
        f"  load_text_encoder_in_8bit: {_yb(load_te_8bit)}\n"
        f"  offload_optimizer_during_validation: {_yb(offload_optimizer)}\n"
        "data:\n"
        f"  preprocessed_data_root: {_yaml_str(preprocessed_data_root)}\n"
        f"  num_dataloader_workers: {config.num_dataloader_workers}\n"
        "validation:\n"
        f"{validation_block}"
        f"  negative_prompt: {_yaml_str(config.validation_negative_prompt)}\n"
        "  images: null\n"
        f"  video_dims: {video_dims}\n"
        f"  frame_rate: {config.validation_frame_rate}\n"
        f"  seed: {config.validation_seed}\n"
        f"  inference_steps: {config.validation_inference_steps}\n"
        f"  interval: {'null' if validation_interval is None else validation_interval}\n"
        f"  guidance_scale: {config.validation_guidance_scale}\n"
        f"  stg_scale: {config.stg_scale}\n"
        f"  stg_blocks: {_yaml_int_inline(config.stg_blocks)}\n"
        f"  stg_mode: {_yaml_str(config.stg_mode)}\n"
        f"  generate_audio: {_yb(validation_generate_audio)}\n"
        f"  skip_initial_validation: {_yb(skip_initial_validation)}\n"
        "checkpoints:\n"
        f"  interval: {config.checkpoint_interval}\n"
        f"  keep_last_n: {config.checkpoint_keep_last_n}\n"
        f"  precision: {_yaml_str(config.checkpoint_precision)}\n"
        "flow_matching:\n"
        f"  timestep_sampling_mode: {_yaml_str(config.timestep_sampling_mode)}\n"
        "  timestep_sampling_params: {}\n"
        "hub:\n"
        f"  push_to_hub: {_yb(config.push_to_hub)}\n"
        f"{hub_model_id_line}"
        "wandb:\n"
        f"  enabled: {_yb(config.wandb_enabled)}\n"
        f"seed: {config.seed}\n"
        f"output_dir: {_yaml_str(output_dir)}\n"
    )

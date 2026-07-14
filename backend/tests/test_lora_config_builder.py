"""Unit tests for the trainer YAML builder's collection-type branching.

The builder is pure (config + paths -> YAML string), so these assert on the
emitted `training_strategy` block (the official `flexible` strategy) and the
audio/validation knobs that flip between a standard LoRA (video + optional
audio modality) and an IC-LoRA (video with a reference + first_frame condition).
"""

from __future__ import annotations

from handlers.lora_config_builder import (
    MIN_TRAINING_VRAM_GB,
    STANDARD_PRESET_MIN_VRAM_GB,
    ValidationSampleDict,
    ValidationSampleSpec,
    build_validation_sample_dicts,
    build_training_yaml,
    preset_loads_text_encoder_in_8bit,
    preset_reference_downscale_factor,
    recommended_preset_for_vram,
)
from state.lora_training_state import TrainingConfig


class TestPresetForVram:
    def test_unknown_vram_returns_none(self) -> None:
        # 0 = VRAM not recorded -> caller leaves the user's preset untouched.
        assert recommended_preset_for_vram(0) is None

    def test_eighty_gb_and_up_uses_standard(self) -> None:
        assert recommended_preset_for_vram(STANDARD_PRESET_MIN_VRAM_GB) == "standard"
        assert recommended_preset_for_vram(96) == "standard"

    def test_midrange_uses_low_vram(self) -> None:
        assert recommended_preset_for_vram(MIN_TRAINING_VRAM_GB) == "low_vram"
        assert recommended_preset_for_vram(48) == "low_vram"
        assert recommended_preset_for_vram(STANDARD_PRESET_MIN_VRAM_GB - 1) == "low_vram"


class TestPresetLoadsTextEncoderIn8bit:
    def test_low_vram_loads_8bit(self) -> None:
        # Mirrors t2v_lora_low_vram.yaml's load_text_encoder_in_8bit: true; the
        # preprocess stage reuses this so both stages agree on Gemma3 precision.
        assert preset_loads_text_encoder_in_8bit("low_vram") is True

    def test_standard_keeps_bf16(self) -> None:
        assert preset_loads_text_encoder_in_8bit("standard") is False


class TestPresetReferenceDownscaleFactor:
    def test_low_vram_halves_references(self) -> None:
        # IC-LoRA concatenates reference + target tokens (~4x self-attention
        # memory); low_vram halves reference resolution via the official
        # --reference-downscale-factor lever so the backward fits a 32 GB card.
        assert preset_reference_downscale_factor("low_vram") == 2

    def test_standard_keeps_full_size_references(self) -> None:
        # Matches the official v2v_ic_lora.yaml (downscale_factor: 1) on 80 GB+
        # cards where the doubled sequence fits without downscaling.
        assert preset_reference_downscale_factor("standard") == 1


def _yaml(dataset_type, **config_kwargs):
    return build_training_yaml(
        config=TrainingConfig(**config_kwargs),
        dataset_type=dataset_type,
        model_path="/remote/model.safetensors",
        text_encoder_path="/remote/te",
        preprocessed_data_root="/remote/.precomputed",
        output_dir="/remote/out",
    )


class TestStandardLora:
    def test_emits_flexible_t2v(self):
        yaml = _yaml("standard")
        assert 'name: "flexible"' in yaml
        assert "text_to_video" not in yaml
        assert "video_to_video" not in yaml
        assert "reference_latents" not in yaml
        # t2v has no conditions block (matches official t2v config).
        assert "conditions:" not in yaml

    def test_with_audio_flips_audio_block(self):
        yaml = _yaml("standard", with_audio=True)
        assert "  audio:\n" in yaml
        assert "    is_generated: true\n" in yaml
        assert '    latents_dir: "audio_latents"\n' in yaml
        # generate_audio is a validation knob, still driven by with_audio.
        assert "generate_audio: true" in yaml

    def test_without_audio_omits_audio_block(self):
        yaml = _yaml("standard", with_audio=False)
        assert "  audio:" not in yaml
        assert "audio_latents" not in yaml
        assert "generate_audio: false" in yaml

    def test_low_vram_audio_trains_audio_but_validates_video_only(self):
        # On a 32 GB card the validation forward (full denoising loop, no
        # gradient-checkpointing peak benefit) OOMs when it also generates
        # audio — the audio tokens push the sequence past what ~22 GB of int8
        # weights leaves free. The official a2v config sets generate_audio:
        # false for the same reason. So low_vram + with_audio still emits the
        # audio *training* block (the LoRA learns audio) but turns audio
        # generation OFF in validation (video-only validation outputs).
        yaml = _yaml("standard", preset="low_vram", with_audio=True)
        assert "  audio:\n" in yaml
        assert "    is_generated: true\n" in yaml
        assert '    latents_dir: "audio_latents"\n' in yaml
        assert "generate_audio: false" in yaml

    def test_standard_preset_audio_validates_with_audio(self):
        # 80 GB+ cards can afford audio validation: standard + with_audio keeps
        # generate_audio on.
        yaml = _yaml("standard", preset="standard", with_audio=True)
        assert "generate_audio: true" in yaml


class TestIcLora:
    def test_emits_flexible_ic_lora_with_reference_condition(self):
        yaml = _yaml("ic_lora")
        assert 'name: "flexible"' in yaml
        assert "text_to_video" not in yaml
        assert "video_to_video" not in yaml
        # Reference conditioning on the video modality.
        assert "    conditions:\n" in yaml
        assert "      - type: reference\n" in yaml
        assert '        latents_dir: "reference_latents"\n' in yaml
        assert "        probability: 1.0\n" in yaml
        # Official first_frame condition, default probability 0.2.
        assert "      - type: first_frame\n" in yaml
        assert "        probability: 0.2\n" in yaml

    def test_first_frame_probability_honors_config(self):
        yaml = _yaml("ic_lora", first_frame_conditioning_p=0.25)
        assert "      - type: first_frame\n" in yaml
        assert "        probability: 0.25\n" in yaml

    def test_uses_explicit_video_target_modules_by_default(self):
        yaml = _yaml("ic_lora")
        # IC-LoRA is video-only: target the explicit video modules from the
        # official v2v config, not the short patterns that also hit audio.
        assert '    - "attn1.to_k"' in yaml
        assert '    - "attn2.to_v"' in yaml
        assert '    - "ff.net.0.proj"' in yaml
        assert '    - "ff.net.2"' in yaml
        # The short patterns (which would match audio modules too) are gone.
        assert '    - "to_k"\n' not in yaml

    def test_custom_target_modules_honored_for_ic_lora(self):
        yaml = _yaml("ic_lora", target_modules=["to_q", "to_k"])
        assert '    - "to_q"' in yaml
        assert '    - "to_k"' in yaml
        assert "attn1.to_k" not in yaml

    def test_never_trains_audio_even_when_requested(self):
        # IC-LoRA conditions on reference latents; audio is meaningless here.
        yaml = _yaml("ic_lora", with_audio=True)
        assert "  audio:" not in yaml
        assert "audio_latents" not in yaml
        assert "generate_audio: false" in yaml

    def test_skips_validation(self):
        # Validation needs paired reference videos we don't stage yet. The
        # trainer rejects a flexible v2v config with a truthy validation
        # interval but no reference conditions on the samples, so validation is
        # disabled via a null interval (a non-zero interval would trip that
        # validator).
        yaml = _yaml("ic_lora", steps=1000)
        assert "skip_initial_validation: true" in yaml
        assert "  interval: null" in yaml  # validation disabled
        # The checkpoints block keeps a real interval (this isn't validation).
        assert "  interval: 250" in yaml

    def test_default_type_is_standard(self):
        default_yaml = build_training_yaml(
            config=TrainingConfig(),
            model_path="/m",
            text_encoder_path="/te",
            preprocessed_data_root="/p",
            output_dir="/o",
        )
        assert 'name: "flexible"' in default_yaml


class TestPresetAcceleration:
    def test_standard_preset_defaults(self):
        yaml = _yaml("standard")
        assert "quantization: null" in yaml
        assert 'optimizer_type: "adamw"' in yaml
        assert "load_text_encoder_in_8bit: false" in yaml
        assert "offload_optimizer_during_validation: false" in yaml

    def test_low_vram_preset_falls_back_to_int8(self):
        yaml = _yaml("standard", preset="low_vram")
        assert "quantization: int8-quanto" in yaml
        assert 'optimizer_type: "adamw8bit"' in yaml
        assert "load_text_encoder_in_8bit: true" in yaml
        assert "offload_optimizer_during_validation: true" in yaml

    def test_explicit_acceleration_overrides_preset(self):
        # An explicit field wins over the preset-derived fallback.
        yaml = _yaml(
            "standard",
            preset="low_vram",
            quantization="null",
            optimizer_type="adamw",
            load_text_encoder_in_8bit=False,
            offload_optimizer_during_validation=False,
        )
        assert "quantization: null" in yaml
        assert 'optimizer_type: "adamw"' in yaml
        assert "load_text_encoder_in_8bit: false" in yaml


class TestFullConfigSurface:
    def test_lora_and_optimization_fields_render(self):
        yaml = _yaml(
            "standard",
            rank=64,
            alpha=128,
            dropout=0.1,
            target_modules=["to_q", "to_k"],
            batch_size=2,
            gradient_accumulation_steps=4,
            max_grad_norm=2.5,
            scheduler_type="cosine",
            enable_gradient_checkpointing=False,
            num_dataloader_workers=6,
        )
        assert "rank: 64" in yaml
        assert "alpha: 128" in yaml
        assert "dropout: 0.1" in yaml
        assert '    - "to_q"' in yaml
        assert '    - "to_k"' in yaml
        assert '    - "to_v"' not in yaml
        assert "batch_size: 2" in yaml
        assert "gradient_accumulation_steps: 4" in yaml
        assert "max_grad_norm: 2.5" in yaml
        assert 'scheduler_type: "cosine"' in yaml
        assert "enable_gradient_checkpointing: false" in yaml
        assert "num_dataloader_workers: 6" in yaml

    def test_validation_fields_render(self):
        yaml = _yaml(
            "standard",
            validation_prompts=["a cat", "a dog"],
            validation_negative_prompt="bad",
            validation_video_width=768,
            validation_video_height=512,
            validation_video_frames=89,
            validation_frame_rate=24.0,
            validation_inference_steps=40,
            validation_interval=100,
            validation_guidance_scale=5.5,
            validation_seed=7,
            stg_scale=2.0,
            stg_blocks=[10, 20],
            stg_mode="stg_a",
        )
        assert '    - "a cat"' in yaml
        assert '    - "a dog"' in yaml
        assert 'negative_prompt: "bad"' in yaml
        assert "video_dims: [768, 512, 89]" in yaml
        assert "frame_rate: 24.0" in yaml
        assert "inference_steps: 40" in yaml
        assert "interval: 100" in yaml
        assert "guidance_scale: 5.5" in yaml
        assert "seed: 7" in yaml
        assert "stg_scale: 2.0" in yaml
        assert "stg_blocks: [10, 20]" in yaml
        assert 'stg_mode: "stg_a"' in yaml

    def test_checkpoints_hub_misc_render(self):
        yaml = _yaml(
            "standard",
            checkpoint_interval=500,
            checkpoint_keep_last_n=5,
            checkpoint_precision="float32",
            timestep_sampling_mode="uniform",
            push_to_hub=True,
            hub_model_id="me/model",
            wandb_enabled=True,
            seed=99,
            load_checkpoint="/remote/ckpt",
        )
        assert "interval: 500" in yaml
        assert "keep_last_n: 5" in yaml
        assert 'precision: "float32"' in yaml
        assert 'timestep_sampling_mode: "uniform"' in yaml
        assert "push_to_hub: true" in yaml
        assert 'hub_model_id: "me/model"' in yaml
        assert "enabled: true" in yaml
        assert "seed: 99" in yaml
        assert 'load_checkpoint: "/remote/ckpt"' in yaml

    def test_t2v_omits_first_frame_condition(self):
        # Official t2v config has no first_frame condition; the legacy
        # `first_frame_conditioning_p` is dropped for t2v under `flexible`.
        yaml = _yaml("standard", first_frame_conditioning_p=0.25)
        assert "first_frame" not in yaml
        assert "first_frame_conditioning_p" not in yaml

    def test_skip_initial_validation_override_for_standard(self):
        yaml = _yaml("standard", skip_initial_validation=True)
        assert "skip_initial_validation: true" in yaml


class TestBackCompat:
    def test_default_config_matches_official_flexible_yaml(self):
        # A config carrying only the legacy 7-field subset (everything else
        # defaulted) emits the official `flexible` t2v document (no legacy
        # strategy names, no first_frame_conditioning_p for t2v).
        yaml = _yaml("standard")
        assert 'name: "flexible"' in yaml
        assert "text_to_video" not in yaml
        assert "video_to_video" not in yaml
        assert "first_frame_conditioning_p" not in yaml
        assert "rank: 32" in yaml
        assert "alpha: 32" in yaml
        assert "dropout: 0.0" in yaml
        assert '    - "to_k"' in yaml
        assert '    - "to_q"' in yaml
        assert '    - "to_v"' in yaml
        assert '    - "to_out.0"' in yaml
        assert "learning_rate: 0.0001" in yaml
        assert "steps: 2000" in yaml
        assert "batch_size: 1" in yaml
        assert "gradient_accumulation_steps: 1" in yaml
        assert "max_grad_norm: 1.0" in yaml
        assert 'scheduler_type: "linear"' in yaml
        assert "enable_gradient_checkpointing: true" in yaml
        assert 'mixed_precision_mode: "bf16"' in yaml
        assert "num_dataloader_workers: 2" in yaml
        assert "video_dims: [576, 576, 49]" in yaml
        assert "frame_rate: 25.0" in yaml
        assert "seed: 42" in yaml
        assert "inference_steps: 30" in yaml
        assert "interval: 250" in yaml
        assert "guidance_scale: 4.0" in yaml
        assert "stg_scale: 1.0" in yaml
        assert "stg_blocks: [29]" in yaml
        assert 'stg_mode: "stg_av"' in yaml
        assert "skip_initial_validation: false" in yaml
        assert "keep_last_n: 3" in yaml
        assert 'precision: "bfloat16"' in yaml
        assert 'timestep_sampling_mode: "shifted_logit_normal"' in yaml
        assert "push_to_hub: false" in yaml
        assert "hub_model_id: null" in yaml
        assert "load_checkpoint: null" in yaml


class TestValidationPromptsWhenDisabled:
    """When validation is disabled (interval null), emit NO validation prompts
    so the trainer skips the startup text-encoder + model load it would
    otherwise do to cache embeddings (validation_runner `if not prompts`)."""

    def test_ic_lora_emits_empty_prompts(self):
        # IC-LoRA forces the interval to null (can't validate without paired
        # references), so the prompt list must be emitted empty.
        yaml = _yaml("ic_lora")
        assert "interval: null" in yaml
        assert "prompts: []" in yaml
        assert "A high quality sample from the trained concept." not in yaml

    def test_standard_with_validation_keeps_prompts(self):
        # A standard LoRA validates (interval defaults to 250), so its prompts
        # are still emitted (the trainer needs them to sample).
        yaml = _yaml("standard")
        assert "prompts: []" not in yaml
        assert "A high quality sample from the trained concept." in yaml


def _yaml_with_samples(
    dataset_type: str,
    samples: list[ValidationSampleDict],
    **config_kwargs: object,
) -> str:
    return build_training_yaml(
        config=TrainingConfig(**config_kwargs),  # type: ignore[arg-type]
        dataset_type=dataset_type,  # type: ignore[arg-type]
        model_path="/remote/model.safetensors",
        text_encoder_path="/remote/te",
        preprocessed_data_root="/remote/.precomputed",
        output_dir="/remote/out",
        validation_samples=samples,
    )


class TestBuildValidationSampleDicts:
    def test_t2v_emits_prompt_only_samples_for_user_prompts(self) -> None:
        samples = build_validation_sample_dicts(
            prompt_samples=["a cat playing", "a dog running"],
            holdout=[],
            dataset_type="standard",
            reference_downscale_factor=1,
        )
        assert samples == [
            {"prompt": "a cat playing", "conditions": []},
            {"prompt": "a dog running", "conditions": []},
        ]

    def test_t2v_emits_prompt_only_samples_for_holdout_clips(self) -> None:
        samples = build_validation_sample_dicts(
            prompt_samples=[],
            holdout=[ValidationSampleSpec("a held-out clip caption")],
            dataset_type="standard",
            reference_downscale_factor=1,
        )
        assert samples == [{"prompt": "a held-out clip caption", "conditions": []}]

    def test_t2v_ignores_reference_path_on_holdout(self) -> None:
        # t2v has no reference conditioning; a stray reference path is ignored.
        samples = build_validation_sample_dicts(
            prompt_samples=[],
            holdout=[ValidationSampleSpec("cap", reference_video_path="/r/v.mp4")],
            dataset_type="standard",
            reference_downscale_factor=2,
        )
        assert samples == [{"prompt": "cap", "conditions": []}]

    def test_ic_lora_drops_user_prompts(self) -> None:
        # IC-LoRA validation needs a reference video; bare prompts can't supply
        # one, so they're dropped (not emitted as prompt-only samples).
        samples = build_validation_sample_dicts(
            prompt_samples=["a bare prompt"],
            holdout=[],
            dataset_type="ic_lora",
            reference_downscale_factor=2,
        )
        assert samples == []

    def test_ic_lora_emits_reference_conditioned_samples_for_holdout(self) -> None:
        samples = build_validation_sample_dicts(
            prompt_samples=[],
            holdout=[
                ValidationSampleSpec("target caption", reference_video_path="/r/ref.mp4")
            ],
            dataset_type="ic_lora",
            reference_downscale_factor=2,
        )
        assert len(samples) == 1
        sample = samples[0]
        assert sample["prompt"] == "target caption"
        conditions = sample["conditions"]
        assert len(conditions) == 1
        cond = conditions[0]
        assert cond["type"] == "reference"
        assert cond["video"] == "/r/ref.mp4"
        assert cond["downscale_factor"] == 2
        assert cond["temporal_scale_factor"] == 1
        assert cond["include_in_output"] is True

    def test_ic_lora_skips_holdout_without_reference(self) -> None:
        samples = build_validation_sample_dicts(
            prompt_samples=[],
            holdout=[ValidationSampleSpec("cap", reference_video_path=None)],
            dataset_type="ic_lora",
            reference_downscale_factor=2,
        )
        assert samples == []

    def test_ic_lora_downscale_factor_mirrors_preset(self) -> None:
        # standard preset -> factor 1 (full-size references, official v2v config).
        samples = build_validation_sample_dicts(
            prompt_samples=[],
            holdout=[ValidationSampleSpec("cap", reference_video_path="/r.mp4")],
            dataset_type="ic_lora",
            reference_downscale_factor=1,
        )
        assert samples[0]["conditions"][0]["downscale_factor"] == 1

    def test_empty_prompt_holdout_skipped(self) -> None:
        # A blank caption contributes nothing.
        samples = build_validation_sample_dicts(
            prompt_samples=[],
            holdout=[ValidationSampleSpec("   ")],
            dataset_type="standard",
            reference_downscale_factor=1,
        )
        assert samples == []


class TestValidationSamplesYaml:
    def test_t2v_samples_block_renders_prompt_only(self) -> None:
        samples = [
            {"prompt": "a cat playing", "conditions": []},
        ]
        yaml = _yaml_with_samples("standard", samples, validation_interval=50)
        assert "  samples:\n" in yaml
        assert "    - prompt: \"a cat playing\"\n" in yaml
        assert "      conditions: []\n" in yaml
        assert "  interval: 50\n" in yaml

    def test_ic_lora_samples_block_renders_reference_condition(self) -> None:
        samples = [
            {
                "prompt": "target caption",
                "conditions": [
                    {
                        "type": "reference",
                        "video": "/r/ref.mp4",
                        "downscale_factor": 2,
                        "temporal_scale_factor": 1,
                        "include_in_output": True,
                    }
                ],
            }
        ]
        yaml = _yaml_with_samples("ic_lora", samples, validation_interval=50)
        # IC-LoRA now validates (interval honored) because samples carry refs.
        assert "  interval: 50\n" in yaml
        assert '        - type: "reference"\n' in yaml
        assert '          video: "/r/ref.mp4"\n' in yaml
        assert "          downscale_factor: 2\n" in yaml
        assert "          include_in_output: true\n" in yaml

    def test_empty_samples_disables_validation(self) -> None:
        yaml = _yaml_with_samples("ic_lora", [], validation_interval=50)
        assert "  samples: []\n" in yaml
        assert "  interval: null\n" in yaml
        assert "skip_initial_validation: true\n" in yaml

    def test_empty_samples_t2v_disables_validation(self) -> None:
        yaml = _yaml_with_samples("standard", [], validation_interval=50)
        assert "  samples: []\n" in yaml
        assert "  interval: null\n" in yaml

    def test_legacy_prompts_path_untouched_when_samples_none(self) -> None:
        # validation_samples=None keeps the deprecated prompts format (export
        # path + existing callers), so prior tests/behavior are preserved.
        yaml = _yaml("standard")
        assert "  prompts:\n" in yaml
        assert "samples:" not in yaml

    def test_samples_path_omits_prompts_key(self) -> None:
        samples = [{"prompt": "p", "conditions": []}]
        yaml = _yaml_with_samples("standard", samples)
        # The trainer accepts either prompts OR samples; emitting both is invalid
        # (its validator rejects a config that sets samples alongside prompts).
        assert "prompts:" not in yaml
        assert "  samples:\n" in yaml


"""Render a fill-in-the-blanks Hugging Face **model card** for a LoRA dataset.

This is the publication document a user ships alongside trained weights (the
`README.md` of an HF model repo). Unlike `lora_publish.py` — which builds the
*final* card from a finished `TrainingJob` (weights + chosen examples) — this
runs at **dataset-export time**, before training, so it emits a *template*
tailored to the dataset: every value we already know (type, trigger word,
rank/alpha, recommended inference settings from the chosen profile) is pre-
filled, and everything we can't know yet (exact checkpoint filename, examples,
results) stays as a clearly-marked `{{ placeholder }}` for the user to complete.

The structure follows the LTX LoRA / IC-LoRA model-card template: IC-LoRA-only
sections (control signal, how-it-works) are emitted only for IC-LoRA datasets,
and dropped for a standard LoRA, so the card the user gets is already trimmed to
their case. Pure plumbing over a `LoraDataset` + `TrainingConfig`, so it stays
trivially unit-testable.
"""

from __future__ import annotations

from state.lora_training_state import LoraDataset, TrainingConfig

# Default Hugging Face foundation-model repo. The exact version matters for HF's
# adapter back-link, so it's left as a placeholder the user must confirm — we
# only suggest the org + a hint of the expected shape.
_BASE_MODEL_HINT = "Lightricks/{{ base_model_repo }}"
_LICENSE_LINK = "https://github.com/Lightricks/LTX-2/blob/main/LICENSE"


def _tags(dataset: LoraDataset) -> list[str]:
    """Discoverability tags, pre-seeded; the user adds the effect/control tag."""
    kind = "ic-lora" if dataset.type == "ic_lora" else "lora"
    modality = "video-to-video" if dataset.type == "ic_lora" else "text-to-video"
    tags = ["ltx-video", kind, "ltx-2", modality]
    if dataset.trigger_word and dataset.trigger_word.strip():
        tags.append(dataset.trigger_word.strip())
    return tags


def _front_matter(dataset: LoraDataset) -> str:
    is_ic = dataset.type == "ic_lora"
    pipeline_tag = "video-to-video" if is_ic else "text-to-video"
    lines = [
        "---",
        "base_model:",
        f"- {_BASE_MODEL_HINT}          # exact Hub repo ID, e.g. Lightricks/LTX-2.3",
        "base_model_relation: adapter",
        "license: other",
        "license_name: ltx-2-community-license",
        f"license_link: {_LICENSE_LINK}",
        "language:",
        "- en",
        "tags:",
        *[f"- {t}" for t in _tags(dataset)],
        "- \"{{ control_or_effect_tag }}\"   # e.g. depth, pose, outpaint  |  claymation, anime, vintage-film",
        f"pipeline_tag: {pipeline_tag}",
        "---",
    ]
    return "\n".join(lines)


def _kept_clip_count(dataset: LoraDataset) -> int:
    return sum(
        1
        for c in dataset.clips
        if not c.deleted_at and c.triage not in ("reject", "holdout")
    )


def build_template_card(dataset: LoraDataset, config: TrainingConfig | None = None) -> str:
    """Render the pre-filled model-card template for `dataset`.

    `config` (the chosen training profile, or trainer defaults when None) seeds
    the recommended-settings + training sections. The result is Markdown ready
    to drop into a Hugging Face repo as `README.md` once the `{{ ... }}` blanks
    are filled and the guidance comments stripped.
    """
    cfg = config or TrainingConfig()
    is_ic = dataset.type == "ic_lora"
    kind = "IC-LoRA" if is_ic else "LoRA"
    modality = "video-to-video" if is_ic else "text-to-video"
    trigger = (dataset.trigger_word or "").strip()
    name = dataset.name or "Untitled"
    clip_count = _kept_clip_count(dataset)
    unit = "pair" if is_ic else "clip"
    res = f"{cfg.validation_video_width}×{cfg.validation_video_height}"

    parts: list[str] = [_front_matter(dataset)]

    parts.append(
        f"# LTX-2 {kind} {name}\n"
        "<!-- Replace every {{ placeholder }} below, delete sections that don't apply, "
        "and strip these guidance comments before publishing. -->"
    )
    parts.append(
        f"This is **{name}**, {'an' if is_ic else 'a'} **{kind}** ({modality}) trained on top of "
        "**LTX-2**, {{ one-sentence description of what it does and the problem it solves }}.\n\n"
        "It is based on the [LTX-2](https://huggingface.co/Lightricks) foundation model."
    )

    parts.append(
        "## Model Files\n\n"
        "`{{ exact-checkpoint-filename.safetensors }}`\n"
        "<!-- List every shipped file; note the recommended default if there are several. -->"
    )

    details = [
        "## Model Details\n",
        "- **Base Model:** LTX-2",
        f"- **Training Type:** {kind}",
    ]
    if is_ic:
        details.append(
            "- **Control Type:** {{ what the model conditions on, e.g. Depth / Video & Audio }}"
        )
        details.append(
            "- **Reference Downscale Factor:** {{ 1 / 2 }} "
            "({{ reference resolution relative to output }})"
        )
    details.append(f"- **LoRA rank / alpha:** {cfg.rank} / {cfg.alpha}")
    parts.append("\n".join(details))

    parts.append(
        "## Intended Use & Out-of-Scope\n\n"
        "**Intended use:** {{ the workflows / content this LoRA is designed for }}.\n\n"
        "**Out of scope:** {{ uses where it underperforms or should not be used }}."
    )

    if is_ic:
        parts.append(
            "## Control Signal Requirements\n\n"
            "- **Control signal type:** {{ depth / pose / canny / HDR / lipdub audio+video / other }}\n"
            "- **Expected input:** {{ image sequence / video / audio+video / mask / tracks }}\n"
            "- **Preprocessing:** {{ required extractor, normalization, color transform, etc. }}\n"
            f"- **Alignment:** reference and output share fps/resolution/frame-count "
            f"({res}, {cfg.validation_video_frames} frames @ {cfg.validation_frame_rate:g} fps as trained).\n"
            "- **Mask support:** {{ whether masks are supported; delete if not applicable }}"
        )
        parts.append(
            "## How It Works\n\n"
            "{{ Explain the input convention and what the model expects at inference. }}"
        )

    comfy_steps = [
        "## Usage\n",
        "### ComfyUI\n",
        "1. Copy the LoRA weights into `models/loras`.",
        "2. Load the **LTX-2** base model and add `{{ exact-checkpoint-filename.safetensors }}` as the LoRA.",
        "3. Start at strength `{{ recommended_strength }}` and adjust to taste.",
    ]
    if is_ic:
        comfy_steps.append(
            "4. Use the matching IC-LoRA workflow from the "
            "[LTX-2 ComfyUI repository](https://github.com/Lightricks/ComfyUI-LTXVideo/) "
            "and wire the reference/control input."
        )
    parts.append("\n".join(comfy_steps))

    settings = [
        "## Recommended Settings\n",
        "- **LoRA strength / weight:** {{ e.g. 0.8–1.0 }}",
        f"- **Inference steps:** {cfg.validation_inference_steps}",
        f"- **Guidance scale:** {cfg.validation_guidance_scale:g}",
        f"- **Resolution & frames:** {res}, {cfg.validation_video_frames} frames "
        f"@ {cfg.validation_frame_rate:g} fps",
    ]
    if trigger:
        settings.append(f"- **Prompting:** include the trigger word `{trigger}` in your prompt.")
    else:
        settings.append("- **Prompting:** {{ trigger word(s) if any, and recommended prompt structure }}.")
    parts.append("\n".join(settings))

    parts.append(
        "## Examples\n\n"
        "<!-- Upload media to the repo and reference it relatively. -->\n"
        "{{ ![short caption](example_media_filename) }}"
    )

    parts.append(
        "## Dataset\n\n"
        f"Trained on **{clip_count}** {unit}{'s' if clip_count != 1 else ''}. "
        "{{ The model was trained using a proprietary dataset. / "
        "Link the public dataset repo if disclosable. }}"
    )

    optimizer = cfg.optimizer_type or f"{cfg.preset} default"
    parts.append(
        "## Training\n\n"
        f"- **Technique:** {kind} (rank {cfg.rank}, alpha {cfg.alpha}) on the DiT transformer\n"
        f"- **Hyperparameters:** {cfg.mixed_precision_mode} precision, optimizer `{optimizer}`, "
        f"learning rate {cfg.learning_rate:g}, scheduler `{cfg.scheduler_type}`\n"
        f"- **Steps:** {cfg.steps}\n"
        "- **Infrastructure:** LTX-2 Trainer"
    )

    parts.append("## License\n\nSee the **LTX-2-community-license** for full terms.")

    parts.append(
        "## Acknowledgments\n\n"
        "- Base model by **Lightricks**\n"
        "- Training infrastructure: **LTX-2 Trainer**\n"
        "- Dataset prepared with **LTX Desktop**"
    )

    return "\n\n".join(parts) + "\n"

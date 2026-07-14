"""Per-LoRA prompt-writing assistant templates.

Each LoRA the user can apply from Gen Space carries an optional *prompt
template* — the system prompt Gemini Flash uses to watch the reference video and
write a tailored text-to-video prompt that activates the adapter (trigger word,
prompt structure, identity-preservation clause). The app auto-generates a
sensible default per entry from the LoRA's name / variant; the user can edit it
in a per-LoRA modal, and the override is persisted here.

The store is a single durable JSON ledger keyed by registry entry id, so it
works uniformly across all entry kinds (official union, user-trained, imported)
without touching the training ledger or the imported-LoRA library. The registry
overlays the stored values (or the synthesized defaults) onto every entry it
builds.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import shutil
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from api_types import (
    ControlConditioningType,
    LoraInferenceVariantApi,
)
from handlers.base import StateHandlerBase

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig
    from state.app_state_types import AppState

logger = logging.getLogger(__name__)


class LoraPromptTemplateOverride(BaseModel):
    """A user-edited template/trigger override for one registry entry."""

    model_config = ConfigDict(strict=True)

    prompt_template: str | None = None
    trigger_word: str | None = None


class LoraPromptTemplateState(BaseModel):
    model_config = ConfigDict(strict=True)

    items: dict[str, LoraPromptTemplateOverride] = Field(default_factory=dict)


class LoraPromptTemplateStore(StateHandlerBase):
    """Durable, entry-id-keyed overrides for per-LoRA prompt templates."""

    def __init__(self, state: AppState, lock: RLock, config: RuntimeConfig) -> None:
        super().__init__(state, lock, config)
        self._ledger_file: Path = config.app_data_dir / "lora_prompt_templates.json"
        self._items: dict[str, LoraPromptTemplateOverride] = {}
        self._loaded: bool = False

    def load_state(self) -> None:
        """Load the ledger once at startup."""
        with self.lock:
            if self._loaded:
                return
            self._items = self._load_file_unlocked().items
            self._loaded = True

    def get_override(self, entry_id: str) -> LoraPromptTemplateOverride | None:
        with self.lock:
            return self._items.get(entry_id)

    def set_override(
        self,
        entry_id: str,
        *,
        prompt_template: str | None,
        trigger_word: str | None,
    ) -> LoraPromptTemplateOverride:
        # Normalize: an empty string is treated as "reset to default" (None).
        cleaned_template = prompt_template.strip() if prompt_template else None
        cleaned_trigger = trigger_word.strip() if trigger_word else None
        override = LoraPromptTemplateOverride(
            prompt_template=cleaned_template,
            trigger_word=cleaned_trigger,
        )
        with self.lock:
            # Drop the entry entirely when both fields are cleared so the
            # registry falls back to the auto-generated default.
            if override.prompt_template is None and override.trigger_word is None:
                self._items.pop(entry_id, None)
            else:
                self._items[entry_id] = override
            self._persist_unlocked()
            return override

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_file_unlocked(self) -> LoraPromptTemplateState:
        if not self._ledger_file.exists():
            return LoraPromptTemplateState()
        try:
            with open(self._ledger_file, "r", encoding="utf-8") as f:
                return LoraPromptTemplateState.model_validate(json.load(f))
        except Exception as exc:
            backup = self._ledger_file.with_name(
                f"{self._ledger_file.stem}.corrupt-{int(time.time())}.json"
            )
            try:
                shutil.copy2(self._ledger_file, backup)
            except Exception:
                pass
            logger.warning(
                "%s could not be parsed; starting empty: %s",
                self._ledger_file.name,
                exc,
            )
            return LoraPromptTemplateState()

    def _persist_unlocked(self) -> None:
        path = self._ledger_file
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(LoraPromptTemplateState(items=self._items).model_dump(mode="json"), f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)


# ---------------------------------------------------------------------
# Default template synthesis
# ---------------------------------------------------------------------
#
# The defaults are generalized from the conehead IC-LoRA system prompt: a
# structured, identity-preserving, trigger-first prompt the adapter was trained
# on. They're a starting point; the user is expected to refine them per LoRA in
# the edit modal (e.g. paste the author's exact system prompt when they have it).

def _effect_phrase(description: str | None) -> str:
    """Use user-authored behavior metadata; display names are not semantics."""
    clean = re.sub(r"\s+", " ", description or "").strip()
    return clean or "the transformation learned during training"

def _purpose_from_description(description: str | None) -> str:
    """Select specialized guidance only from an explicit behavior description."""
    normalized = re.sub(r"[^a-z0-9]+", " ", (description or "").lower()).strip()
    if any(
        phrase in normalized
        for phrase in (
            "cleanplate",
            "clean plate",
            "remove subject",
            "subject removal",
            "remove foreground",
            "removes foreground",
            "removing foreground",
            "empty scene",
            "reconstruct background",
            "reconstructs the hidden background",
        )
    ):
        return "clean_plate"
    return "generic"


def build_default_prompt_template(
    *,
    description: str | None,
    variant: LoraInferenceVariantApi,
    trigger_word: str | None,
    conditioning_types: tuple[ControlConditioningType, ...] = (),
) -> str | None:
    """Synthesize a per-LoRA system prompt for the auto-prompt assistant.

    Returns None for `standard` style LoRAs, which have no reference video to
    prompt from (the assistant is only offered in the UI when a template and a
    reference video exist).
    """
    trigger = trigger_word.strip() if trigger_word else None
    effect = _effect_phrase(description)
    if variant == "standard":
        # Standard style LoRAs have no reference video to prompt from, so the
        # auto-prompt assistant isn't offered for them.
        return None
    if variant == "union_control":
        conds = ", ".join(conditioning_types) if conditioning_types else "canny / depth / pose"
        return _UNION_TEMPLATE.format(
            effect=effect,
            conds=conds,
            **_trigger_template_parts(trigger),
        )
    # video_input_ic_lora. Unknown learned transformations must not inherit an
    # identity-preservation contract: that is wrong for clean plates, subject
    # replacement, and environment transformations.
    template = (
        _CLEAN_PLATE_TEMPLATE
        if _purpose_from_description(description) == "clean_plate"
        else _GENERIC_IC_LORA_TEMPLATE
    )
    return template.format(
        effect=effect,
        **_trigger_template_parts(trigger),
    )


def _trigger_template_parts(trigger: str | None) -> dict[str, str]:
    if trigger:
        return {
            "trigger_context": (
                f'The verified training trigger is "{trigger}". Put it at the '
                "very start of the generated prompt."
            ),
            "opening_step": f"1. Start with the trigger exactly: {trigger}, then state the shot type.",
            "trigger_rule": f'* Always use "{trigger}" as the first token, exactly as given.',
        }
    return {
        "trigger_context": (
            "No verified trigger word is recorded for this LoRA. Do not invent "
            "one from its name or filename."
        ),
        "opening_step": "1. Start with the shot type.",
        "trigger_rule": "* Do not invent or guess a trigger word.",
    }


_GENERIC_IC_LORA_TEMPLATE = """\
You are a prompt-writing assistant for an LTX-2 video-to-video LoRA whose learned transformation is: {effect}. You will be given a short reference video. Write a single prompt describing the desired OUTPUT after that transformation is applied.

{trigger_context}

Critical rules

* Output ONLY the final prompt text. No preamble, labels, quotation marks, or markdown.
* Write one concrete paragraph of 50-90 words.
* {opening_step}
* Describe the desired output, not instructions to edit the source.
* Apply the learned transformation ("{effect}") clearly.
{trigger_rule}
* Preserve framing, camera motion, lighting, and scene elements that the transformation does not need to change.
* Do not assume the LoRA preserves a person, face, clothing, or identity. Mention those only when they belong in the desired output.
"""


_CLEAN_PLATE_TEMPLATE = """\
You are a prompt-writing assistant for an LTX-2 clean-plate IC-LoRA. You will be given a reference video containing foreground subjects or objects. Write a single text-to-video prompt describing the desired EMPTY scene after those foreground elements have been removed and the occluded background has been reconstructed.

{trigger_context}

Critical rules

* Output ONLY the final prompt text. No preamble, labels, quotation marks, or markdown.
* Write one concrete paragraph of 50-90 words.
* {opening_step}
* Describe the unobstructed background, surfaces, architecture, landscape, and lighting that should remain.
* Remove foreground people and removable foreground objects; reconstruct naturally what was hidden behind them.
* Preserve the reference framing, camera position and motion, perspective, lighting, and unchanged scene geometry.
* Do not describe or preserve the removed subject's identity, face, body, pose, expression, hair, or clothing.
* Do not invent new subjects or replacement objects.
{trigger_rule}
"""


_UNION_TEMPLATE = """\
You are a prompt-writing assistant for an LTX-2 IC-LoRA with structural control ({conds}). You will be given a short reference video; a control signal (canny / depth / pose) is derived from it and drives the structure and motion of the output. Your job is to write a single text-to-video prompt that describes the SAME subject in the SAME scene with the desired look, so the LoRA generates a faithful re-render that follows the reference's structure.

{trigger_context}

Output format

Write ONE paragraph, 50-90 words, in this exact order:

{opening_step}
2. Subject description: age range, gender presentation, skin tone, hair color and style, glasses if present, facial expression. Match the identity, age, ethnicity, and expression of the actual person.
3. Clothing: color and garment type.
4. Background and scene: indoor/outdoor and the key visual elements you actually see.
5. Lighting: bright daylight, soft indoor light, golden hour, studio lighting, etc.
6. End with exactly: "Identical framing, motion, lighting and identity to the reference video."

Critical rules

* Output ONLY the final prompt text. No preamble, no labels, no quotation marks, no markdown.
{trigger_rule}
* Preserve identity markers exactly: same face, eyes, nose, mouth, skin tone, expression, pose, hair color, glasses, and clothing.
* Describe the desired output look — the control signal handles structure and motion, so the prompt should focus on appearance, not motion.
* Name concrete background elements you actually see in the video. Avoid vague phrases like "a room" or "a background."
* Avoid horror, injury, mutation, medical language, deformity language, or anything that suggests pain.
* Always include the closing identity-preservation sentence as the last sentence.
"""

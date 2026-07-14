"""Gemini-backed ``LoraPromptProfiler`` implementation.

Layers (first match wins): built-in official-LTX profiles → HuggingFace card
→ user-pasted example prompt → None. See ``lora_prompt_profiler.py`` for the
DTO/Protocol and the layering rationale.

The Gemini call mirrors the video captioner (same endpoint, retry policy, and
response shape) but is text-only: we feed it the model-card markdown (or an
example prompt) plus a *meta-prompt* that asks it to return a configured
system prompt + trigger word. This is deliberately not a brittle regex over the
card — Gemini handles the varied card formats and emits a coherent, ready-to-use
system prompt that bakes in the card's exact prompt structure.
"""

from __future__ import annotations

import logging
import re
import time
from typing import cast
from urllib.parse import urlparse

from pydantic import BaseModel, Field, ValidationError

from services.http_client.http_client import HTTPClient, HttpResponseLike, HttpTimeoutError
from services.lora_prompt_profiler.lora_prompt_profiler import (
    LoraPromptProfile,
    LoraPromptProfileResult,
)
from services.services_utils import JSONValue

logger = logging.getLogger(__name__)

_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
_RETRYABLE_STATUS = frozenset({429, 500, 503})
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 1.0

_HF_README_URL = "https://huggingface.co/{repo}/raw/main/README.md"
# Public model metadata endpoint. Returns the card's structured `cardData`
# (including `widget` example prompts) as JSON. Unlike the raw README URL, this
# works for gated repos — the metadata is public even when the files aren't.
_HF_MODEL_API_URL = "https://huggingface.co/api/models/{repo}"


# ---------------------------------------------------------------------
# Built-in profiles for the official LTX IC-LoRAs
# ---------------------------------------------------------------------
#
# These are the LoRAs whose correct prompt structure is NOT the app's default
# single-paragraph trigger-first template — they use a two-panel
# "Reference shows … / Edited shows …" format with a specific reference-video
# requirement, and the default template silently fails to activate them. We
# match by a lowercase substring of the imported filename OR the user-given
# name, so the user gets the exact template instantly with no network call.
# Other official LoRAs (e.g. cross-eyed) that work with the default template are
# intentionally left out so we don't regress a working case.

_COLORIZATION_TEMPLATE = """\
You are writing a text-to-video prompt that activates the LTX-2 IC-LoRA "Colorization" adapter for a given reference video.

This adapter uses a strict two-panel prompt structure and requires a GRAYSCALE reference video. Write the prompt in EXACTLY this two-line format:

Reference shows: <describe the reference video — the subject, pose/expression, framing, setting, and background geometry, described in grayscale terms>.
Edited shows: COLORIZE <describe the SAME scene now rendered in full natural color — realistic skin tones, clothing colors, object colors, and environment colors>.

Rules:
* The reference video MUST be treated as grayscale; describe it in grayscale terms.
* Preserve the subject's identity, pose, framing, and background geometry exactly from the reference — color is the ONLY change.
* Start the "Edited shows" line with the trigger word COLORIZE.
* Output ONLY the two-line prompt. No preamble, no labels, no quotation marks, no markdown.
"""

_WATER_TEMPLATE = """\
You are writing a text-to-video prompt that activates the LTX-2 IC-LoRA "Water Simulation" adapter for a given reference video.

This adapter uses a strict two-panel prompt structure and requires a DRY reference video (no water present). Write the prompt in EXACTLY this two-line format:

Reference shows: <describe the reference video — the subject, pose/expression, framing, setting, and background geometry, with no water present>.
Edited shows: ADD WATER <describe the SAME scene with water naturally introduced — where it appears, how it behaves, and how it interacts with the subject/scene>.

Rules:
* The reference video must be dry (no water); the edited version introduces water.
* Preserve the subject's identity, pose, framing, and background geometry exactly from the reference.
* Start the "Edited shows" line with the trigger word ADD WATER.
* Output ONLY the two-line prompt. No preamble, no labels, no quotation marks, no markdown.
"""

_INSTANT_SHAVE_TEMPLATE = """\
You are writing a text-to-video prompt that activates the LTX-2 IC-LoRA "Instant Shave" adapter for a given reference video.

This adapter uses a strict two-panel prompt structure to remove facial hair. Write the prompt in EXACTLY this two-line format:

Reference shows: <describe the reference video — the subject, their facial hair (beard/stubble/moustache), pose/expression, framing, setting, and background geometry>.
Edited shows: REMOVEBEARD <describe the SAME subject now clean-shaven, with the skin tone and jawline unchanged>.

Rules:
* Preserve the subject's identity, pose, framing, clothing, and background geometry exactly from the reference — facial hair is the ONLY change.
* Start the "Edited shows" line with the trigger word REMOVEBEARD.
* Output ONLY the two-line prompt. No preamble, no labels, no quotation marks, no markdown.
"""

_DEBLUR_TEMPLATE = """\
You are writing a text-to-video prompt that activates the LTX-2 IC-LoRA "Deblur" adapter for a given reference video.

This adapter uses a strict two-panel prompt structure to sharpen an out-of-focus reference. Write the prompt in EXACTLY this two-line format:

Reference shows: <describe the reference video — the subject, pose/expression, framing, setting, and background geometry, described as heavily out of focus with soft defocused blur and no fine detail>.
Edited shows: DEBLUR <describe the SAME scene now in sharp focus with crisp detail and clean edges>.

Rules:
* Preserve the subject's identity, pose, framing, and background geometry exactly from the reference — focus and sharpness are the ONLY change.
* Start the "Edited shows" line with the trigger word DEBLUR.
* Output ONLY the two-line prompt. No preamble, no labels, no quotation marks, no markdown.
"""

# (substring matched case-insensitively against filename+name) -> profile
_BUILTIN_PROFILES: tuple[tuple[str, LoraPromptProfile], ...] = (
    (
        "colorization",
        LoraPromptProfile(trigger_word="COLORIZE", system_prompt=_COLORIZATION_TEMPLATE),
    ),
    (
        "water-simulation",
        LoraPromptProfile(trigger_word="ADD WATER", system_prompt=_WATER_TEMPLATE),
    ),
    (
        "water_simulation",
        LoraPromptProfile(trigger_word="ADD WATER", system_prompt=_WATER_TEMPLATE),
    ),
    (
        "instant-shave",
        LoraPromptProfile(trigger_word="REMOVEBEARD", system_prompt=_INSTANT_SHAVE_TEMPLATE),
    ),
    (
        "instant_shave",
        LoraPromptProfile(trigger_word="REMOVEBEARD", system_prompt=_INSTANT_SHAVE_TEMPLATE),
    ),
    (
        "deblur",
        LoraPromptProfile(trigger_word="DEBLUR", system_prompt=_DEBLUR_TEMPLATE),
    ),
)


def _builtin_profile(name: str, filename: str) -> LoraPromptProfile | None:
    # Normalize separators so "Water Simulation", "water-simulation", and
    # "water_simulation" all match the same profile regardless of how the user
    # named the file/entry.
    haystack = re.sub(r"[-_\s]+", " ", f"{filename} {name}".lower())
    for needle, profile in _BUILTIN_PROFILES:
        if re.sub(r"[-_\s]+", " ", needle) in haystack:
            return profile
    return None


# ---------------------------------------------------------------------
# HuggingFace card fetch
# ---------------------------------------------------------------------
#
# Two sources, tried in order:
#   1. The raw README (`/raw/main/README.md`) — richest (prose + examples),
#      but 401s for gated repos and 404s when the README is missing or on a
#      non-`main` default branch.
#   2. The public model API (`/api/models/{repo}`) — returns the card's
#      structured `cardData` as JSON, including the `widget` example prompts
#      (the exact prompts the LoRA was trained on). This is public even for
#      gated repos, so it's the reliable fallback when the raw README fails.

def _hf_repo_from_url(url: str) -> str | None:
    """Extract ``<org>/<repo>`` from a HuggingFace URL.

    Accepts the common forms: ``…/<org>/<repo>``, ``…/<org>/<repo>/tree/main``,
    ``…/<org>/<repo>/blob/main/<file>``, ``…/<org>/<repo>/resolve/main/<file>``.
    Returns None for anything that isn't a two-segment repo path.
    """
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return None
    if parsed.scheme not in ("", "http", "https"):
        return None
    host = (parsed.netloc or parsed.path).lower()
    if "huggingface.co" not in host:
        return None
    # Path segments after the host (drop a leading 'huggingface.co' if it ended
    # up in path for schemeless URLs).
    segments = [s for s in parsed.path.split("/") if s]
    if segments and segments[0].lower().endswith("huggingface.co"):
        segments = segments[1:]
    if len(segments) < 2:
        return None
    org, repo = segments[0], segments[1]
    if org in {"datasets", "spaces", "settings", "docs"}:
        return None
    return f"{org}/{repo}"


def _extract_widget_texts(card_json: JSONValue) -> list[str]:
    """Pull the widget example prompts out of the HF model API response.

    The API exposes them as `widgetData[]` (top level) or `cardData.widget[]`.
    Each item is `{"text": "...", "output": {...}}`; the `text` is the exact
    example prompt, which carries the LoRA's trigger word + prompt structure.
    """
    widget: JSONValue = None
    if isinstance(card_json, dict):
        widget = card_json.get("widgetData")
        if widget is None:
            card_data = card_json.get("cardData")
            if isinstance(card_data, dict):
                widget = card_data.get("widget")
    if not isinstance(widget, list):
        return []
    texts: list[str] = []
    for item in widget:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return texts


def _synthesize_card_from_api(repo: str, card_json: JSONValue) -> str | None:
    """Build a Gemini-feedable card string from the HF model API JSON."""
    if not isinstance(card_json, dict):
        return None
    texts = _extract_widget_texts(card_json)
    tags_val = card_json.get("tags")
    tags = (
        ", ".join(t for t in tags_val if isinstance(t, str))
        if isinstance(tags_val, list) and tags_val
        else ""
    )
    pipeline = card_json.get("pipeline_tag")
    pipeline_str = pipeline if isinstance(pipeline, str) and pipeline else ""
    if not texts and not tags and not pipeline_str:
        return None
    parts = [f"HuggingFace model card for {repo}."]
    if tags:
        parts.append(f"Tags: {tags}")
    if pipeline_str:
        parts.append(f"Pipeline: {pipeline_str}")
    if texts:
        parts.append("Example prompts the LoRA was trained on:")
        parts.extend(f"- {t}" for t in texts)
    return "\n".join(parts)


def _fetch_hf_card_from_api(repo: str, http: HTTPClient) -> str | None:
    url = _HF_MODEL_API_URL.format(repo=repo)
    try:
        response = http.get(url, headers={"User-Agent": "ltx-desktop"}, timeout=30)
    except HttpTimeoutError:
        logger.warning("lora_prompt_profiler: HF model API fetch timed out for %s", repo)
        return None
    except Exception as exc:  # noqa: BLE001 — any transport failure is non-fatal here
        logger.warning("lora_prompt_profiler: HF model API fetch failed for %s: %s", repo, exc)
        return None
    if response.status_code != 200:
        logger.info(
            "lora_prompt_profiler: HF model API fetch for %s returned status=%d",
            repo,
            response.status_code,
        )
        return None
    try:
        data = cast(JSONValue, response.json())
    except Exception as exc:  # noqa: BLE001 — malformed JSON is non-fatal
        logger.warning("lora_prompt_profiler: HF model API returned non-JSON for %s: %s", repo, exc)
        return None
    return _synthesize_card_from_api(repo, data)


def _fetch_hf_card(repo: str, http: HTTPClient) -> str | None:
    """Fetch a HuggingFace model card as text, falling back to the model API.

    Returns the raw README when it's reachable (200 + non-empty), otherwise a
    synthesized card from the public model API (which works for gated repos).
    On a transport error (network down) there's no fallback — the API would
    fail too — so the caller reports a failed profile.
    """
    readme_url = _HF_README_URL.format(repo=repo)
    try:
        response = http.get(readme_url, headers={"User-Agent": "ltx-desktop"}, timeout=30)
    except HttpTimeoutError:
        logger.warning("lora_prompt_profiler: HF README fetch timed out for %s", repo)
        return None
    except Exception as exc:  # noqa: BLE001 — transport error: don't fall back (network likely down)
        logger.warning("lora_prompt_profiler: HF README fetch failed for %s: %s", repo, exc)
        return None
    if response.status_code == 200:
        text = response.text
        if text.strip():
            return text
    else:
        logger.info(
            "lora_prompt_profiler: HF README fetch for %s returned status=%d",
            repo,
            response.status_code,
        )
    return _fetch_hf_card_from_api(repo, http)


# ---------------------------------------------------------------------
# Gemini meta-prompt
# ---------------------------------------------------------------------

def _meta_prompt(*, source_label: str, content: str) -> str:
    return f"""You are configuring a prompt-writing assistant for an LTX-2 video LoRA adapter.

Below is {source_label}. Read it and work out how to activate the LoRA, then write a SYSTEM PROMPT that another AI will follow to watch a reference video and write a per-video generation prompt that reliably activates this LoRA.

The system prompt you write must instruct the writer to:
- Use the LoRA's exact trigger word (if any) where the source puts it (usually the first token of the example prompts).
- Follow the EXACT prompt structure the LoRA was trained on — copy the structure from the source verbatim (e.g. the two-panel "Reference shows: … / Edited shows: <trigger> …" format, or a single descriptive paragraph). Keep the same ordering, clauses, and closing identity-preservation sentence the source uses.
- Preserve the subject's identity, pose, framing, and background geometry from the reference.
- Describe the SAME scene as the reference video, applying only the LoRA's effect.
- Honour any reference-video requirement the source states (e.g. grayscale input for colorization, dry input for water) — include it as an explicit instruction.
- Output ONLY the final prompt text — no preamble, labels, quotation marks, or markdown.

Respond in EXACTLY this format and nothing else:
TRIGGER: <the single trigger word, or NONE if there is no trigger>
SYSTEM_PROMPT:
<the system prompt text, possibly multiple lines>

=== {source_label} BEGIN ===
{content}
=== {source_label} END ===
"""


class _GeminiPart(BaseModel):
    text: str


class _GeminiContent(BaseModel):
    parts: list[_GeminiPart] = Field(min_length=1)


class _GeminiCandidate(BaseModel):
    content: _GeminiContent
    finishReason: str | None = None


class _GeminiResponsePayload(BaseModel):
    candidates: list[_GeminiCandidate] = Field(min_length=1)


def _call_gemini_text(http: HTTPClient, *, api_key: str, prompt: str) -> str | None:
    """Text-only Gemini call. Returns the model's text, or None on any failure."""
    payload: dict[str, JSONValue] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 2048,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    response: HttpResponseLike | None = None
    status = 0
    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = http.post(
                _GEMINI_URL,
                headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
                json_payload=payload,
                timeout=60,
            )
        except HttpTimeoutError:
            logger.warning("lora_prompt_profiler: Gemini call timed out")
            return None
        except Exception as exc:  # noqa: BLE001 — non-fatal
            logger.warning("lora_prompt_profiler: Gemini call failed: %s", exc)
            return None
        status = response.status_code
        if status == 200:
            break
        if status in _RETRYABLE_STATUS and attempt < _MAX_ATTEMPTS - 1:
            time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
            continue
        break
    if status != 200 or response is None:
        logger.info("lora_prompt_profiler: Gemini returned status=%d", status)
        return None
    try:
        parsed = _GeminiResponsePayload.model_validate(response.json())
    except ValidationError:
        logger.warning("lora_prompt_profiler: could not parse Gemini response")
        return None
    text = parsed.candidates[0].content.parts[0].text.strip()
    return text or None


def _parse_profile(raw: str) -> LoraPromptProfile | None:
    """Parse the ``TRIGGER:`` / ``SYSTEM_PROMPT:`` envelope Gemini emits."""
    lines = raw.splitlines()
    trigger: str | None = None
    sys_idx: int | None = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.upper().startswith("TRIGGER:"):
            value = stripped.split(":", 1)[1].strip()
            trigger = None if value.upper() == "NONE" else value
        elif stripped.upper().startswith("SYSTEM_PROMPT:"):
            sys_idx = i
            break
    if sys_idx is None:
        return None
    body = "\n".join(lines[sys_idx + 1:]).strip()
    if not body:
        return None
    return LoraPromptProfile(trigger_word=trigger, system_prompt=body)


def _profile_via_gemini(
    http: HTTPClient, *, api_key: str, source_label: str, content: str
) -> LoraPromptProfile | None:
    if not api_key:
        return None
    raw = _call_gemini_text(http, api_key=api_key, prompt=_meta_prompt(source_label=source_label, content=content))
    if raw is None:
        return None
    profile = _parse_profile(raw)
    if profile is None:
        logger.warning("lora_prompt_profiler: Gemini response did not match the expected envelope")
    return profile


# ---------------------------------------------------------------------
# Public implementation
# ---------------------------------------------------------------------

class GeminiLoraPromptProfiler:
    """``LoraPromptProfiler`` backed by a built-in table + HuggingFace + Gemini."""

    def __init__(self, http: HTTPClient) -> None:
        self._http = http

    def profile(
        self,
        *,
        name: str,
        filename: str,
        variant: str,
        huggingface_url: str | None,
        example_prompt: str | None,
        api_key: str,
    ) -> LoraPromptProfileResult:
        del variant  # reserved for future variant-specific defaults

        builtin = _builtin_profile(name, filename)
        if builtin is not None:
            logger.info("lora_prompt_profiler: matched built-in profile for '%s'", name)
            return LoraPromptProfileResult(
                status="builtin",
                message="Matched a built-in official LTX profile — system prompt configured.",
                profile=builtin,
            )

        hf_url = (huggingface_url or "").strip()
        example = (example_prompt or "").strip()

        if not hf_url and not example:
            return LoraPromptProfileResult(
                status="skipped",
                message="No HuggingFace URL or example prompt provided — using the default prompt.",
            )

        if not api_key:
            return LoraPromptProfileResult(
                status="failed",
                message="No Gemini API key set — add one in Settings to auto-configure the system prompt.",
            )

        # Try the HuggingFace card first; if it fails and an example prompt was
        # also supplied, fall back to the example. Collect per-source failure
        # reasons so the final message tells the user exactly what went wrong
        # (previously this was all silent).
        attempts: list[str] = []

        if hf_url:
            repo = _hf_repo_from_url(hf_url)
            if repo is None:
                logger.info("lora_prompt_profiler: could not parse HF URL: %s", huggingface_url)
                attempts.append("That HuggingFace URL couldn't be read.")
            else:
                card = _fetch_hf_card(repo, self._http)
                if card is None:
                    attempts.append(
                        f"Couldn't fetch the HuggingFace page for {repo} "
                        "(it may be gated/private, missing a README, or a network error)."
                    )
                else:
                    profile = _profile_via_gemini(
                        self._http,
                        api_key=api_key,
                        source_label="the LoRA's HuggingFace model card",
                        content=card,
                    )
                    if profile is None:
                        attempts.append("Gemini couldn't produce a system prompt from the HuggingFace page.")
                    else:
                        return LoraPromptProfileResult(
                            status="configured",
                            message="System prompt configured from the HuggingFace page via Gemini.",
                            profile=profile,
                        )

        if example:
            profile = _profile_via_gemini(
                self._http,
                api_key=api_key,
                source_label="an example prompt the LoRA was trained on",
                content=example,
            )
            if profile is None:
                attempts.append("Gemini couldn't produce a system prompt from the example prompt.")
            else:
                return LoraPromptProfileResult(
                    status="configured",
                    message="System prompt configured from the example prompt via Gemini.",
                    profile=profile,
                )

        return LoraPromptProfileResult(
            status="failed",
            message=" ".join(attempts) + " You can edit the system prompt manually.",
        )

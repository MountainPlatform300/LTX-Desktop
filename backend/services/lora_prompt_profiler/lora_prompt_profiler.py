"""LoRA prompt profiler — derives a per-LoRA system prompt + trigger word.

An imported LoRA only activates if the generation prompt matches the structure
and trigger word it was trained on. That information lives *outside* the
.safetensors file (on the model card / in an example prompt), so the importer
can't recover it from the filename alone — which is why some imported LoRAs
silently produce no effect (see `services/lora_diagnostics.py` for the load-side
proof that this is a prompting issue, not a loading issue).

The profiler layers four sources, first match wins:

1. Built-in profiles for the official LTX IC-LoRAs (matched by filename) —
   instant, offline, exact.
2. A HuggingFace model-card URL — the card markdown is fetched server-side and
   fed to Gemini, which returns a configured system prompt + trigger word.
3. A user-pasted example prompt — fed to the same Gemini meta-prompt, for LoRAs
   with no HF page (Civitai / direct files / Discord).
4. None — the caller falls back to the existing name-derived default.

The DTO + Protocol live here; the Gemini-backed implementation lives in
``gemini_lora_prompt_profiler.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True, slots=True)
class LoraPromptProfile:
    """A profiled LoRA's activation info, used to seed the auto-prompt assistant.

    ``trigger_word`` is the token/phrase the adapter was trained on (e.g.
    ``COLORIZE``, ``ADD WATER``, ``REMOVEBEARD``); ``None`` when the LoRA uses no
    explicit trigger (a descriptive prompt activates it). ``system_prompt`` is a
    ready-to-use prompt-writing-assistant system prompt that bakes in the card's
    exact prompt structure + trigger + identity/reference clauses.
    """

    trigger_word: str | None
    system_prompt: str


# Outcome of a profiling attempt. ``status`` is the machine-readable result the
# API/frontend switches on; ``message`` is a user-facing sentence explaining it
# (shown in the import modal so profiling is never silent again); ``profile`` is
# the derived profile on the success states.
#
# Statuses:
# - "builtin"    — matched a built-in official-LTX profile (offline, exact).
# - "configured" — Gemini produced a system prompt from the HF card / example.
# - "skipped"    — no source was provided; the LoRA keeps the name-derived
#                  default. Not an error.
# - "failed"     — a source was provided but profiling couldn't complete (no
#                  Gemini key, HF fetch failed, Gemini call failed, parse
#                  failed). ``message`` says which.
LoraPromptProfileStatus = Literal["builtin", "configured", "skipped", "failed"]


@dataclass(frozen=True, slots=True)
class LoraPromptProfileResult:
    status: LoraPromptProfileStatus
    message: str
    profile: LoraPromptProfile | None = None


class LoraPromptProfiler(Protocol):
    """Derive a LoRA's activation profile from a card URL / example / filename."""

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
        """Return the profiling outcome. Never raises — a profiling failure
        (network, parse, bad URL) is reported as ``status="failed"`` with a
        user-facing ``message`` so the import never fails and is never silent
        just because profiling did. ``profile`` is set on success states."""
        ...


class NullLoraPromptProfiler:
    """No-op profiler: always reports "skipped" (caller uses the default template).

    The ``ServiceBundle`` default — keeps bundles that don't care about
    profiling (e.g. settings-only tests) constructible without a real HTTP
    client. Production wires the Gemini-backed implementation explicitly.
    """

    def profile(
        self,
        *,
        name: str,  # noqa: ARG002
        filename: str,  # noqa: ARG002
        variant: str,  # noqa: ARG002
        huggingface_url: str | None,  # noqa: ARG002
        example_prompt: str | None,  # noqa: ARG002
        api_key: str,  # noqa: ARG002
    ) -> LoraPromptProfileResult:
        return LoraPromptProfileResult(
            status="skipped",
            message="Prompt profiling is not configured — using the default prompt.",
        )

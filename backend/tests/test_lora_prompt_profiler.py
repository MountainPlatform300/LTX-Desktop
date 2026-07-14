"""Tests for the Gemini-backed LoRA prompt profiler.

Exercises the real ``GeminiLoraPromptProfiler`` against a ``FakeHTTPClient``
(no mocks): built-in official profiles, HuggingFace card fetch + Gemini
meta-prompt, example-prompt fallback, URL parsing, and the never-raises
contract. The profiler returns a ``LoraPromptProfileResult`` whose ``status``
+ ``message`` make profiling outcomes visible to the import modal. The
handler/route integration is covered in ``test_imported_lora_library.py`` via
``FakeLoraPromptProfiler``.
"""

from __future__ import annotations

import pytest

from services.lora_prompt_profiler.gemini_lora_prompt_profiler import (
    GeminiLoraPromptProfiler,
    _hf_repo_from_url,
)
from services.lora_prompt_profiler.lora_prompt_profiler import (
    NullLoraPromptProfiler,
)
from tests.fakes.services import FakeHTTPClient, FakeResponse


def _gemini_response(text: str) -> FakeResponse:
    return FakeResponse(
        status_code=200,
        json_payload={
            "candidates": [
                {"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}
            ]
        },
    )


def _profile(**overrides):
    base = dict(
        name="Custom LoRA",
        filename="custom.safetensors",
        variant="video_input_ic_lora",
        huggingface_url=None,
        example_prompt=None,
        api_key="key",
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------
# Built-in official profiles
# ---------------------------------------------------------------------


class TestBuiltinProfiles:
    def test_colorization_by_filename(self):
        profiler = GeminiLoraPromptProfiler(FakeHTTPClient())
        result = profiler.profile(
            **_profile(
                name="Whatever",
                filename="ltx-2.3-22b-ic-lora-colorization-0.9.safetensors",
            )
        )
        assert result.status == "builtin"
        assert result.profile is not None
        assert result.profile.trigger_word == "COLORIZE"
        assert "Reference shows" in result.profile.system_prompt
        assert "Edited shows" in result.profile.system_prompt
        assert "COLORIZE" in result.profile.system_prompt

    def test_water_simulation_by_name(self):
        profiler = GeminiLoraPromptProfiler(FakeHTTPClient())
        result = profiler.profile(
            **_profile(name="Water Simulation", filename="adapter.safetensors")
        )
        assert result.status == "builtin"
        assert result.profile is not None
        assert result.profile.trigger_word == "ADD WATER"

    def test_instant_shave_by_filename(self):
        profiler = GeminiLoraPromptProfiler(FakeHTTPClient())
        result = profiler.profile(
            **_profile(name="X", filename="instant_shave_v1.safetensors")
        )
        assert result.status == "builtin"
        assert result.profile is not None
        assert result.profile.trigger_word == "REMOVEBEARD"

    def test_no_match_no_source_is_skipped(self):
        http = FakeHTTPClient()
        profiler = GeminiLoraPromptProfiler(http)
        result = profiler.profile(**_profile(name="Mystery", filename="mystery.safetensors"))
        assert result.status == "skipped"
        assert result.profile is None
        assert http.calls == []

    def test_deblur_by_filename(self):
        profiler = GeminiLoraPromptProfiler(FakeHTTPClient())
        result = profiler.profile(
            **_profile(
                name="Whatever",
                filename="ltx-2.3-22b-ic-lora-deblur-0.9.safetensors",
            )
        )
        assert result.status == "builtin"
        assert result.profile is not None
        assert result.profile.trigger_word == "DEBLUR"
        assert "Reference shows" in result.profile.system_prompt
        assert "Edited shows" in result.profile.system_prompt
        assert "DEBLUR" in result.profile.system_prompt


# ---------------------------------------------------------------------
# HuggingFace card → Gemini
# ---------------------------------------------------------------------


class TestHuggingFaceCard:
    def test_fetches_card_and_profiles_via_gemini(self):
        http = FakeHTTPClient()
        http.queue("get", FakeResponse(status_code=200, text="# Water Sim\nADD WATER ..."))
        http.queue(
            "post",
            _gemini_response(
                "TRIGGER: ADD WATER\nSYSTEM_PROMPT:\nWrite prompts starting with ADD WATER."
            ),
        )
        profiler = GeminiLoraPromptProfiler(http)
        result = profiler.profile(
            **_profile(
                name="Custom",
                huggingface_url="https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Water-Sim",
            )
        )
        assert result.status == "configured"
        assert result.profile is not None
        assert result.profile.trigger_word == "ADD WATER"
        assert result.profile.system_prompt.startswith("Write prompts starting with ADD WATER")
        assert http.calls[0].method == "get"
        assert "raw/main/README.md" in http.calls[0].url
        assert "generativelanguage.googleapis.com" in http.calls[1].url

    def test_card_not_found_is_failed(self):
        http = FakeHTTPClient()
        # Raw README 404, then the model-API fallback also 404.
        http.queue("get", FakeResponse(status_code=404, text=""))
        http.queue("get", FakeResponse(status_code=404, text=""))
        profiler = GeminiLoraPromptProfiler(http)
        result = profiler.profile(
            **_profile(huggingface_url="https://huggingface.co/org/repo")
        )
        assert result.status == "failed"
        assert result.profile is None
        assert "fetch" in result.message.lower()
        # Card fetch failed -> no Gemini call.
        assert all(c.method == "get" for c in http.calls)

    def test_card_fetch_transport_error_is_failed(self):
        http = FakeHTTPClient()
        http.queue("get", RuntimeError("network down"))
        profiler = GeminiLoraPromptProfiler(http)
        result = profiler.profile(
            **_profile(huggingface_url="https://huggingface.co/org/repo")
        )
        assert result.status == "failed"
        assert result.profile is None

    def test_invalid_hf_url_is_failed(self):
        http = FakeHTTPClient()
        profiler = GeminiLoraPromptProfiler(http)
        result = profiler.profile(
            **_profile(huggingface_url="https://example.com/x")
        )
        assert result.status == "failed"
        assert result.profile is None
        assert http.calls == []


# ---------------------------------------------------------------------
# Example-prompt fallback
# ---------------------------------------------------------------------


class TestExamplePrompt:
    def test_profiles_via_gemini(self):
        http = FakeHTTPClient()
        http.queue(
            "post",
            _gemini_response(
                "TRIGGER: FOOBAZ\nSYSTEM_PROMPT:\nUse FOOBAZ then describe the scene."
            ),
        )
        profiler = GeminiLoraPromptProfiler(http)
        result = profiler.profile(**_profile(example_prompt="FOOBAZ a man in a park"))
        assert result.status == "configured"
        assert result.profile is not None
        assert result.profile.trigger_word == "FOOBAZ"
        assert "FOOBAZ" in result.profile.system_prompt

    def test_trigger_none_is_parsed(self):
        http = FakeHTTPClient()
        http.queue(
            "post",
            _gemini_response("TRIGGER: NONE\nSYSTEM_PROMPT:\nDescribe the scene."),
        )
        profiler = GeminiLoraPromptProfiler(http)
        result = profiler.profile(**_profile(example_prompt="a descriptive prompt"))
        assert result.status == "configured"
        assert result.profile is not None
        assert result.profile.trigger_word is None

    def test_no_api_key_is_failed(self):
        http = FakeHTTPClient()
        profiler = GeminiLoraPromptProfiler(http)
        result = profiler.profile(**_profile(example_prompt="foo", api_key=""))
        assert result.status == "failed"
        assert result.profile is None
        assert "gemini" in result.message.lower()
        assert http.calls == []

    def test_gemini_non_retryable_error_is_failed(self):
        http = FakeHTTPClient()
        http.queue("post", FakeResponse(status_code=400, text="bad request"))
        profiler = GeminiLoraPromptProfiler(http)
        result = profiler.profile(**_profile(example_prompt="foo"))
        assert result.status == "failed"
        assert result.profile is None

    def test_malformed_envelope_is_failed(self):
        http = FakeHTTPClient()
        http.queue("post", _gemini_response("nope, no envelope here"))
        profiler = GeminiLoraPromptProfiler(http)
        result = profiler.profile(**_profile(example_prompt="foo"))
        assert result.status == "failed"
        assert result.profile is None

    def test_gemini_transport_error_is_failed(self):
        http = FakeHTTPClient()
        http.queue("post", RuntimeError("boom"))
        profiler = GeminiLoraPromptProfiler(http)
        result = profiler.profile(**_profile(example_prompt="foo"))
        assert result.status == "failed"
        assert result.profile is None

    def test_hf_failure_falls_back_to_example(self):
        # If the HF card can't be fetched but an example prompt was also
        # supplied, the profiler falls back to the example and succeeds.
        http = FakeHTTPClient()
        # Raw README 404, model-API fallback 404, then the example-prompt Gemini call.
        http.queue("get", FakeResponse(status_code=404, text=""))
        http.queue("get", FakeResponse(status_code=404, text=""))
        http.queue(
            "post",
            _gemini_response("TRIGGER: FALLBACK\nSYSTEM_PROMPT:\nUse FALLBACK."),
        )
        profiler = GeminiLoraPromptProfiler(http)
        result = profiler.profile(
            **_profile(
                huggingface_url="https://huggingface.co/org/repo",
                example_prompt="FALLBACK a scene",
            )
        )
        assert result.status == "configured"
        assert result.profile is not None
        assert result.profile.trigger_word == "FALLBACK"

    def test_gated_repo_falls_back_to_model_api(self):
        # Gated repos 401 on the raw README; the public model API still returns
        # the card's widget example prompts, which Gemini profiles from.
        http = FakeHTTPClient()
        http.queue("get", FakeResponse(status_code=401, text=""))  # raw README (gated)
        http.queue(
            "get",
            FakeResponse(
                status_code=200,
                json_payload={
                    "id": "Lightricks/LTX-2.3-22b-IC-LoRA-Deblur",
                    "tags": ["ltx-video", "ic-lora", "deblur"],
                    "pipeline_tag": "video-to-video",
                    "widgetData": [
                        {
                            "text": (
                                "Reference shows a butterfly on a flower, heavily out of focus. "
                                "Edited shows the same scene in sharp focus. DEBLUR a butterfly "
                                "on a flower."
                            )
                        },
                    ],
                },
            ),
        )
        http.queue(
            "post",
            _gemini_response(
                "TRIGGER: DEBLUR\nSYSTEM_PROMPT:\nUse DEBLUR in a two-panel prompt."
            ),
        )
        profiler = GeminiLoraPromptProfiler(http)
        result = profiler.profile(
            **_profile(
                name="Custom",
                filename="adapter.safetensors",
                huggingface_url="https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Deblur",
            )
        )
        assert result.status == "configured"
        assert result.profile is not None
        assert result.profile.trigger_word == "DEBLUR"
        # First call = raw README; second = model API; third = Gemini.
        assert "raw/main/README.md" in http.calls[0].url
        assert "/api/models/" in http.calls[1].url
        # The widget example text must be fed to Gemini.
        gemini_payload = http.calls[2].json_payload
        contents = gemini_payload["contents"][0]["parts"][0]["text"]
        assert "DEBLUR a butterfly" in contents


# ---------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Colorization", "Lightricks/LTX-2.3-22b-IC-LoRA-Colorization"),
        ("https://huggingface.co/org/repo/tree/main", "org/repo"),
        ("https://huggingface.co/org/repo/blob/main/README.md", "org/repo"),
        ("https://huggingface.co/org/repo/resolve/main/weights.safetensors", "org/repo"),
        ("huggingface.co/org/repo", "org/repo"),
        ("https://huggingface.co/datasets/org/repo", None),
        ("https://example.com/org/repo", None),
        ("https://huggingface.co/only-one-segment", None),
        ("not a url", None),
        ("", None),
    ],
)
def test_hf_repo_from_url(url, expected):
    assert _hf_repo_from_url(url) == expected


# ---------------------------------------------------------------------
# Null profiler
# ---------------------------------------------------------------------


def test_null_profiler_reports_skipped():
    result = NullLoraPromptProfiler().profile(
        **_profile(huggingface_url="https://huggingface.co/org/repo")
    )
    assert result.status == "skipped"
    assert result.profile is None

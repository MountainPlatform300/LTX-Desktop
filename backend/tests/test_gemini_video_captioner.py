"""Unit tests for the Gemini-backed video captioner.

These exercise the real `GeminiVideoCaptioner` against a stub `HTTPClient` so
we can assert on the request payload and the response parsing (truncation
detection) without touching the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from services.interfaces import JSONValue
from services.video_captioner.gemini_video_captioner import GeminiVideoCaptioner
from services.video_captioner.video_captioner import VideoCaptionerError


@dataclass
class _FakeResponse:
    status_code: int = 200
    text: str = ""
    _json: Any = None

    def json(self) -> object:
        return self._json


@dataclass
class _FakeHTTP:
    """Captures the last request payload and returns a queued response."""

    response: _FakeResponse
    captured: dict[str, JSONValue] = field(default_factory=dict)

    def post(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        json_payload: dict[str, JSONValue] | None = None,
        data: Any = None,
        timeout: int = 30,
    ) -> _FakeResponse:
        del url, headers, data, timeout
        if json_payload is not None:
            self.captured = dict(json_payload)
        return self.response

    def get(self, *args: Any, **kwargs: Any) -> _FakeResponse:  # pragma: no cover - unused
        raise NotImplementedError

    def put(self, *args: Any, **kwargs: Any) -> _FakeResponse:  # pragma: no cover - unused
        raise NotImplementedError


def _gemini_ok(text: str) -> _FakeResponse:
    return _FakeResponse(
        status_code=200,
        _json={"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]},
    )


def _write_clip(tmp_path) -> str:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake mp4 bytes")
    return str(clip)


def test_caption_disables_thinking_and_raises_output_budget(tmp_path) -> None:
    """gemini-2.5-flash is a thinking model; reasoning tokens come out of the
    same maxOutputTokens budget. We must disable thinking (so a verbose
    per-LoRA auto-prompt template can't burn the whole budget on reasoning and
    return MAX_TOKENS) and use a generous output ceiling."""
    http = _FakeHTTP(response=_gemini_ok("a short caption"))
    captioner = GeminiVideoCaptioner(http)

    captioner.caption(
        video_path=_write_clip(tmp_path),
        api_key="key",
        with_audio=False,
        instructions="You are a prompt-writing assistant ... 8 structured fields ...",
    )

    config = http.captured["generationConfig"]
    assert config["maxOutputTokens"] == 2048
    assert config["thinkingConfig"]["thinkingBudget"] == 0


def test_caption_returns_text_on_stop(tmp_path) -> None:
    http = _FakeHTTP(response=_gemini_ok("a person walking in a park"))
    captioner = GeminiVideoCaptioner(http)

    out = captioner.caption(
        video_path=_write_clip(tmp_path), api_key="key", with_audio=False
    )
    assert out == "a person walking in a park"


def test_caption_raises_on_max_tokens_truncation(tmp_path) -> None:
    http = _FakeHTTP(
        response=_FakeResponse(
            status_code=200,
            _json={
                "candidates": [
                    {"content": {"parts": [{"text": "truncated mid senten"}]}, "finishReason": "MAX_TOKENS"}
                ]
            },
        )
    )
    captioner = GeminiVideoCaptioner(http)

    with pytest.raises(VideoCaptionerError) as exc:
        captioner.caption(video_path=_write_clip(tmp_path), api_key="key", with_audio=False)
    assert "truncated" in exc.value.detail.lower()


def test_caption_does_not_surface_upstream_error_body(tmp_path) -> None:
    http = _FakeHTTP(
        response=_FakeResponse(
            status_code=400,
            text='{"api_key":"gemini_must_not_be_exposed"}',
        )
    )
    captioner = GeminiVideoCaptioner(http)

    with pytest.raises(VideoCaptionerError) as exc:
        captioner.caption(
            video_path=_write_clip(tmp_path),
            api_key="key",
            with_audio=False,
        )

    assert "gemini_must_not_be_exposed" not in exc.value.detail
    assert "Gemini rejected" in exc.value.detail

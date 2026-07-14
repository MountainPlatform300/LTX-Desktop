from __future__ import annotations

import torch
from safetensors.torch import save as save_safetensors

from services.text_encoder.ltx_text_encoder import LTXTextEncoder
from tests.fakes.services import FakeHTTPClient, FakeResponse


def _encoder(http: FakeHTTPClient) -> LTXTextEncoder:
    encoder = LTXTextEncoder(torch.device("cpu"), http, "https://api.example.test")
    encoder.get_model_id_from_checkpoint = lambda _path: "model-id"  # type: ignore[method-assign]
    return encoder


def test_api_embeddings_accept_validated_safetensors() -> None:
    http = FakeHTTPClient()
    combined = torch.ones((1, 2, 4_128), dtype=torch.float32)
    http.queue(
        "post",
        FakeResponse(
            status_code=200,
            headers={"Content-Type": "application/x-safetensors"},
            content=save_safetensors({"embeddings": combined}),
        ),
    )

    result = _encoder(http).encode_via_api(
        "a prompt",
        "api-key",
        "model.safetensors",
        False,
    )

    assert result is not None
    assert result.video_context.shape == (1, 2, 4_096)
    assert result.audio_context is not None
    assert result.audio_context.shape == (1, 2, 32)
    assert http.calls[0].headers is not None
    assert "application/x-safetensors" in http.calls[0].headers["Accept"]


def test_api_embeddings_reject_legacy_pickle_payload() -> None:
    http = FakeHTTPClient()
    http.queue(
        "post",
        FakeResponse(
            status_code=200,
            headers={"Content-Type": "application/octet-stream"},
            content=b"\x80\x04legacy-pickle-payload",
        ),
    )

    result = _encoder(http).encode_via_api(
        "a prompt",
        "api-key",
        "model.safetensors",
        False,
    )

    assert result is None


def test_api_embeddings_reject_invalid_tensor_shape() -> None:
    http = FakeHTTPClient()
    invalid = torch.ones((1, 2, 32), dtype=torch.float32)
    http.queue(
        "post",
        FakeResponse(
            status_code=200,
            content=save_safetensors({"embeddings": invalid}),
        ),
    )

    result = _encoder(http).encode_via_api(
        "a prompt",
        "api-key",
        "model.safetensors",
        False,
    )

    assert result is None

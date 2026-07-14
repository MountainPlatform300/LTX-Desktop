"""Tests for the real ``PexelsClientImpl`` download URL guard (no mocks).

The download endpoint receives a client-supplied URL, so ``download`` must
confine it to HTTPS ``*.pexels.com`` CDN links to prevent SSRF (an
authenticated caller otherwise making the backend GET an arbitrary URL to
probe localhost / link-local services). These cases exercise the validator
directly against ``PexelsClientImpl`` with a ``FakeHTTPClient`` and assert no
network call is made for rejected URLs.
"""

from __future__ import annotations

import pytest

from services.pexels_client.pexels_client import PexelsError
from services.pexels_client.pexels_client_impl import PexelsClientImpl
from tests.fakes.services import FakeHTTPClient, FakeResponse


def _client() -> PexelsClientImpl:
    return PexelsClientImpl(FakeHTTPClient())


def _ok_bytes() -> FakeResponse:
    return FakeResponse(status_code=200, content=b"mp4-bytes")


@pytest.mark.parametrize(
    "url",
    [
        "https://videos.pexels.com/file.mp4",
        "https://images.pexels.com/photo.jpg",
        "https://videos.photos.pexels.com/clip.mp4",
        "https://PEXELS.COM/x.mp4",
    ],
)
def test_download_accepts_pexels_cdn_urls(url: str) -> None:
    c = _client()
    c._http.queue("get", _ok_bytes())  # type: ignore[attr-defined]
    assert c.download(url=url, api_key="pk") == b"mp4-bytes"


@pytest.mark.parametrize(
    "url",
    [
        "http://videos.pexels.com/file.mp4",  # not HTTPS
        "https://evil.com/file.mp4",  # wrong host
        "https://pexels.com.evil.com/file.mp4",  # host suffix spoof
        "https://evil-pexels.com/file.mp4",  # hyphen spoof
        "https://169.254.169.254/latest/meta-data",  # link-local IP
        "https://localhost:8000/api",  # localhost
        "ftp://videos.pexels.com/file.mp4",  # wrong scheme
        "",
    ],
)
def test_download_rejects_non_pexels_urls(url: str) -> None:
    c = _client()
    with pytest.raises(PexelsError):
        c.download(url=url, api_key="pk")
    # The guard must run before any network call.
    assert c._http.calls == []  # type: ignore[attr-defined]


def test_download_refuses_redirect_to_untrusted_host() -> None:
    c = _client()
    c._http.queue(  # type: ignore[attr-defined]
        "get",
        FakeResponse(
            status_code=302,
            headers={"Location": "http://127.0.0.1:8000/api/settings"},
        ),
    )

    with pytest.raises(PexelsError):
        c.download(url="https://videos.pexels.com/file.mp4", api_key="pk")

    assert len(c._http.calls) == 1  # type: ignore[attr-defined]


def test_download_allows_validated_pexels_redirect() -> None:
    c = _client()
    c._http.queue(  # type: ignore[attr-defined]
        "get",
        FakeResponse(
            status_code=302,
            headers={"location": "https://videos.photos.pexels.com/final.mp4"},
        ),
        _ok_bytes(),
    )

    result = c.download(url="https://videos.pexels.com/file.mp4", api_key="pk")

    assert result == b"mp4-bytes"
    assert [call.url for call in c._http.calls] == [  # type: ignore[attr-defined]
        "https://videos.pexels.com/file.mp4",
        "https://videos.photos.pexels.com/final.mp4",
    ]

"""Pexels stock-media service for LoRA dataset building (BYOK).

Lets the LoRA trainer search Pexels for photos/videos and download the
chosen assets into the user's training collection. BYOK via the app's
`pexels_api_key` (a single `Authorization: <key>` header — Pexels does not
use a Bearer prefix). API-only (no local backend); Protocol + real
(`PexelsClientImpl`) + fake follow the standard service convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

PexelsMediaKind = Literal["video", "photo"]


class PexelsError(Exception):
    def __init__(self, detail: str, *, status_code: int = 502) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


@dataclass(frozen=True)
class PexelsMediaResult:
    """One search hit, normalized across the photo + video endpoints."""

    id: str
    kind: PexelsMediaKind
    width: int
    height: int
    # Videos only; None for photos.
    duration_seconds: float | None
    # Thumbnail/poster to render in the grid.
    preview_url: str
    # The file the user downloads into their collection (best mp4 / full-res
    # photo) and its extension (no leading dot).
    download_url: str
    download_ext: str
    # Link back to the asset's Pexels page (required for attribution).
    pexels_url: str
    author: str
    author_url: str
    alt: str


@dataclass(frozen=True)
class PexelsSearchResult:
    items: list[PexelsMediaResult]
    page: int
    per_page: int
    total_results: int
    has_next: bool


class PexelsClient(Protocol):
    def search(
        self,
        *,
        query: str,
        media: PexelsMediaKind,
        page: int,
        per_page: int,
        orientation: str,
        api_key: str,
    ) -> PexelsSearchResult:
        """Search photos or videos. An empty `query` falls back to the
        curated/popular feed so the browser opens with content. Returns
        normalized results. Raises `PexelsError`.
        """
        ...

    def download(self, *, url: str, api_key: str) -> bytes:
        """Download an asset's file bytes from its (public CDN) URL. The
        `url` must be one returned by `search` (a Pexels-hosted file).
        Raises `PexelsError`.
        """
        ...

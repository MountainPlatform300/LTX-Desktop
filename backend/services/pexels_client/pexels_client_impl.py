"""HTTP-backed `PexelsClient` (api.pexels.com).

Search hits the photo (`/v1/search`, `/v1/curated`) and video
(`/videos/search`, `/videos/popular`) endpoints and normalizes both into
`PexelsMediaResult`. Auth is a single `Authorization: <key>` header (no
Bearer prefix). Downloads pull the chosen file straight from Pexels' public
CDN (no auth header needed) and return the bytes for the caller to persist.
"""

from __future__ import annotations

import logging
from typing import Any, cast
from urllib.parse import urlencode, urljoin, urlparse

from services.http_client.http_client import HTTPClient
from services.pexels_client.pexels_client import (
    PexelsError,
    PexelsMediaKind,
    PexelsMediaResult,
    PexelsSearchResult,
)

logger = logging.getLogger(__name__)

PEXELS_API_BASE_URL = "https://api.pexels.com"
# Prefer mp4 renditions no wider than this so the browser doesn't pull
# multi-GB 4K masters for a training clip (downscaling happens later anyway).
_MAX_VIDEO_WIDTH = 1920
# Pexels file URLs are always HTTPS under *.pexels.com (e.g.
# `images.pexels.com`, `videos.pexels.com`, `videos.photos.pexels.com`).
# The download endpoint receives a client-supplied URL, so confine it to
# that host to prevent SSRF (an authenticated caller otherwise making the
# backend fetch an arbitrary URL to probe localhost/link-local services).
_PEXELS_DOWNLOAD_HOST = "pexels.com"
_PEXELS_DOWNLOAD_HOST_SUFFIX = ".pexels.com"
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_MAX_DOWNLOAD_REDIRECTS = 3


def _validate_pexels_download_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise PexelsError("Pexels download URL must be HTTPS.", status_code=400)
    host = (parsed.hostname or "").lower()
    if not host or not (
        host == _PEXELS_DOWNLOAD_HOST or host.endswith(_PEXELS_DOWNLOAD_HOST_SUFFIX)
    ):
        raise PexelsError("Pexels download URL must be on pexels.com.", status_code=400)


class PexelsClientImpl:
    def __init__(self, http: HTTPClient, *, base_url: str = PEXELS_API_BASE_URL) -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")

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
        if not api_key:
            raise PexelsError(
                "Add a Pexels API key in Settings to browse stock media.",
                status_code=400,
            )
        q = query.strip()
        params: dict[str, str | int] = {"page": page, "per_page": per_page}
        if q:
            params["query"] = q
        if orientation:
            params["orientation"] = orientation

        if media == "photo":
            path = "/v1/search" if q else "/v1/curated"
        else:
            path = "/videos/search" if q else "/videos/popular"
        url = f"{self._base_url}{path}?{urlencode(params)}"

        logger.info("pexels.search media=%s q=%r page=%d", media, q[:60], page)
        resp = self._http.get(
            url,
            headers={"Authorization": api_key, "Accept": "application/json"},
            timeout=30,
        )
        if resp.status_code == 401:
            raise PexelsError("Invalid Pexels API key.", status_code=400)
        if resp.status_code == 429:
            raise PexelsError(
                "Pexels rate limit reached — try again later.", status_code=429
            )
        if resp.status_code != 200:
            detail = resp.text[:300] if resp.text else "Unknown error"
            raise PexelsError(f"Pexels search failed ({resp.status_code}): {detail}")

        payload = self._json_object(resp.json())
        if media == "photo":
            raw_items = [self._photo_item(p) for p in self._json_list(payload.get("photos"))]
        else:
            raw_items = [self._video_item(v) for v in self._json_list(payload.get("videos"))]
        items = [it for it in raw_items if it is not None]
        total = int(payload.get("total_results") or 0)
        eff_page = int(payload.get("page") or page)
        eff_per = int(payload.get("per_page") or per_page)
        has_next = bool(payload.get("next_page")) or (eff_page * eff_per < total)
        return PexelsSearchResult(
            items=items,
            page=eff_page,
            per_page=eff_per,
            total_results=total,
            has_next=has_next,
        )

    def download(self, *, url: str, api_key: str) -> bytes:
        # Pexels file URLs are public CDN links; no auth header required.
        # Redirects are followed manually and validated *before* each request;
        # checking only the final response URL would be too late because
        # requests follows redirects by default.
        current_url = url
        for redirect_count in range(_MAX_DOWNLOAD_REDIRECTS + 1):
            _validate_pexels_download_url(current_url)
            logger.info("pexels.download url=%s", current_url)
            resp = self._http.get(
                current_url,
                timeout=600,
                allow_redirects=False,
            )
            if resp.status_code in _REDIRECT_STATUSES:
                if redirect_count == _MAX_DOWNLOAD_REDIRECTS:
                    raise PexelsError("Pexels download redirected too many times")
                location = next(
                    (
                        value
                        for key, value in resp.headers.items()
                        if key.lower() == "location"
                    ),
                    "",
                )
                if not location:
                    raise PexelsError("Pexels download returned an invalid redirect")
                current_url = urljoin(current_url, location)
                continue
            if resp.status_code != 200 or not resp.content:
                raise PexelsError(f"Pexels download failed ({resp.status_code})")
            logger.info("pexels.download ok bytes=%d", len(resp.content))
            return resp.content
        raise PexelsError("Pexels download redirected too many times")

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _photo_item(self, raw: object) -> PexelsMediaResult | None:
        if not isinstance(raw, dict):
            return None
        photo = cast(dict[str, Any], raw)
        src = photo.get("src")
        if not isinstance(src, dict):
            return None
        src_map = cast(dict[str, Any], src)
        download_url = str(src_map.get("original") or src_map.get("large2x") or "")
        preview_url = str(
            src_map.get("large") or src_map.get("medium") or src_map.get("original") or ""
        )
        if not download_url:
            return None
        return PexelsMediaResult(
            id=str(photo.get("id") or ""),
            kind="photo",
            width=int(photo.get("width") or 0),
            height=int(photo.get("height") or 0),
            duration_seconds=None,
            preview_url=preview_url or download_url,
            download_url=download_url,
            download_ext=_ext_from_url(download_url, default="jpg"),
            pexels_url=str(photo.get("url") or ""),
            author=str(photo.get("photographer") or ""),
            author_url=str(photo.get("photographer_url") or ""),
            alt=str(photo.get("alt") or ""),
        )

    def _video_item(self, raw: object) -> PexelsMediaResult | None:
        if not isinstance(raw, dict):
            return None
        video = cast(dict[str, Any], raw)
        files = self._json_list(video.get("video_files"))
        download_url = _pick_video_file(files)
        if not download_url:
            return None
        user = video.get("user")
        user_map = cast(dict[str, Any], user) if isinstance(user, dict) else {}
        return PexelsMediaResult(
            id=str(video.get("id") or ""),
            kind="video",
            width=int(video.get("width") or 0),
            height=int(video.get("height") or 0),
            duration_seconds=float(video.get("duration") or 0) or None,
            preview_url=str(video.get("image") or ""),
            download_url=download_url,
            download_ext=_ext_from_url(download_url, default="mp4"),
            pexels_url=str(video.get("url") or ""),
            author=str(user_map.get("name") or ""),
            author_url=str(user_map.get("url") or ""),
            alt="",
        )

    @staticmethod
    def _json_object(payload: object) -> dict[str, Any]:
        if isinstance(payload, dict):
            return cast(dict[str, Any], payload)
        raise PexelsError("Unexpected Pexels response format")

    @staticmethod
    def _json_list(value: object) -> list[Any]:
        if isinstance(value, list):
            return cast(list[Any], value)
        return []


def _pick_video_file(files: list[Any]) -> str:
    """Choose the best mp4 rendition: the widest one that's still <=
    `_MAX_VIDEO_WIDTH`; if every file is larger, the narrowest available.
    """
    mp4s: list[tuple[int, str]] = []
    for f in files:
        if not isinstance(f, dict):
            continue
        fmap = cast(dict[str, Any], f)
        link = str(fmap.get("link") or "")
        file_type = str(fmap.get("file_type") or "")
        if not link or (file_type and "mp4" not in file_type):
            continue
        width = int(fmap.get("width") or 0)
        mp4s.append((width, link))
    if not mp4s:
        # Fall back to the first link of any type.
        for f in files:
            if isinstance(f, dict):
                link = str(cast(dict[str, Any], f).get("link") or "")
                if link:
                    return link
        return ""
    within = [m for m in mp4s if m[0] <= _MAX_VIDEO_WIDTH]
    if within:
        return max(within, key=lambda m: m[0])[1]
    return min(mp4s, key=lambda m: m[0])[1]


def _ext_from_url(url: str, *, default: str) -> str:
    tail = url.split("?", 1)[0].rsplit(".", 1)
    if len(tail) == 2 and 1 <= len(tail[1]) <= 5 and tail[1].isalnum():
        return tail[1].lower()
    return default

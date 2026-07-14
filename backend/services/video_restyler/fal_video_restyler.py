"""Fal-backed `VideoRestyler` (LTX-2 video-to-video / image-to-video).

Binary inputs (video/image) are uploaded to Fal's CDN first and passed as
hosted URLs — some endpoints (notably Kling motion-control) reject inline
base64 data URIs, and video payloads are large enough that hosting them is
more robust anyway. The result video URL is downloaded and returned as
bytes. These are synchronous `fal.run` calls — short training clips keep
them well within the request timeout.
"""

from __future__ import annotations

import logging
from typing import Any, cast

from services.http_client.http_client import HTTPClient
from services.services_utils import JSONValue
from services.video_restyler.video_restyler import VideoRestylerError

logger = logging.getLogger(__name__)

FAL_API_BASE_URL = "https://fal.run"
# CDN upload host (separate from the model-run host). The SDK uses the same
# initiate -> PUT -> hosted-URL flow under the hood.
FAL_REST_BASE_URL = "https://rest.alpha.fal.ai"
_VIDEO_TO_VIDEO_ENDPOINT = "/fal-ai/ltx-2-19b/distilled/video-to-video"
_IMAGE_TO_VIDEO_ENDPOINT = "/fal-ai/ltx-2-19b/distilled/image-to-video"
# Full (non-distilled) v2v endpoint exposes the first-frame `image_url` +
# `video_strength` knobs we need to anchor an edited frame while keeping the
# original motion. Kling motion-control transfers a reference video's motion
# onto a character image (paired edit-dataset generation).
_MOTION_EDIT_ENDPOINT = "/fal-ai/ltx-2-19b/video-to-video"
_KLING_MOTION_CONTROL_ENDPOINT = "/fal-ai/kling-video/v3/standard/motion-control"
# Kling O3 reference video-to-video (Pro): generate a new shot guided by the
# reference clip (@Video1), preserving its motion/camera style, optionally with
# appearance/style reference images (@Image1...). Reference video must be
# .mp4/.mov, 3-10s, 720-2160px, <=200MB.
_KLING_O3_V2V_REFERENCE_ENDPOINT = "/fal-ai/kling-video/o3/pro/video-to-video/reference"


class FalVideoRestyler:
    def __init__(
        self,
        http: HTTPClient,
        *,
        fal_api_base_url: str = FAL_API_BASE_URL,
        fal_rest_base_url: str = FAL_REST_BASE_URL,
    ) -> None:
        self._http = http
        self._base_url = fal_api_base_url.rstrip("/")
        self._rest_url = fal_rest_base_url.rstrip("/")

    def restyle(self, *, video_bytes: bytes, prompt: str, api_key: str) -> bytes:
        video_url = self._upload(
            data=video_bytes, content_type="video/mp4", file_name="source.mp4", api_key=api_key
        )
        return self._run(
            endpoint=_VIDEO_TO_VIDEO_ENDPOINT,
            payload={"prompt": prompt, "video_url": video_url},
            api_key=api_key,
        )

    def animate(self, *, image_bytes: bytes, prompt: str, api_key: str) -> bytes:
        image_url = self._upload(
            data=image_bytes, content_type="image/png", file_name="frame.png", api_key=api_key
        )
        return self._run(
            endpoint=_IMAGE_TO_VIDEO_ENDPOINT,
            payload={"prompt": prompt, "image_url": image_url},
            api_key=api_key,
        )

    def motion_edit(
        self,
        *,
        video_bytes: bytes,
        image_bytes: bytes,
        prompt: str,
        video_strength: float,
        api_key: str,
    ) -> bytes:
        video_url = self._upload(
            data=video_bytes, content_type="video/mp4", file_name="driver.mp4", api_key=api_key
        )
        image_url = self._upload(
            data=image_bytes, content_type="image/png", file_name="frame.png", api_key=api_key
        )
        return self._run(
            endpoint=_MOTION_EDIT_ENDPOINT,
            payload={
                "prompt": prompt,
                "video_url": video_url,
                "image_url": image_url,
                "video_strength": video_strength,
                "match_video_length": True,
            },
            api_key=api_key,
        )

    def motion_transfer(
        self,
        *,
        image_bytes: bytes,
        video_bytes: bytes,
        prompt: str,
        character_orientation: str,
        api_key: str,
    ) -> bytes:
        image_url = self._upload(
            data=image_bytes, content_type="image/png", file_name="frame.png", api_key=api_key
        )
        video_url = self._upload(
            data=video_bytes, content_type="video/mp4", file_name="driver.mp4", api_key=api_key
        )
        return self._run(
            endpoint=_KLING_MOTION_CONTROL_ENDPOINT,
            payload={
                "prompt": prompt,
                "image_url": image_url,
                "video_url": video_url,
                "character_orientation": character_orientation,
            },
            api_key=api_key,
        )

    def kling_v2v_edit(
        self,
        *,
        video_bytes: bytes,
        image_bytes: bytes | None,
        prompt: str,
        keep_audio: bool,
        api_key: str,
    ) -> bytes:
        video_url = self._upload(
            data=video_bytes, content_type="video/mp4", file_name="source.mp4", api_key=api_key
        )
        payload: dict[str, JSONValue] = {
            "prompt": prompt,
            "video_url": video_url,
            "keep_audio": keep_audio,
        }
        if image_bytes is not None:
            image_url = self._upload(
                data=image_bytes, content_type="image/png", file_name="frame.png", api_key=api_key
            )
            payload["image_urls"] = [image_url]
        return self._run(
            endpoint=_KLING_O3_V2V_REFERENCE_ENDPOINT,
            payload=payload,
            api_key=api_key,
        )

    def _upload(self, *, data: bytes, content_type: str, file_name: str, api_key: str) -> str:
        """Upload bytes to the Fal CDN and return the public hosted URL.

        Two-step initiate -> PUT flow: ask for a presigned upload URL +
        the resulting file URL, then PUT the bytes. Raises
        `VideoRestylerError` on any failure.
        """
        if not api_key:
            raise VideoRestylerError(
                "Add a Fal API key in Settings to use AI video tools.",
                status_code=400,
            )
        logger.info(
            "fal.upload start type=%s name=%s bytes=%d", content_type, file_name, len(data)
        )
        initiate = self._http.post(
            f"{self._rest_url}/storage/upload/initiate?storage_type=fal-cdn-v3",
            headers={
                "Authorization": f"Key {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json_payload={"content_type": content_type, "file_name": file_name},
            timeout=60,
        )
        if initiate.status_code != 200:
            detail = initiate.text[:500] if initiate.text else "Unknown error"
            raise VideoRestylerError(
                f"Fal upload init failed ({initiate.status_code}): {detail}"
            )
        try:
            payload = self._json_object(initiate.json())
            upload_url = str(payload["upload_url"])
            file_url = str(payload["file_url"])
        except (KeyError, VideoRestylerError) as exc:
            raise VideoRestylerError("Unexpected Fal upload response format") from exc

        put_resp = self._http.put(
            upload_url,
            data=data,
            headers={"Content-Type": content_type},
            timeout=600,
        )
        if put_resp.status_code not in (200, 201):
            detail = put_resp.text[:500] if put_resp.text else "Unknown error"
            raise VideoRestylerError(f"Fal upload failed ({put_resp.status_code}): {detail}")
        logger.info("fal.upload ok name=%s url=%s", file_name, file_url)
        return file_url

    def _run(
        self, *, endpoint: str, payload: dict[str, JSONValue], api_key: str
    ) -> bytes:
        if not api_key:
            raise VideoRestylerError(
                "Add a Fal API key in Settings to use AI video tools.",
                status_code=400,
            )
        logger.info("fal.run start endpoint=%s keys=%s", endpoint, sorted(payload.keys()))
        response = self._http.post(
            f"{self._base_url}{endpoint}",
            headers={"Authorization": f"Key {api_key}", "Content-Type": "application/json"},
            json_payload=payload,
            timeout=600,
        )
        if response.status_code != 200:
            detail = response.text[:500] if response.text else "Unknown error"
            logger.error("fal.run failed endpoint=%s status=%d detail=%s", endpoint, response.status_code, detail)
            raise VideoRestylerError(f"Fal video job failed ({response.status_code}): {detail}")

        video_url = self._extract_video_url(self._json_object(response.json()))
        logger.info("fal.run ok endpoint=%s video_url=%s", endpoint, video_url)
        download = self._http.get(video_url, timeout=600)
        if download.status_code != 200 or not download.content:
            raise VideoRestylerError(f"Fal video download failed ({download.status_code})")
        logger.info("fal.run downloaded endpoint=%s bytes=%d", endpoint, len(download.content))
        return download.content

    @staticmethod
    def _extract_video_url(payload: dict[str, Any]) -> str:
        video = payload.get("video")
        if isinstance(video, dict):
            url = cast(dict[str, Any], video).get("url")
            if isinstance(url, str) and url:
                return url
        if isinstance(video, str) and video:
            return video
        for key in ("video_url", "url"):
            url = payload.get(key)
            if isinstance(url, str) and url:
                return url
        raise VideoRestylerError("Fal response missing video url")

    @staticmethod
    def _json_object(payload: object) -> dict[str, Any]:
        if isinstance(payload, dict):
            return cast(dict[str, Any], payload)
        raise VideoRestylerError("Unexpected Fal response format")

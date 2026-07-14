"""Media-extraction handler for external clients (Premiere plugin etc.).

The handler exposes two operations to other clients on the same
machine — pull a single video frame as a PNG, and pull a slice of
audio as a WAV. They exist because the only realistic source of
i2v reference images / a2v audio for a Premiere UXP plugin is "the
clip the user already has selected on the timeline," which the
plugin can resolve to a filesystem path but cannot itself extract
a frame from (UXP runs in a sandbox without `child_process`).

Why this lives in the backend rather than Electron:

  - The backend is the source of truth for queue items; the plugin
    POSTs `imagePath` / `audioPath` to `/api/queue/items` and the
    runner picks them up. Centralising the extract here means the
    plugin doesn't have to round-trip the path through Electron.
  - Electron is no longer the only path: the headless plugin path
    still has access to the backend, but it does *not* assume an
    Electron process is running. Putting extraction in the backend
    keeps "no Electron required" honest.
  - Tests are easier — the FastAPI TestClient pattern works
    out-of-the-box for a backend handler.

Trust model:

  - All routes are localhost-only via FastAPI's CORS allowlist; the
    auth-token middleware also gates non-`/health` calls. Both
    constraints already apply to /api/queue/items, so re-using them
    for /api/media/* doesn't add new attack surface.
  - The handler trusts `source_path` is something the same trusted
    client could already read by other means (the plugin literally
    obtained it from Premiere's project model). We do still validate
    that the path exists and is a regular file — so a client that
    accidentally passes a directory or `/dev/zero` gets a clean 4xx
    rather than a hung ffmpeg subprocess.
  - Output paths are server-controlled — written under
    APP_DATA_DIR/temp/media_extracts/ with a uuid filename so a
    misbehaving client can't trick us into overwriting application
    state.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)

# Sandboxed output directory under APP_DATA_DIR. Lives outside `outputs/`
# so a future "browse my generations" UI doesn't accidentally enumerate
# transient extract files. We stamp this directory on every handler
# instance creation rather than at import time so tests with
# tmp_path-based RuntimeConfig don't collide on a shared module-level
# constant.
_EXTRACTS_SUBDIR = Path("temp") / "media_extracts"

# Defensive ceiling for extracted audio length. The Pydantic schema
# already clamps the *requested* duration but ffmpeg can run past it
# if the source file is shorter (it cuts off naturally) or if the
# request asks for "to end of file" (durationSeconds=0). 600s is
# the largest a2v window we expect on the MLX path; anything beyond
# is a misuse.
_AUDIO_MAX_DURATION_SECONDS = 600.0

# Hard timeout on ffmpeg subprocess. A frame extract from a healthy
# 4K clip takes <2s; a 5-min audio extract finishes in single-digit
# seconds. 60s is generous — beyond that, ffmpeg is wedged and we
# kill rather than let the request hang the FastAPI worker.
_FFMPEG_TIMEOUT_SECONDS = 60.0


class MediaExtractionError(Exception):
    """Raised when ffmpeg fails or the source path is invalid.

    The route layer maps this to an HTTP 422 (client supplied a
    path that didn't yield extractable media) or 500 (ffmpeg blew
    up internally). The exception message is safe to surface to the
    client — it doesn't include arbitrary subprocess stderr; only
    the curated `reason` we set when constructing it.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class _FfmpegResult:
    return_code: int
    stderr_tail: str  # last ~400 chars of stderr, for logging only


class MediaHandler:
    """Frame and audio extraction via the bundled ffmpeg.

    Stateless per-call. We do *not* derive from `StateHandlerBase`
    because no AppState mutation happens here — extraction is a
    pure function of (source_path, requested_time, output_path).
    Keeping the handler stateless also means the FastAPI dependency
    layer can resolve a fresh instance per request without locking.
    """

    def __init__(self, config: RuntimeConfig) -> None:
        self._config = config
        self._extracts_dir = (config.app_data_dir / _EXTRACTS_SUBDIR).resolve()
        self._extracts_dir.mkdir(parents=True, exist_ok=True)

    @property
    def extracts_dir(self) -> Path:
        """Sandboxed directory for extracted files. Public so tests
        can clean up between runs without reaching into private
        attributes."""
        return self._extracts_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_frame(self, source_path: str, time_seconds: float) -> Path:
        """Extract a single frame at `time_seconds` to a PNG file.

        Returns the absolute path of the new PNG. The path is stable
        for the life of the process but we do not promise long-term
        retention — clients should pass it to the queue / consume it
        promptly.

        `time_seconds` past the end of the source raises
        `MediaExtractionError` (422 at the route layer). Format-level
        seek would produce zero frames in that case (ffmpeg returns
        rc=0 but writes nothing); rather than paper over it with a
        decode-from-start fallback (slow on long source clips) we
        push duration-clamping to the caller. The Premiere plugin
        already knows clip duration via the host project model, so
        clamping client-side is a single subtraction.
        """
        source = self._validate_source(source_path)
        output_path = self._allocate_output_path(suffix=".png")

        # `-ss` *before* `-i` makes ffmpeg use the format-level seek
        # which is much faster on long clips (avoids decoding from
        # frame zero up to the seek point). The cost is that the
        # seek can land on a key-frame boundary rather than the exact
        # requested time — for an i2v reference image that's fine
        # (the model doesn't care about a single-frame offset) and
        # the perf win matters because Premiere clips can be hours
        # long when scrubbing source footage.
        cmd = [
            self._ffmpeg_executable(),
            "-loglevel", "error",
            "-ss", f"{max(0.0, time_seconds):.3f}",
            "-i", str(source),
            "-frames:v", "1",
            "-q:v", "2",  # high-quality JPEG q-factor; PNG is lossless but the flag doesn't hurt.
            "-y",
            str(output_path),
        ]
        result = self._run_ffmpeg(cmd)
        if result.return_code != 0 or not output_path.exists():
            self._cleanup_silently(output_path)
            raise MediaExtractionError(
                f"ffmpeg failed to extract frame from {source.name!r} "
                f"at {time_seconds:.3f}s (rc={result.return_code})"
            )
        logger.info(
            "extract_frame ok: %s @ %.3fs -> %s",
            source.name,
            time_seconds,
            output_path.name,
        )
        return output_path

    def extract_audio(
        self,
        source_path: str,
        start_seconds: float,
        duration_seconds: float,
    ) -> Path:
        """Extract a slice of audio to a WAV file (mono, 48kHz).

        `duration_seconds == 0` means "from start to end of file";
        any positive value is honoured up to `_AUDIO_MAX_DURATION_SECONDS`.
        Mono+48kHz matches the format the a2v MLX pipeline expects;
        the dgrauet a2v pipeline accepts WAV and resamples internally
        so we don't have to introspect the source format.
        """
        source = self._validate_source(source_path)
        output_path = self._allocate_output_path(suffix=".wav")
        cmd: list[str] = [
            self._ffmpeg_executable(),
            "-loglevel", "error",
            "-ss", f"{max(0.0, start_seconds):.3f}",
            "-i", str(source),
        ]
        # 0.0 means "to end of file" — leave -t off entirely. Any
        # positive value is bounded by both Pydantic validation
        # (le=300) and the defensive ceiling here, in that order;
        # the second guard is defence-in-depth in case a future
        # schema change loosens the Pydantic constraint.
        if duration_seconds > 0.0:
            bounded = min(duration_seconds, _AUDIO_MAX_DURATION_SECONDS)
            cmd += ["-t", f"{bounded:.3f}"]
        cmd += [
            "-vn",            # drop video stream
            "-ac", "1",       # mono
            "-ar", "48000",   # 48 kHz
            "-y",
            str(output_path),
        ]
        result = self._run_ffmpeg(cmd)
        if result.return_code != 0 or not output_path.exists():
            self._cleanup_silently(output_path)
            raise MediaExtractionError(
                f"ffmpeg failed to extract audio from {source.name!r} "
                f"({start_seconds:.3f}s + {duration_seconds:.3f}s, "
                f"rc={result.return_code})"
            )
        logger.info(
            "extract_audio ok: %s [%.3fs +%.3fs] -> %s",
            source.name,
            start_seconds,
            duration_seconds,
            output_path.name,
        )
        return output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _has_audio_stream(self, source: Path) -> bool:
        """True if `source` has at least one audio stream.

        ffmpeg with no output mapping exits rc=0 and prints stream info to
        stderr; we detect an audio stream by the `Stream #...: Audio` marker.
        Used by `mux_reference_audio` to no-op cleanly when the reference has
        no audio (e.g. a silent screen recording).
        """
        result = self._run_ffmpeg([
            self._ffmpeg_executable(), "-hide_banner", "-i", str(source),
        ])
        # ffmpeg exits 1 for "at least one output file must be specified" but
        # still prints the input streams to stderr, so we scan stderr regardless
        # of return code.
        return ": Audio" in result.stderr_tail or "Audio:" in result.stderr_tail

    def mux_reference_audio(self, video_path: str, reference_path: str) -> Path:
        """Combine a generated (video-only) file with the reference's audio.

        Writes a new mp4 next to `video_path` (same stem, `_audio` suffix) and
        returns it. If the reference has no audio stream, the original video
        path is returned unchanged so the caller can use it as-is. Audio is
        trimmed to the video length via `-shortest` and re-encoded to AAC.
        """
        video = self._validate_source(video_path)
        reference = self._validate_source(reference_path)
        if not self._has_audio_stream(reference):
            logger.info(
                "mux_reference_audio: reference %r has no audio, returning video as-is",
                reference.name,
            )
            return video

        output_path = video.with_name(f"{video.stem}_audio.mp4")
        cmd = [
            self._ffmpeg_executable(),
            "-loglevel", "error",
            "-i", str(video),
            "-i", str(reference),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            "-y",
            str(output_path),
        ]
        result = self._run_ffmpeg(cmd)
        if result.return_code != 0 or not output_path.exists():
            self._cleanup_silently(output_path)
            raise MediaExtractionError(
                f"ffmpeg failed to mux reference audio from {reference.name!r} "
                f"(rc={result.return_code})"
            )
        logger.info(
            "mux_reference_audio ok: video=%s reference=%s -> %s",
            video.name,
            reference.name,
            output_path.name,
        )
        return output_path

    def transcode_video_for_browser(self, video_path: str, output_path: str) -> Path:
        """Write an H.264/yuv420p copy suitable for Electron video playback.

        OpenCV's ``mp4v`` writer is useful for model-conditioning inputs but
        Chromium does not reliably decode MPEG-4 Part 2. Control previews shown
        in GenSpace therefore need a browser-compatible H.264 sibling.
        """
        video = self._validate_source(video_path)
        output = Path(output_path)
        if not output.is_absolute():
            raise MediaExtractionError(
                f"outputPath must be absolute (got {output_path!r})"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self._ffmpeg_executable(),
            "-loglevel", "error",
            "-i", str(video),
            "-map", "0:v:0",
            "-an",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-y",
            str(output),
        ]
        result = self._run_ffmpeg(cmd)
        if result.return_code != 0 or not output.exists():
            self._cleanup_silently(output)
            raise MediaExtractionError(
                f"ffmpeg failed to transcode {video.name!r} for browser playback "
                f"(rc={result.return_code})"
            )
        logger.info(
            "transcode_video_for_browser ok: %s -> %s",
            video.name,
            output.name,
        )
        return output

    def downscale_video(self, video_path: str, width: int, height: int) -> Path:
        """Re-encode ``video_path`` to (width, height) via ffmpeg.

        Writes a new mp4 (same stem, ``_scaled`` suffix) beside the source and
        returns it, preserving any audio stream. Used to reach exact IC-LoRA
        output resolutions (e.g. 720p) that are downscaled from the x2
        upsampler's 1920x1152 result.
        """
        video = self._validate_source(video_path)
        output_path = video.with_name(f"{video.stem}_scaled.mp4")
        cmd = [
            self._ffmpeg_executable(),
            "-loglevel", "error",
            "-i", str(video),
            "-vf", f"scale={int(width)}:{int(height)}:flags=lanczos",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-movflags", "+faststart",
            "-y",
            str(output_path),
        ]
        result = self._run_ffmpeg(cmd)
        if result.return_code != 0 or not output_path.exists():
            self._cleanup_silently(output_path)
            raise MediaExtractionError(
                f"ffmpeg failed to downscale {video.name!r} to {width}x{height} "
                f"(rc={result.return_code})"
            )
        logger.info(
            "downscale_video ok: %s -> %s (%dx%d)",
            video.name, output_path.name, width, height,
        )
        return output_path

    def _validate_source(self, source_path: str) -> Path:
        if not source_path:
            raise MediaExtractionError("sourcePath is empty")
        # Resolve to an absolute path so we can compare meaningfully,
        # but don't canonicalise via realpath() — that would break
        # symlinked media libraries (a common pro-video workflow).
        path = Path(source_path)
        if not path.is_absolute():
            raise MediaExtractionError(
                f"sourcePath must be absolute (got {source_path!r})"
            )
        if not path.exists():
            raise MediaExtractionError(f"sourcePath does not exist: {source_path}")
        if not path.is_file():
            raise MediaExtractionError(
                f"sourcePath is not a regular file: {source_path}"
            )
        return path

    def _allocate_output_path(self, *, suffix: str) -> Path:
        # Stamp+uuid filename so two simultaneous extracts (e.g. user
        # rapidly drags multiple clips) can't clobber each other. We
        # don't bother with a per-source content hash — the cost of
        # an occasional duplicate is just a few MB of disk.
        stamp = time.strftime("%Y%m%d_%H%M%S")
        token = uuid.uuid4().hex[:8]
        return self._extracts_dir / f"extract_{stamp}_{token}{suffix}"

    def _ffmpeg_executable(self) -> str:
        # imageio-ffmpeg ships a portable ffmpeg binary on every
        # platform we support. Resolving via the package keeps us
        # working on machines that don't have a system ffmpeg
        # installed (most non-developer Macs and Windows boxes).
        try:
            import imageio_ffmpeg  # type: ignore[import-not-found]
        except ImportError as exc:
            raise MediaExtractionError(
                "ffmpeg unavailable: imageio-ffmpeg is not installed in the "
                "backend Python environment. Re-run `pnpm setup:dev`."
            ) from exc
        try:
            return str(imageio_ffmpeg.get_ffmpeg_exe())  # type: ignore[no-any-return]
        except Exception as exc:
            raise MediaExtractionError(
                f"ffmpeg unavailable: imageio-ffmpeg could not locate its "
                f"bundled binary ({exc})"
            ) from exc

    def _run_ffmpeg(self, cmd: list[str]) -> _FfmpegResult:
        # Capture stderr so we can include a small tail in error logs
        # when ffmpeg fails. We keep stdout discarded — ffmpeg writes
        # progress to stderr by default and there's nothing useful on
        # stdout for our use cases (-loglevel error already silences
        # the verbose stuff).
        cmd_str = " ".join(shlex.quote(c) for c in cmd)
        logger.debug("ffmpeg cmd: %s", cmd_str)
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                timeout=_FFMPEG_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            logger.warning(
                "ffmpeg timed out after %.1fs: %s",
                _FFMPEG_TIMEOUT_SECONDS,
                cmd_str,
            )
            raise MediaExtractionError(
                f"ffmpeg timed out after {_FFMPEG_TIMEOUT_SECONDS:.0f}s"
            ) from exc
        except OSError as exc:
            # FileNotFoundError if the resolved path is wrong, PermissionError
            # if AV-scan blocked it on Windows, etc. Treat all OSErrors
            # the same — the user-facing fix is "reinstall the backend
            # python env"; the developer-facing fix lives in the log.
            logger.warning(
                "ffmpeg subprocess could not start: %s (cmd=%s)",
                exc,
                cmd_str,
            )
            raise MediaExtractionError(
                f"ffmpeg subprocess could not start: {exc}"
            ) from exc

        stderr_text = completed.stderr.decode("utf-8", errors="replace")
        # Trim the stderr to a tail so a runaway error log doesn't bloat
        # the structured log lines downstream. 400 chars catches the
        # last 4-6 ffmpeg complaint lines, which is plenty for triage.
        stderr_tail = stderr_text[-400:] if stderr_text else ""
        if completed.returncode != 0:
            logger.warning(
                "ffmpeg rc=%d stderr_tail=%r cmd=%s",
                completed.returncode,
                stderr_tail,
                cmd_str,
            )
        return _FfmpegResult(
            return_code=completed.returncode,
            stderr_tail=stderr_tail,
        )

    def _cleanup_silently(self, path: Path) -> None:
        # On any failure we delete the half-written output file so
        # subsequent calls don't accidentally hand the client a zero-
        # byte PNG. Best-effort — if delete itself fails we just
        # carry on; the file will be cleaned up by a future extracts-
        # dir GC pass.
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass

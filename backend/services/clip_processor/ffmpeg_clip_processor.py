"""ffmpeg-backed `ClipProcessor` implementation.

Uses the portable ffmpeg binary shipped by ``imageio-ffmpeg`` (same one
`MediaHandler` uses) so probing works on machines without a system
ffmpeg. We parse the human-readable banner ffmpeg prints to stderr for
`-i <file>` rather than depending on ffprobe (imageio-ffmpeg does not
bundle ffprobe). This mirrors the Electron `getVideoDimensions` parser.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

from services.clip_processor.clip_processor import (
    ClipProbeResult,
    ClipProcessorError,
    EditPlan,
    SceneSpan,
)

logger = logging.getLogger(__name__)

_PROBE_TIMEOUT_SECONDS = 30.0
# Render/scene-detect re-encode or fully decode the clip, so they get a
# generous ceiling. Dataset clips are short (seconds) so even a slow
# machine finishes well inside this.
_RENDER_TIMEOUT_SECONDS = 300.0
# Drop spans shorter than this when scene-splitting — sub-second segments
# are almost always false cuts (a flash/cut transition), not usable clips.
_MIN_SCENE_SECONDS = 1.0
# `showinfo` prints one line per selected frame: `... pts_time:12.34 ...`
_PTS_TIME_RE = re.compile(r"pts_time:([\d.]+)")

# `Duration: 00:01:23.45, start: ...`
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
# `Stream #0:0(und): Video: h264 (High) (avc1 / 0x...), yuv420p, 1920x1080 ...`
_VIDEO_LINE_RE = re.compile(r"Stream #\d+:\d+.*: Video:\s*(?P<codec>[\w]+)")
_DIMENSIONS_RE = re.compile(r"(?<!\d)(\d{2,5})x(\d{2,5})(?!\d)")
_FPS_RE = re.compile(r"([\d.]+)\s*fps")
_AUDIO_LINE_RE = re.compile(r"Stream #\d+:\d+.*: Audio:")

# Still-image inputs the trainer treats as a single frame. We loop them into a
# full N-frame clip during normalization so an image used as an IC-LoRA
# reference satisfies the "reference matches the target's length" requirement.
_IMAGE_SUFFIXES: frozenset[str] = frozenset({".png", ".jpg", ".jpeg"})


def _atempo_chain(speed: float) -> list[str]:
    """Decompose a tempo multiplier into `atempo` filters.

    ffmpeg's `atempo` only accepts 0.5–2.0 per instance, so larger/smaller
    factors are chained (e.g. 4× → atempo=2.0,atempo=2.0).
    """
    chain: list[str] = []
    remaining = speed
    while remaining > 2.0:
        chain.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        chain.append("atempo=0.5")
        remaining /= 0.5
    chain.append(f"atempo={remaining:g}")
    return chain


class FfmpegClipProcessor:
    def probe(self, *, video_path: str) -> ClipProbeResult:
        path = Path(video_path)
        if not video_path:
            raise ClipProcessorError("videoPath is empty", status_code=400)
        if not path.is_file():
            raise ClipProcessorError(f"Clip not found: {video_path}", status_code=400)

        stderr = self._run_probe(path)
        duration = self._parse_duration(stderr)
        video_line = self._find_video_line(stderr)
        if video_line is None:
            raise ClipProcessorError(
                f"No video stream found in {path.name!r}", status_code=422
            )
        width, height = self._parse_dimensions(video_line, path.name)
        fps = self._parse_fps(video_line)
        codec = self._parse_codec(video_line)
        has_audio = bool(_AUDIO_LINE_RE.search(stderr))
        frame_count = int(round(duration * fps)) if duration > 0 and fps > 0 else 0

        return ClipProbeResult(
            duration_seconds=duration,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            has_audio=has_audio,
            video_codec=codec,
        )

    def render(self, *, source_path: str, plan: EditPlan, out_path: str) -> None:
        source = Path(source_path)
        if not source.is_file():
            raise ClipProcessorError(f"Clip not found: {source_path}", status_code=400)
        if plan.is_empty:
            raise ClipProcessorError("No edits to apply", status_code=400)

        cmd: list[str] = [self._ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-y"]
        # `-ss`/`-t` before `-i` for a fast format-level seek; `-t` is the
        # segment duration so the meaning is unambiguous regardless of seek.
        if plan.trim is not None:
            start = max(0.0, plan.trim.start_seconds)
            duration = max(0.0, plan.trim.end_seconds - plan.trim.start_seconds)
            if duration <= 0:
                raise ClipProcessorError("Trim end must be after start", status_code=400)
            cmd += ["-ss", f"{start:.3f}", "-t", f"{duration:.3f}"]
        cmd += ["-i", str(source)]

        video_filters = self._video_filters(plan)
        if video_filters:
            cmd += ["-vf", ",".join(video_filters)]

        audio_filters = self._audio_filters(plan)
        # Re-encode H.264 + AAC; veryfast/crf=18 is visually lossless for prep.
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]
        if plan.mute:
            cmd += ["-an"]
        else:
            # `-c:a aac` is a no-op when the source has no audio stream.
            if audio_filters:
                cmd += ["-af", ",".join(audio_filters)]
            cmd += ["-c:a", "aac"]
        cmd += [str(out_path)]

        completed = self._run(cmd, timeout=_RENDER_TIMEOUT_SECONDS)
        if completed.returncode != 0 or not Path(out_path).exists():
            tail = completed.stderr.decode("utf-8", errors="replace")[-300:]
            raise ClipProcessorError(
                f"ffmpeg failed to render edited clip (rc={completed.returncode}): {tail}",
                status_code=500,
            )

    def _video_filters(self, plan: EditPlan) -> list[str]:
        filters: list[str] = []
        crop = plan.crop
        if crop is not None:
            if crop.width <= 0 or crop.height <= 0:
                raise ClipProcessorError("Crop width/height must be positive", status_code=400)
            filters.append(f"crop={crop.width}:{crop.height}:{crop.x}:{crop.y}")
        scale = plan.scale
        if scale is not None:
            if scale.width <= 0 or scale.height <= 0:
                raise ClipProcessorError("Scale width/height must be positive", status_code=400)
            filters.append(f"scale={scale.width}:{scale.height}")
        if plan.fps is not None:
            if plan.fps <= 0:
                raise ClipProcessorError("fps must be positive", status_code=400)
            filters.append(f"fps={plan.fps:g}")
        if plan.reverse:
            filters.append("reverse")
        if plan.speed is not None and plan.speed != 1.0:
            if plan.speed <= 0:
                raise ClipProcessorError("speed must be positive", status_code=400)
            filters.append(f"setpts=PTS/{plan.speed:g}")
        return filters

    def _audio_filters(self, plan: EditPlan) -> list[str]:
        filters: list[str] = []
        if plan.reverse:
            filters.append("areverse")
        if plan.speed is not None and plan.speed != 1.0:
            filters.extend(_atempo_chain(plan.speed))
        return filters

    def normalize_for_training(
        self,
        *,
        source_path: str,
        out_path: str,
        fps: float,
        short_side: int,
        frames: int,
        max_duration_seconds: float | None = None,
        exact_width: int | None = None,
        exact_height: int | None = None,
    ) -> None:
        source = Path(source_path)
        if not source.is_file():
            raise ClipProcessorError(f"Clip not found: {source_path}", status_code=400)
        if fps <= 0 or short_side <= 0 or frames <= 0:
            raise ClipProcessorError(
                "fps, short_side and frames must be positive", status_code=400
            )

        cmd: list[str] = [
            self._ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-y"
        ]
        is_image = source.suffix.lower() in _IMAGE_SUFFIXES
        if is_image:
            # A still image has no inherent duration, so `-frames:v` alone would
            # emit a single frame. `-loop 1` repeats it indefinitely and the
            # `-frames:v N` cap below trims it to exactly the bucket's length,
            # producing a static reference clip the trainer can pair with the
            # target. (`-t` is meaningless for a single image; skip it.)
            cmd += ["-loop", "1"]
        elif max_duration_seconds is not None and max_duration_seconds > 0:
            cmd += ["-t", f"{max_duration_seconds:.3f}"]
        cmd += ["-i", str(source)]

        if exact_width is not None and exact_height is not None:
            if exact_width <= 0 or exact_height <= 0:
                raise ClipProcessorError(
                    "exact_width and exact_height must be positive", status_code=400
                )
            # Force the precise resolution so both clips of a pair match exactly
            # (even dims required by yuv420p). `setsar=1` drops any non-square
            # pixel aspect so the file's W×H is the displayed W×H.
            ew = exact_width - (exact_width % 2)
            eh = exact_height - (exact_height % 2)
            scale = f"scale={ew}:{eh},setsar=1"
        else:
            # `iw`/`ih` here are post-autorotate (ffmpeg inserts the rotate filter
            # before -vf), so the short-side math is correct for portrait clips.
            # `min(...)` clamps to the source size so we never upscale; `-2` keeps
            # aspect with an even dimension.
            scale = (
                f"scale=w='if(gt(iw,ih),-2,min(iw,{short_side}))'"
                f":h='if(gt(iw,ih),min(ih,{short_side}),-2)'"
            )
        cmd += ["-vf", f"{scale},fps={fps:g}"]
        cmd += ["-frames:v", str(frames)]
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]
        # Bake rotation into pixels then drop the side-data so W×H == display.
        cmd += ["-metadata:s:v:0", "rotate=0", "-an", str(out_path)]

        completed = self._run(cmd, timeout=_RENDER_TIMEOUT_SECONDS)
        if completed.returncode != 0 or not Path(out_path).exists():
            tail = completed.stderr.decode("utf-8", errors="replace")[-300:]
            raise ClipProcessorError(
                f"ffmpeg failed to normalize clip (rc={completed.returncode}): {tail}",
                status_code=500,
            )

    def detect_scenes(self, *, video_path: str, threshold: float) -> list[SceneSpan]:
        source = Path(video_path)
        if not source.is_file():
            raise ClipProcessorError(f"Clip not found: {video_path}", status_code=400)
        duration = self.probe(video_path=video_path).duration_seconds
        if duration <= 0:
            return []

        bounded = min(0.9, max(0.1, threshold))
        cmd = [
            self._ffmpeg_executable(),
            "-hide_banner",
            "-i", str(source),
            "-filter:v", f"select='gt(scene,{bounded:.3f})',showinfo",
            "-f", "null", "-",
        ]
        completed = self._run(cmd, timeout=_RENDER_TIMEOUT_SECONDS)
        stderr = completed.stderr.decode("utf-8", errors="replace")
        cuts = sorted(
            t for t in (float(m) for m in _PTS_TIME_RE.findall(stderr)) if 0.0 < t < duration
        )

        boundaries = [0.0, *cuts, duration]
        spans: list[SceneSpan] = []
        for start, end in zip(boundaries, boundaries[1:]):
            if end - start >= _MIN_SCENE_SECONDS:
                spans.append(SceneSpan(start_seconds=start, end_seconds=end))
        # Always hand back at least the whole clip so the caller can import it.
        if not spans:
            spans.append(SceneSpan(start_seconds=0.0, end_seconds=duration))
        return spans

    def extract_frame(self, *, video_path: str, time_seconds: float) -> bytes:
        source = Path(video_path)
        if not source.is_file():
            raise ClipProcessorError(f"Clip not found: {video_path}", status_code=400)
        cmd = [
            self._ffmpeg_executable(),
            "-hide_banner",
            "-loglevel", "error",
            "-ss", f"{max(0.0, time_seconds):.3f}",
            "-i", str(source),
            "-frames:v", "1",
            "-f", "image2pipe",
            "-vcodec", "png",
            "-",
        ]
        completed = self._run(cmd, timeout=_PROBE_TIMEOUT_SECONDS)
        if completed.returncode != 0 or not completed.stdout:
            tail = completed.stderr.decode("utf-8", errors="replace")[-300:]
            raise ClipProcessorError(
                f"ffmpeg failed to extract frame (rc={completed.returncode}): {tail}",
                status_code=500,
            )
        return completed.stdout

    def generate_sprite(
        self, *, video_path: str, out_path: str, tile_count: int, tile_width: int
    ) -> int:
        source = Path(video_path)
        if not source.is_file():
            raise ClipProcessorError(f"Clip not found: {video_path}", status_code=400)
        tiles = max(1, tile_count)
        width = max(16, tile_width)
        frame_count = self.probe(video_path=video_path).frame_count
        # Pick every `step`-th decoded frame so the strip spans the whole
        # clip. When the frame count is unknown/short we fall back to every
        # frame; `tile=Nx1` simply leaves trailing cells black.
        step = max(1, frame_count // tiles) if frame_count > 0 else 1
        vf = (
            f"select='not(mod(n\\,{step}))',"
            f"scale={width}:-2,"
            f"tile={tiles}x1"
        )
        cmd = [
            self._ffmpeg_executable(),
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", str(source),
            "-frames:v", "1",
            "-vf", vf,
            "-an",
            "-q:v", "4",
            str(out_path),
        ]
        completed = self._run(cmd, timeout=_RENDER_TIMEOUT_SECONDS)
        if completed.returncode != 0 or not Path(out_path).exists():
            tail = completed.stderr.decode("utf-8", errors="replace")[-300:]
            raise ClipProcessorError(
                f"ffmpeg failed to build sprite (rc={completed.returncode}): {tail}",
                status_code=500,
            )
        return tiles

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run(self, cmd: list[str], *, timeout: float) -> subprocess.CompletedProcess[bytes]:
        try:
            return subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            raise ClipProcessorError("ffmpeg timed out", status_code=504) from exc
        except OSError as exc:
            raise ClipProcessorError(f"ffmpeg could not start: {exc}", status_code=500) from exc

    def _run_probe(self, path: Path) -> str:
        # `ffmpeg -i <file>` with no output writes the stream banner to
        # stderr and exits non-zero ("At least one output file must be
        # specified") — that's expected; we only want the banner.
        cmd = [self._ffmpeg_executable(), "-hide_banner", "-i", str(path)]
        completed = self._run(cmd, timeout=_PROBE_TIMEOUT_SECONDS)
        return completed.stderr.decode("utf-8", errors="replace")

    def _ffmpeg_executable(self) -> str:
        try:
            import imageio_ffmpeg  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ClipProcessorError(
                "ffmpeg unavailable: imageio-ffmpeg is not installed in the "
                "backend Python environment. Re-run `pnpm setup:dev`.",
                status_code=500,
            ) from exc
        try:
            return str(imageio_ffmpeg.get_ffmpeg_exe())  # type: ignore[no-any-return]
        except Exception as exc:
            raise ClipProcessorError(
                f"ffmpeg unavailable: {exc}", status_code=500
            ) from exc

    def _parse_duration(self, stderr: str) -> float:
        match = _DURATION_RE.search(stderr)
        if match is None:
            return 0.0
        hours, minutes, seconds = match.groups()
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    def _find_video_line(self, stderr: str) -> str | None:
        for line in stderr.splitlines():
            if _VIDEO_LINE_RE.search(line):
                return line
        return None

    def _parse_dimensions(self, video_line: str, name: str) -> tuple[int, int]:
        match = _DIMENSIONS_RE.search(video_line)
        if match is None:
            raise ClipProcessorError(
                f"Could not determine resolution of {name!r}", status_code=422
            )
        return int(match.group(1)), int(match.group(2))

    def _parse_fps(self, video_line: str) -> float:
        match = _FPS_RE.search(video_line)
        if match is None:
            return 0.0
        try:
            return float(match.group(1))
        except ValueError:
            return 0.0

    def _parse_codec(self, video_line: str) -> str | None:
        match = _VIDEO_LINE_RE.search(video_line)
        return match.group("codec") if match else None

"""Test doubles for backend side-effect services."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image
from api_types import ImageConditioningInput, VideoCameraMotion
from services.interfaces import (
    ClipProbeResult,
    ClipProcessorError,
    EditPlan,
    ImageEditorError,
    LoraPromptProfile,
    LoraPromptProfileResult,
    PexelsError,
    PexelsMediaResult,
    PexelsSearchResult,
    SceneSpan,
    VideoCaptionerError,
    VideoInfoPayload,
    VideoRestylerError,
)
from services.ltx_api_client.ltx_api_client import LTXRetakeResult
from services.trainer_target.trainer_target import (
    AccountInfo,
    GpuOffer,
    GpuTelemetry,
    NetworkVolume,
    PodInfo,
    RemoteCommandStatus,
    TrainerCredentials,
    ValidationArtifact,
)
from state.lora_training_state import TargetHandle
from tests.fakes.fake_gpu_info import FakeGpuInfo


@dataclass
class FakeResponse:
    status_code: int = 200
    text: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    content: bytes = b""
    json_payload: Any = field(default_factory=dict)

    def json(self) -> Any:
        return self.json_payload


@dataclass
class HttpCall:
    method: str
    url: str
    headers: dict[str, str] | None
    json_payload: dict[str, Any] | None
    data: Any
    timeout: int


class FakeHTTPClient:
    def __init__(self) -> None:
        self.calls: list[HttpCall] = []
        self._queues: dict[str, list[FakeResponse | Exception]] = {
            "post": [],
            "get": [],
            "put": [],
        }

    def queue(self, method: str, *items: FakeResponse | Exception) -> None:
        self._queues[method].extend(items)

    def _dequeue(self, method: str) -> FakeResponse:
        queue = self._queues[method]
        if not queue:
            raise RuntimeError(f"No queued {method.upper()} response")
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def post(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        json_payload: dict[str, Any] | None = None,
        data: Any = None,
        timeout: int = 30,
    ) -> FakeResponse:
        self.calls.append(HttpCall("post", url, headers, json_payload, data, timeout))
        return self._dequeue("post")

    def get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: int = 30,
        allow_redirects: bool = True,
    ) -> FakeResponse:
        del allow_redirects
        self.calls.append(HttpCall("get", url, headers, None, None, timeout))
        return self._dequeue("get")

    def put(
        self,
        url: str,
        data: Any = None,
        headers: dict[str, str] | None = None,
        timeout: int = 300,
    ) -> FakeResponse:
        self.calls.append(HttpCall("put", url, headers, None, data, timeout))
        return self._dequeue("put")


class FakeTaskRunner:
    def __init__(self) -> None:
        self.jobs_run = 0
        self.last_task_name: str | None = None
        self.errors: list[Exception] = []

    def run_background(
        self,
        target,
        *,
        task_name: str,
        on_error=None,
        daemon: bool = True,
    ) -> None:  # noqa: ARG002
        self.jobs_run += 1
        self.last_task_name = task_name
        try:
            target()
        except Exception as exc:
            self.errors.append(exc)
            if on_error is not None:
                on_error(exc)


class FakeLTXAPIClient:
    def __init__(self) -> None:
        self.upload_file_calls: list[dict[str, Any]] = []
        self.text_to_video_calls: list[dict[str, Any]] = []
        self.image_to_video_calls: list[dict[str, Any]] = []
        self.audio_to_video_calls: list[dict[str, Any]] = []
        self.retake_calls: list[dict[str, Any]] = []
        self.raise_on_upload_file: Exception | None = None
        self.raise_on_text_to_video: Exception | None = None
        self.raise_on_image_to_video: Exception | None = None
        self.raise_on_audio_to_video: Exception | None = None
        self.raise_on_retake: Exception | None = None
        self.text_to_video_result = b"fake-ltx-api-t2v-video"
        self.image_to_video_result = b"fake-ltx-api-i2v-video"
        self.audio_to_video_result = b"fake-ltx-api-a2v-video"
        self.retake_result = LTXRetakeResult(video_bytes=b"fake-ltx-api-retake-video", result_payload=None)
        self.upload_file_results: dict[str, str] = {}

    def upload_file(
        self,
        *,
        api_key: str,
        file_path: str,
    ) -> str:
        self.upload_file_calls.append(
            {
                "api_key": api_key,
                "file_path": file_path,
            }
        )
        if self.raise_on_upload_file is not None:
            raise self.raise_on_upload_file
        default_uri = f"storage://uploaded/{Path(file_path).name}"
        return self.upload_file_results.get(file_path, default_uri)

    def generate_text_to_video(
        self,
        *,
        api_key: str,
        prompt: str,
        model: str,
        resolution: str,
        duration: float,
        fps: float,
        generate_audio: bool,
        camera_motion: VideoCameraMotion = "none",
    ) -> bytes:
        self.text_to_video_calls.append(
            {
                "api_key": api_key,
                "prompt": prompt,
                "model": model,
                "resolution": resolution,
                "duration": duration,
                "fps": fps,
                "generate_audio": generate_audio,
                "camera_motion": camera_motion,
            }
        )
        if self.raise_on_text_to_video is not None:
            raise self.raise_on_text_to_video
        return self.text_to_video_result

    def generate_image_to_video(
        self,
        *,
        api_key: str,
        prompt: str,
        image_uri: str,
        model: str,
        resolution: str,
        duration: float,
        fps: float,
        generate_audio: bool,
        camera_motion: VideoCameraMotion = "none",
    ) -> bytes:
        self.image_to_video_calls.append(
            {
                "api_key": api_key,
                "prompt": prompt,
                "image_uri": image_uri,
                "model": model,
                "resolution": resolution,
                "duration": duration,
                "fps": fps,
                "generate_audio": generate_audio,
                "camera_motion": camera_motion,
            }
        )
        if self.raise_on_image_to_video is not None:
            raise self.raise_on_image_to_video
        return self.image_to_video_result

    def generate_audio_to_video(
        self,
        *,
        api_key: str,
        prompt: str,
        audio_uri: str,
        image_uri: str | None,
        model: str,
        resolution: str,
    ) -> bytes:
        self.audio_to_video_calls.append(
            {
                "api_key": api_key,
                "prompt": prompt,
                "audio_uri": audio_uri,
                "image_uri": image_uri,
                "model": model,
                "resolution": resolution,
            }
        )
        if self.raise_on_audio_to_video is not None:
            raise self.raise_on_audio_to_video
        return self.audio_to_video_result

    def retake(
        self,
        *,
        api_key: str,
        video_path: str,
        start_time: float,
        duration: float,
        prompt: str,
        mode: str,
    ) -> LTXRetakeResult:
        self.retake_calls.append(
            {
                "api_key": api_key,
                "video_path": video_path,
                "start_time": start_time,
                "duration": duration,
                "prompt": prompt,
                "mode": mode,
            }
        )
        if self.raise_on_retake is not None:
            raise self.raise_on_retake
        return self.retake_result


class FakeZitAPIClient:
    def __init__(self) -> None:
        self.configured = True
        self.text_to_image_calls: list[dict[str, Any]] = []
        self.raise_on_text_to_image: Exception | None = None
        self.text_to_image_result = b"fake-zit-api-image"

    def is_configured(self) -> bool:
        return self.configured

    def generate_text_to_image(
        self,
        *,
        api_key: str,
        prompt: str,
        width: int,
        height: int,
        seed: int,
        num_inference_steps: int,
    ) -> bytes:
        self.text_to_image_calls.append(
            {
                "api_key": api_key,
                "prompt": prompt,
                "width": width,
                "height": height,
                "seed": seed,
                "num_inference_steps": num_inference_steps,
            }
        )
        if self.raise_on_text_to_image is not None:
            raise self.raise_on_text_to_image
        return self.text_to_image_result


class FakeModelDownloader:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_next: Exception | None = None

    def _raise_if_needed(self) -> None:
        if self.fail_next is None:
            return
        error = self.fail_next
        self.fail_next = None
        raise error

    def download_file(
        self,
        repo_id: str,
        filename: str,
        local_dir: str,
        token: str | None,
        on_progress: Callable[[int], None] | None = None,
    ) -> Path:
        self._raise_if_needed()
        self.calls.append({"kind": "file", "repo_id": repo_id, "filename": filename, "local_dir": local_dir, "on_progress": on_progress})

        if on_progress is not None:
            on_progress(512)
            on_progress(1024)

        destination = Path(local_dir) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"\x00" * 1024)
        return destination

    def download_snapshot(
        self,
        repo_id: str,
        local_dir: str,
        token: str | None,
        on_progress: Callable[[int], None] | None = None,
    ) -> Path:
        self._raise_if_needed()
        self.calls.append(
            {
                "kind": "snapshot",
                "repo_id": repo_id,
                "local_dir": local_dir,
                "on_progress": on_progress,
            }
        )

        if on_progress is not None:
            on_progress(512)
            on_progress(1024)

        root = Path(local_dir)
        root.mkdir(parents=True, exist_ok=True)

        (root / "model.safetensors").write_bytes(b"\x00" * 1024)

        return root


class FakeGpuCleaner:
    def __init__(self) -> None:
        self.cleanup_calls = 0

    def cleanup(self) -> None:
        self.cleanup_calls += 1


class FakeCapture:
    def __init__(
        self,
        frames: list[Any] | None = None,
        *,
        fps: float = 24,
        width: int = 64,
        height: int = 64,
        opened: bool = True,
    ) -> None:
        self.frames = list(frames) if frames is not None else ["frame-0", "frame-1", "frame-2"]
        self.fps = fps
        self.width = width
        self.height = height
        self.opened = opened
        self.position = 0
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802
        return self.opened

    def release(self) -> None:
        self.released = True


class FakeWriter:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.frames: list[Any] = []
        self.released = False

    def write(self, frame: Any) -> None:
        self.frames.append(frame)

    def release(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(b"writer-output")
        self.released = True


class FakeVideoProcessor:
    def __init__(self) -> None:
        self.videos: dict[str, FakeCapture] = {}
        self.writers: list[FakeWriter] = []
        self.open_video_calls: list[str] = []
        self.resize_calls: list[tuple[int, int]] = []

    def register_video(self, path: str, capture: FakeCapture) -> None:
        self.videos[path] = capture

    def open_video(self, path: str) -> FakeCapture:
        self.open_video_calls.append(path)
        return self.videos.setdefault(path, FakeCapture())

    def get_video_info(self, cap: FakeCapture) -> VideoInfoPayload:
        return {
            "fps": cap.fps,
            "frame_count": len(cap.frames),
            "width": cap.width,
            "height": cap.height,
        }

    def read_frame(self, cap: FakeCapture, frame_idx: int | None = None) -> Any | None:
        if frame_idx is not None:
            cap.position = frame_idx
        if cap.position >= len(cap.frames):
            return None
        frame = cap.frames[cap.position]
        cap.position += 1
        return frame

    def resize_frame(self, frame: Any, size: tuple[int, int]) -> Any:
        # Frames are opaque string tokens in the fake; record the requested
        # size so tests can assert the reference was downscaled toward the
        # generation target, and return the frame tagged with its new size.
        self.resize_calls.append(size)
        return f"resized{size}:{frame}"

    def apply_canny(self, frame: Any) -> Any:
        return f"canny:{frame}"

    def apply_depth(self, frame: Any, depth_pipeline: Any) -> Any:
        return depth_pipeline.apply(frame)

    def apply_pose(self, frame: Any, pose_pipeline: Any) -> Any:
        return pose_pipeline.apply(frame)

    def encode_frame_jpeg(self, frame: Any, quality: int = 85) -> bytes:  # noqa: ARG002
        return f"jpeg:{frame}".encode("utf-8")

    def create_writer(self, path: str, fourcc: str, fps: float, size: tuple[int, int]) -> FakeWriter:  # noqa: ARG002
        writer = FakeWriter(path)
        self.writers.append(writer)
        return writer

    def release(self, cap_or_writer: FakeCapture | FakeWriter) -> None:
        cap_or_writer.release()


class _FakeVideoPipelineBase:
    pipeline_kind: str

    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []
        self.warmup_calls: list[dict[str, Any]] = []
        self.compile_calls = 0
        self.create_calls = 0
        self.raise_on_generate: Exception | None = None

    def _record_generate(self, payload: dict[str, Any]) -> None:
        self.generate_calls.append(payload)
        if self.raise_on_generate is not None:
            raise self.raise_on_generate

        output_path = Path(payload["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-video")

    def warmup(self, output_path: str) -> None:
        self.warmup_calls.append({"output_path": output_path})
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"warmup")
        path.unlink(missing_ok=True)

    def compile_transformer(self) -> None:
        self.compile_calls += 1


class FakeFastVideoPipeline(_FakeVideoPipelineBase):
    pipeline_kind = "fast"
    _singleton: ClassVar["FakeFastVideoPipeline | None"] = None

    # Set by create() each time the pipeline is (re)built — lets tests assert
    # that swapping standard LoRAs rebuilds the GpuSlot with the new adapter.
    last_lora_path: str | None
    last_lora_scale: float

    @classmethod
    def bind_singleton(cls, pipeline: "FakeFastVideoPipeline") -> None:
        cls._singleton = pipeline

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: str | object,
        streaming_prefetch_count: int | None,
        lora_path: str | None = None,
        lora_scale: float = 1.0,
    ) -> "FakeFastVideoPipeline":
        del checkpoint_path, gemma_root, upsampler_path, device, streaming_prefetch_count
        pipeline = FakeFastVideoPipeline._singleton
        if pipeline is None:
            raise RuntimeError("FakeFastVideoPipeline singleton is not bound")
        # Record which adapter this pipeline was built with so tests can assert
        # that switching standard LoRAs swaps the cached GpuSlot.
        pipeline.last_lora_path = lora_path
        pipeline.last_lora_scale = lora_scale
        pipeline.create_calls += 1
        return pipeline

    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        output_path: str,
    ) -> None:
        self._record_generate(
            {
                "prompt": prompt,
                "seed": seed,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "frame_rate": frame_rate,
                "images": images,
                "output_path": output_path,
            }
        )


class FakeZitOutput:
    def __init__(self, color: str = "red") -> None:
        self.images = [Image.new("RGB", (32, 32), color)]


class FakeImageGenerationPipeline:
    _singleton: ClassVar["FakeImageGenerationPipeline | None"] = None

    @classmethod
    def bind_singleton(cls, pipeline: "FakeImageGenerationPipeline") -> None:
        cls._singleton = pipeline

    @staticmethod
    def create(
        model_path: str,
        device: str | None = None,
    ) -> "FakeImageGenerationPipeline":
        del model_path
        pipeline = FakeImageGenerationPipeline._singleton
        if pipeline is None:
            raise RuntimeError("FakeImageGenerationPipeline singleton is not bound")
        if device is not None:
            pipeline.to(device)
        return pipeline

    def __init__(self) -> None:
        self.device: str | None = None
        self.generate_calls: list[dict[str, Any]] = []
        self.raise_on_generate: Exception | None = None

    def generate(self, **kwargs: Any) -> FakeZitOutput:
        self.generate_calls.append(kwargs)
        if self.raise_on_generate is not None:
            raise self.raise_on_generate
        return FakeZitOutput(color="blue")

    def to(self, device: str) -> None:
        self.device = device


class FakeImageEditPipeline:
    """Fake `ImageEditPipeline` (FLUX.2 Klein) for tests.

    Records both txt2img (`generate`) and reference-conditioned edit
    (`generate_with_references`) calls, returning a small PIL image so the
    handler's save path works without a GPU.
    """

    _singleton: ClassVar["FakeImageEditPipeline | None"] = None

    @classmethod
    def bind_singleton(cls, pipeline: "FakeImageEditPipeline") -> None:
        cls._singleton = pipeline

    @staticmethod
    def create(
        model_path: str,
        device: str | None = None,
    ) -> "FakeImageEditPipeline":
        del model_path
        pipeline = FakeImageEditPipeline._singleton
        if pipeline is None:
            raise RuntimeError("FakeImageEditPipeline singleton is not bound")
        if device is not None:
            pipeline.to(device)
        return pipeline

    def __init__(self) -> None:
        self.device: str | None = None
        self.generate_calls: list[dict[str, Any]] = []
        self.generate_with_references_calls: list[dict[str, Any]] = []
        self.raise_on_generate: Exception | None = None

    def generate(self, **kwargs: Any) -> FakeZitOutput:
        self.generate_calls.append(kwargs)
        if self.raise_on_generate is not None:
            raise self.raise_on_generate
        return FakeZitOutput(color="green")

    def generate_with_references(self, **kwargs: Any) -> FakeZitOutput:
        self.generate_with_references_calls.append(kwargs)
        if self.raise_on_generate is not None:
            raise self.raise_on_generate
        return FakeZitOutput(color="green")

    def to(self, device: str) -> None:
        self.device = device


class FakeIcLoraPipeline:
    _singleton: ClassVar["FakeIcLoraPipeline | None"] = None

    @classmethod
    def bind_singleton(cls, pipeline: "FakeIcLoraPipeline") -> None:
        cls._singleton = pipeline

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        lora_path: str,
        device: str | object,
        streaming_prefetch_count: int | None,
        distilled_lora_path: str | None = None,
        lora_scale: float = 1.0,
    ) -> "FakeIcLoraPipeline":
        pipeline = FakeIcLoraPipeline._singleton
        if pipeline is None:
            raise RuntimeError("FakeIcLoraPipeline singleton is not bound")
        pipeline.create_calls.append(
            {
                "checkpoint_path": checkpoint_path,
                "gemma_root": gemma_root,
                "upsampler_path": upsampler_path,
                "lora_path": lora_path,
                "device": device,
                "streaming_prefetch_count": streaming_prefetch_count,
                "distilled_lora_path": distilled_lora_path,
                "lora_scale": lora_scale,
            }
        )
        return pipeline

    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []
        self.create_calls: list[dict[str, Any]] = []
        self.raise_on_generate: Exception | None = None

    def generate(self, **kwargs: Any) -> None:
        self.generate_calls.append(kwargs)
        if self.raise_on_generate is not None:
            raise self.raise_on_generate

        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-ic-lora-video")


class FakeDepthProcessorPipeline:
    _singleton: ClassVar["FakeDepthProcessorPipeline | None"] = None

    @classmethod
    def bind_singleton(cls, pipeline: "FakeDepthProcessorPipeline") -> None:
        cls._singleton = pipeline

    @staticmethod
    def create(
        model_path: str,
        device: str | object,
    ) -> "FakeDepthProcessorPipeline":
        del model_path, device
        pipeline = FakeDepthProcessorPipeline._singleton
        if pipeline is None:
            raise RuntimeError("FakeDepthProcessorPipeline singleton is not bound")
        return pipeline

    def __init__(self) -> None:
        self.apply_calls: list[Any] = []

    def apply(self, frame: Any) -> Any:
        self.apply_calls.append(frame)
        return f"depth:{frame}"


class FakePoseProcessorPipeline:
    _singleton: ClassVar["FakePoseProcessorPipeline | None"] = None

    @classmethod
    def bind_singleton(cls, pipeline: "FakePoseProcessorPipeline") -> None:
        cls._singleton = pipeline

    @staticmethod
    def create(
        pose_model_path: str,
        person_detector_model_path: str,
        device: str | object,
    ) -> "FakePoseProcessorPipeline":
        del pose_model_path, person_detector_model_path, device
        pipeline = FakePoseProcessorPipeline._singleton
        if pipeline is None:
            raise RuntimeError("FakePoseProcessorPipeline singleton is not bound")
        return pipeline

    def __init__(self) -> None:
        self.apply_calls: list[Any] = []

    def apply(self, frame: Any) -> Any:
        self.apply_calls.append(frame)
        return f"pose:{frame}"


class FakeA2VPipeline:
    _singleton: ClassVar["FakeA2VPipeline | None"] = None

    @classmethod
    def bind_singleton(cls, pipeline: "FakeA2VPipeline") -> None:
        cls._singleton = pipeline

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: str | object,
        streaming_prefetch_count: int | None,
    ) -> "FakeA2VPipeline":
        del checkpoint_path, gemma_root, upsampler_path, device, streaming_prefetch_count
        pipeline = FakeA2VPipeline._singleton
        if pipeline is None:
            raise RuntimeError("FakeA2VPipeline singleton is not bound")
        return pipeline

    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []
        self.raise_on_generate: Exception | None = None

    def generate(self, **kwargs: Any) -> None:
        self.generate_calls.append(kwargs)
        if self.raise_on_generate is not None:
            raise self.raise_on_generate

        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-a2v-video")


class FakeRetakePipeline:
    _singleton: ClassVar["FakeRetakePipeline | None"] = None

    @classmethod
    def bind_singleton(cls, pipeline: "FakeRetakePipeline") -> None:
        cls._singleton = pipeline

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        device: str | object,
        streaming_prefetch_count: int | None,
        *,
        loras: list[object] | None = None,
        quantization: object | None = None,
    ) -> "FakeRetakePipeline":
        del checkpoint_path, gemma_root, device, streaming_prefetch_count, loras, quantization
        pipeline = FakeRetakePipeline._singleton
        if pipeline is None:
            raise RuntimeError("FakeRetakePipeline singleton is not bound")
        return pipeline

    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []
        self.raise_on_generate: Exception | None = None

    def generate(self, **kwargs: Any) -> None:
        self.generate_calls.append(kwargs)
        if self.raise_on_generate is not None:
            raise self.raise_on_generate

        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-retake-video")


class FakeTextEncoder:
    def __init__(self) -> None:
        self.install_calls = 0
        self.encode_calls: list[dict[str, Any]] = []
        self.encode_responses: list[Any] = []

    def install_patches(self, state_getter) -> None:  # noqa: ARG002
        self.install_calls += 1

    def encode_via_api(self, prompt: str, api_key: str, checkpoint_path: str, enhance_prompt: bool) -> Any | None:
        self.encode_calls.append(
            {
                "prompt": prompt,
                "api_key": api_key,
                "checkpoint_path": checkpoint_path,
                "enhance_prompt": enhance_prompt,
            }
        )
        if self.encode_responses:
            return self.encode_responses.pop(0)
        return None


class FakeTrainerTarget:
    """In-memory stand-in for a remote GPU execution backend.

    Drives the LoRA runner tests without any network. By default every
    remote command "succeeds" immediately; tests override
    `command_results` (keyed by the substring matched in the command)
    or `next_status` to exercise running/failed branches. Uploaded dirs
    and downloaded files are recorded; downloads materialize a small
    local file so handler-side path bookkeeping is exercised for real.
    """

    def __init__(self) -> None:
        self.test_connection_calls: list[TrainerCredentials] = []
        self.ensure_workspace_calls: list[TargetHandle | None] = []
        self.ensure_provisioned_calls: list[TrainerCredentials] = []
        self.raise_on_ensure_provisioned: Exception | None = None
        self.uploaded_dirs: list[tuple[str, str]] = []
        self.downloaded_files: list[tuple[str, str]] = []
        self.started_commands: list[str] = []
        self.terminated: list[str] = []
        self.released = 0
        self._job_counter = 0
        # command-substring -> status returned by poll for that job
        self.command_results: dict[str, RemoteCommandStatus] = {}
        # remote_job_id -> queued statuses (popped per poll); falls back
        # to a succeeded status once exhausted.
        self.status_by_job: dict[str, list[RemoteCommandStatus]] = {}
        self.logs_by_job: dict[str, list[str]] = {}
        self.raise_on_test_connection: Exception | None = None
        self.raise_on_ensure_workspace: Exception | None = None
        self.raise_on_release: Exception | None = None
        self.raise_on_terminate: Exception | None = None
        self.pod_id = "fake-pod-1"
        self._job_command: dict[str, str] = {}
        # Connect-flow stubs. Tests can override `account_info` /
        # `raise_on_connect_account`; `ensure_network_volume` records its
        # calls and returns a deterministic volume.
        self.connect_account_calls: list[TrainerCredentials] = []
        self.raise_on_connect_account: Exception | None = None
        self.account_info = AccountInfo(
            gpus=(
                GpuOffer(
                    id="NVIDIA A100 80GB PCIe",
                    label="NVIDIA A100 80GB PCIe",
                    memory_gb=80,
                    price_per_hr=1.89,
                    available=True,
                ),
            ),
            volumes=(),
            pods=(
                PodInfo(
                    id="fake-pod-1",
                    name="ltx-desktop-lora",
                    gpu="NVIDIA A100 80GB PCIe",
                    status="RUNNING",
                    cost_per_hr=1.89,
                    created_by_app=True,
                ),
            ),
        )
        self.ensure_volume_calls: list[tuple[str, int]] = []
        self.ensure_volume_datacenters: list[str | None] = []
        self.deleted_volumes: list[str] = []
        self.raise_on_ensure_volume: Exception | None = None
        self.raise_on_delete_volume: Exception | None = None
        # GPU status + validation-feed stubs. `gpu_telemetry` is returned as-is
        # from `query_gpu` (set `raise_on_query_gpu` to exercise degraded
        # status); `validation_artifacts` is returned from
        # `list_validation_outputs` filtered to step > since_step.
        self.gpu_telemetry: GpuTelemetry = GpuTelemetry(
            name="NVIDIA FakeGPU",
            vram_total_mb=32768,
            vram_used_mb=4096,
            gpu_util_pct=42,
            mem_util_pct=12,
            temp_c=55,
        )
        self.raise_on_query_gpu: Exception | None = None
        self.validation_artifacts: list[ValidationArtifact] = []
        self.query_gpu_calls: int = 0
        self.list_validation_calls: list[tuple[str, int]] = []
        # Checkpoints present on the remote (step numbers). `list_checkpoints`
        # returns this as-is (sorted), so download/redownload tests can stage
        # the highest existing remote adapter without a real `ls`.
        self.checkpoints_steps: list[int] = []
        self.list_checkpoints_calls: list[str] = []
        # Per-source `.pt` file counts the fake reports for
        # `count_precomputed_source` (keyed by source name, e.g. "latents",
        # "audio_latents"). The runner's post-preprocess guard reads these to
        # fail fast when a source silently produced 0 files. Defaults to a
        # populated latents source so a normal run goes `ready` without setup.
        self.precomputed_source_counts: dict[str, int] = {
            "latents": 8,
            "conditions": 8,
            "reference_latents": 8,
        }
        self.count_precomputed_source_calls: list[str] = []
        # Remote paths handed to `delete_remote_paths` (reset clears a stage's
        # artifacts on the workspace). Recorded so resume/reset tests can assert
        # exactly which remote dirs were wiped.
        self.deleted_remote_paths: list[list[str]] = []
        # In-memory pod store for the Trainer compute panel: list_pods returns
        # this as-is, and stop_pod/resume_pod mutate each pod's `desired_status`
        # /`running` so tests can assert the lifecycle transitions. Seed it with
        # the connect-flow pod so a default run shows one running pod.
        self.pods: list[PodInfo] = [
            PodInfo(
                id="fake-pod-1",
                name="ltx-desktop-lora",
                gpu="NVIDIA A100 80GB PCIe",
                status="RUNNING",
                cost_per_hr=1.89,
                created_by_app=True,
                desired_status="RUNNING",
                running=True,
                uptime_seconds=3_600,
                last_started_at="2026-01-01T00:00:00+00:00",
            )
        ]
        self.list_pods_calls: list[TrainerCredentials] = []
        self.stopped_pods: list[str] = []
        self.resumed_pods: list[str] = []
        # Optional failure injection for the compute-panel lifecycle actions
        # (None = success). Mirrors the existing `raise_on_*` pattern so route
        # tests can exercise the ok=False / 502 branches without mocks.
        self.raise_on_list_pods: Exception | None = None
        self.raise_on_stop_pod: Exception | None = None
        self.raise_on_resume_pod: Exception | None = None

    def test_connection(self, *, credentials: TrainerCredentials) -> None:
        self.test_connection_calls.append(credentials)
        if self.raise_on_test_connection is not None:
            raise self.raise_on_test_connection

    def connect_account(self, *, credentials: TrainerCredentials) -> AccountInfo:
        self.connect_account_calls.append(credentials)
        if self.raise_on_connect_account is not None:
            raise self.raise_on_connect_account
        return self.account_info

    def list_pods(self, *, credentials: TrainerCredentials) -> list[PodInfo]:
        self.list_pods_calls.append(credentials)
        if self.raise_on_list_pods is not None:
            raise self.raise_on_list_pods
        return list(self.pods)

    def stop_pod(self, *, credentials: TrainerCredentials, pod_id: str) -> None:
        del credentials
        if self.raise_on_stop_pod is not None:
            raise self.raise_on_stop_pod
        self.stopped_pods.append(pod_id)
        self._set_pod(pod_id, desired_status="STOPPED", running=False)

    def resume_pod(self, *, credentials: TrainerCredentials, pod_id: str) -> None:
        del credentials
        if self.raise_on_resume_pod is not None:
            raise self.raise_on_resume_pod
        self.resumed_pods.append(pod_id)
        self._set_pod(pod_id, desired_status="RUNNING", running=True)

    def _set_pod(self, pod_id: str, *, desired_status: str, running: bool) -> None:
        for i, pod in enumerate(self.pods):
            if pod.id == pod_id:
                self.pods[i] = replace(
                    pod,
                    status=desired_status,
                    desired_status=desired_status,
                    running=running,
                )
                return

    def ensure_network_volume(
        self,
        *,
        credentials: TrainerCredentials,
        name: str,
        size_gb: int,
        datacenter_id: str | None = None,
    ) -> NetworkVolume:
        del credentials
        if self.raise_on_ensure_volume is not None:
            raise self.raise_on_ensure_volume
        self.ensure_volume_calls.append((name, size_gb))
        self.ensure_volume_datacenters.append(datacenter_id)
        volume = NetworkVolume(
            id=f"fake-vol-{len(self.ensure_volume_calls)}",
            name=name,
            size_gb=size_gb,
            datacenter_id=datacenter_id or "EU-RO-1",
            created_by_app=True,
        )
        self.account_info = replace(
            self.account_info, volumes=(*self.account_info.volumes, volume)
        )
        return volume

    def delete_network_volume(
        self, *, credentials: TrainerCredentials, volume_id: str
    ) -> None:
        del credentials
        if self.raise_on_delete_volume is not None:
            raise self.raise_on_delete_volume
        volume = next((v for v in self.account_info.volumes if v.id == volume_id), None)
        if volume is not None and not volume.created_by_app:
            from services.trainer_target.trainer_target import TrainerTargetError

            raise TrainerTargetError(
                f"Refusing to delete RunPod volume {volume_id}: it was not created "
                "by LTX Desktop",
                retryable=False,
            )
        self.deleted_volumes.append(volume_id)
        self.account_info = replace(
            self.account_info,
            volumes=tuple(v for v in self.account_info.volumes if v.id != volume_id),
        )

    def ensure_workspace(
        self, *, credentials: TrainerCredentials, handle: TargetHandle | None
    ) -> TargetHandle:
        self.ensure_workspace_calls.append(handle)
        if self.raise_on_ensure_workspace is not None:
            raise self.raise_on_ensure_workspace
        if handle is not None and handle.pod_id is not None:
            return handle
        return TargetHandle(provider=credentials.provider, pod_id=self.pod_id)

    def ensure_provisioned(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        progress=None,
    ) -> None:
        del handle
        self.ensure_provisioned_calls.append(credentials)
        if progress is not None:
            progress("fake provisioning: 100%")
        if self.raise_on_ensure_provisioned is not None:
            raise self.raise_on_ensure_provisioned

    def upload_directory(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        local_dir: str,
        remote_dir: str,
    ) -> None:
        del credentials, handle
        self.uploaded_dirs.append((local_dir, remote_dir))

    def download_file(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        remote_path: str,
        local_path: str,
    ) -> None:
        del credentials, handle
        self.downloaded_files.append((remote_path, local_path))
        destination = Path(local_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fake-lora-weights")

    def start_command(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        command: str,
        workdir: str,
    ) -> str:
        del credentials, handle, workdir
        self._job_counter += 1
        job_id = f"job-{self._job_counter}"
        self.started_commands.append(command)
        self._job_command[job_id] = command
        return job_id

    def poll_command(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        remote_job_id: str,
    ) -> RemoteCommandStatus:
        del credentials, handle
        queued = self.status_by_job.get(remote_job_id)
        if queued:
            return queued.pop(0)
        command = self._job_command.get(remote_job_id, "")
        for needle, status in self.command_results.items():
            if needle in command:
                return status
        return RemoteCommandStatus(state="succeeded", exit_code=0)

    def read_logs(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        remote_job_id: str,
        tail: int,
    ) -> list[str]:
        del credentials, handle
        return self.logs_by_job.get(remote_job_id, [])[-tail:]

    def terminate(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        remote_job_id: str,
    ) -> None:
        del credentials, handle
        if self.raise_on_terminate is not None:
            raise self.raise_on_terminate
        self.terminated.append(remote_job_id)

    def release_workspace(
        self, *, credentials: TrainerCredentials, handle: TargetHandle
    ) -> None:
        if self.raise_on_release is not None:
            raise self.raise_on_release
        if handle.pod_id is not None and credentials.runpod_network_volume_id:
            self.stopped_pods.append(handle.pod_id)
            self._set_pod(handle.pod_id, desired_status="STOPPED", running=False)
        del credentials, handle
        self.released += 1

    def query_gpu(
        self, *, credentials: TrainerCredentials, handle: TargetHandle
    ) -> GpuTelemetry:
        del credentials, handle
        self.query_gpu_calls += 1
        if self.raise_on_query_gpu is not None:
            raise self.raise_on_query_gpu
        return self.gpu_telemetry

    def list_validation_outputs(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        remote_output_dir: str,
        since_step: int,
    ) -> list[ValidationArtifact]:
        del credentials, handle
        self.list_validation_calls.append((remote_output_dir, since_step))
        return [a for a in self.validation_artifacts if a.step > since_step]

    def list_checkpoints(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        remote_output_dir: str,
    ) -> list[int]:
        del credentials, handle
        self.list_checkpoints_calls.append(remote_output_dir)
        return sorted(self.checkpoints_steps)

    def count_precomputed_source(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        precomputed_dir: str,
        source: str,
    ) -> int:
        del credentials, handle, precomputed_dir
        self.count_precomputed_source_calls.append(source)
        return self.precomputed_source_counts.get(source, 0)

    def delete_remote_paths(
        self,
        *,
        credentials: TrainerCredentials,
        handle: TargetHandle,
        paths: list[str],
    ) -> None:
        del credentials, handle
        self.deleted_remote_paths.append(list(paths))
        # Mirror the real target: a checkpoint wipe clears the recorded remote
        # checkpoints so a subsequent listing (e.g. resume's load_checkpoint
        # resolution) reflects the wiped state.
        for p in paths:
            if "checkpoints" in p:
                self.checkpoints_steps = []


class FakeLoraPromptProfiler:
    """Deterministic `LoraPromptProfiler` for tests.

    Records calls and returns `result` (a "skipped" outcome by default, so the
    import flow falls back to the name-derived default unless a test sets a
    configured/builtin/failed result). Set `raise_on_profile` to simulate an
    unexpected error path (the handler treats any raise as a failed outcome).
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.result: LoraPromptProfileResult = LoraPromptProfileResult(
            status="skipped",
            message="test: skipped",
        )
        self.raise_on_profile: Exception | None = None

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
        self.calls.append(
            {
                "name": name,
                "filename": filename,
                "variant": variant,
                "huggingface_url": huggingface_url,
                "example_prompt": example_prompt,
                "api_key": api_key,
            }
        )
        if self.raise_on_profile is not None:
            raise self.raise_on_profile
        return self.result


class FakeVideoCaptioner:
    """Deterministic `VideoCaptioner` for tests.

    Records calls and returns a canned caption. Set `error` to make the
    next call raise, exercising the route's error mapping.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.caption_text: str = "A short test clip."
        self.error: VideoCaptionerError | None = None

    def caption(
        self,
        *,
        video_path: str,
        api_key: str,
        with_audio: bool,
        instructions: str | None = None,
    ) -> str:
        self.calls.append(
            {
                "video_path": video_path,
                "api_key": api_key,
                "with_audio": with_audio,
                "instructions": instructions,
            }
        )
        if self.error is not None:
            raise self.error
        return self.caption_text


class FakeClipProcessor:
    """Deterministic `ClipProcessor` for tests.

    Returns a canned probe and records calls. Set `error` to make the
    next probe raise, exercising the route's error mapping.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.render_calls: list[dict[str, object]] = []
        self.scene_calls: list[dict[str, object]] = []
        self.frame_calls: list[dict[str, object]] = []
        self.sprite_calls: list[dict[str, object]] = []
        self.result: ClipProbeResult = ClipProbeResult(
            duration_seconds=5.0,
            width=1280,
            height=720,
            fps=24.0,
            frame_count=120,
            has_audio=False,
            video_codec="h264",
        )
        self.scenes: list[SceneSpan] = [
            SceneSpan(start_seconds=0.0, end_seconds=2.5),
            SceneSpan(start_seconds=2.5, end_seconds=5.0),
        ]
        self.error: ClipProcessorError | None = None
        # Per-path probe overrides (keyed by exact path). Lets a test give the
        # normalized OUTPUT clips specific dims/fps/frame counts so the dataset-
        # prep validation paths (fps/WxH/frame mismatch) can be exercised.
        self.results_by_path: dict[str, ClipProbeResult] = {}
        self.normalize_calls: list[dict[str, object]] = []

    def probe(self, *, video_path: str) -> ClipProbeResult:
        self.calls.append(video_path)
        if self.error is not None:
            raise self.error
        return self.results_by_path.get(video_path, self.result)

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
        self.normalize_calls.append(
            {
                "source_path": source_path,
                "out_path": out_path,
                "fps": fps,
                "short_side": short_side,
                "frames": frames,
                "max_duration_seconds": max_duration_seconds,
                "exact_width": exact_width,
                "exact_height": exact_height,
            }
        )
        if self.error is not None:
            raise self.error
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"fake-normalized")

    def render(self, *, source_path: str, plan: EditPlan, out_path: str) -> None:
        self.render_calls.append(
            {
                "source_path": source_path,
                "plan": plan,
                "trim": plan.trim,
                "crop": plan.crop,
                "scale": plan.scale,
                "out_path": out_path,
            }
        )
        if self.error is not None:
            raise self.error
        # Write a placeholder so callers that stat the output succeed.
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"fake")

    def detect_scenes(self, *, video_path: str, threshold: float) -> list[SceneSpan]:
        self.scene_calls.append({"video_path": video_path, "threshold": threshold})
        if self.error is not None:
            raise self.error
        return self.scenes

    def extract_frame(self, *, video_path: str, time_seconds: float) -> bytes:
        self.frame_calls.append({"video_path": video_path, "time_seconds": time_seconds})
        if self.error is not None:
            raise self.error
        return b"fake-frame-png"

    def generate_sprite(
        self, *, video_path: str, out_path: str, tile_count: int, tile_width: int
    ) -> int:
        self.sprite_calls.append(
            {"video_path": video_path, "out_path": out_path, "tile_count": tile_count, "tile_width": tile_width}
        )
        if self.error is not None:
            raise self.error
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"fake-sprite")
        return tile_count


class FakeImageEditor:
    """Deterministic `ImageEditor` for tests.

    Records calls and returns canned edited bytes. Set `error` to make
    the next call raise, exercising the route's error mapping.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.result: bytes = b"fake-edited-png"
        self.error: ImageEditorError | None = None

    def edit(
        self, *, image_bytes: bytes, prompt: str, model: str, api_key: str
    ) -> bytes:
        self.calls.append(
            {"prompt": prompt, "model": model, "api_key": api_key, "size": len(image_bytes)}
        )
        if self.error is not None:
            raise self.error
        return self.result


class FakeVideoRestyler:
    """Deterministic `VideoRestyler` for tests (vid2vid + i2v)."""

    def __init__(self) -> None:
        self.restyle_calls: list[dict[str, object]] = []
        self.animate_calls: list[dict[str, object]] = []
        self.motion_edit_calls: list[dict[str, object]] = []
        self.motion_transfer_calls: list[dict[str, object]] = []
        self.kling_v2v_edit_calls: list[dict[str, object]] = []
        self.result: bytes = b"fake-video-mp4"
        self.error: VideoRestylerError | None = None
        # Transient-failure simulation for retry tests: the next `fail_times`
        # drive calls raise `transient_error` (default a 429), then succeed.
        # Independent of `self.error` (which fails permanently, always).
        self.fail_times: int = 0
        self.transient_error: VideoRestylerError | None = None

    def _maybe_fail(self) -> None:
        if self.fail_times > 0:
            self.fail_times -= 1
            raise self.transient_error or VideoRestylerError(
                "Fal video job failed (429): rate limited"
            )
        if self.error is not None:
            raise self.error

    def restyle(self, *, video_bytes: bytes, prompt: str, api_key: str) -> bytes:
        self.restyle_calls.append(
            {"prompt": prompt, "api_key": api_key, "size": len(video_bytes)}
        )
        if self.error is not None:
            raise self.error
        return self.result

    def animate(self, *, image_bytes: bytes, prompt: str, api_key: str) -> bytes:
        self.animate_calls.append(
            {"prompt": prompt, "api_key": api_key, "size": len(image_bytes)}
        )
        if self.error is not None:
            raise self.error
        return self.result

    def motion_edit(
        self,
        *,
        video_bytes: bytes,
        image_bytes: bytes,
        prompt: str,
        video_strength: float,
        api_key: str,
    ) -> bytes:
        self.motion_edit_calls.append(
            {
                "prompt": prompt,
                "api_key": api_key,
                "video_size": len(video_bytes),
                "image_size": len(image_bytes),
                "video_strength": video_strength,
            }
        )
        if self.error is not None:
            raise self.error
        return self.result

    def motion_transfer(
        self,
        *,
        image_bytes: bytes,
        video_bytes: bytes,
        prompt: str,
        character_orientation: str,
        api_key: str,
    ) -> bytes:
        self._maybe_fail()
        self.motion_transfer_calls.append(
            {
                "prompt": prompt,
                "api_key": api_key,
                "video_size": len(video_bytes),
                "image_size": len(image_bytes),
                "character_orientation": character_orientation,
            }
        )
        if self.error is not None:
            raise self.error
        return self.result

    def kling_v2v_edit(
        self,
        *,
        video_bytes: bytes,
        image_bytes: bytes | None,
        prompt: str,
        keep_audio: bool,
        api_key: str,
    ) -> bytes:
        self._maybe_fail()
        self.kling_v2v_edit_calls.append(
            {
                "prompt": prompt,
                "api_key": api_key,
                "video_size": len(video_bytes),
                "image_size": len(image_bytes) if image_bytes is not None else None,
                "keep_audio": keep_audio,
            }
        )
        return self.result


class FakePexelsClient:
    """Deterministic `PexelsClient` for tests (search + download)."""

    def __init__(self) -> None:
        self.search_calls: list[dict[str, object]] = []
        self.download_calls: list[dict[str, object]] = []
        self.result: PexelsSearchResult | None = None
        self.download_bytes: bytes = b"fake-pexels-bytes"
        self.error: PexelsError | None = None

    def search(
        self,
        *,
        query: str,
        media: str,
        page: int,
        per_page: int,
        orientation: str,
        api_key: str,
    ) -> PexelsSearchResult:
        self.search_calls.append(
            {
                "query": query,
                "media": media,
                "page": page,
                "per_page": per_page,
                "orientation": orientation,
                "api_key": api_key,
            }
        )
        if self.error is not None:
            raise self.error
        if self.result is not None:
            return self.result
        item = PexelsMediaResult(
            id="123",
            kind="video" if media == "video" else "photo",
            width=1920,
            height=1080,
            duration_seconds=8.0 if media == "video" else None,
            preview_url="https://images.pexels.com/preview.jpg",
            download_url=(
                "https://videos.pexels.com/file.mp4"
                if media == "video"
                else "https://images.pexels.com/file.jpg"
            ),
            download_ext="mp4" if media == "video" else "jpg",
            pexels_url="https://www.pexels.com/video/123/",
            author="Jane Doe",
            author_url="https://www.pexels.com/@jane",
            alt="a clip",
        )
        return PexelsSearchResult(
            items=[item], page=page, per_page=per_page, total_results=1, has_next=False
        )

    def download(self, *, url: str, api_key: str) -> bytes:
        self.download_calls.append({"url": url, "api_key": api_key})
        if self.error is not None:
            raise self.error
        return self.download_bytes


@dataclass
class FakeServices:
    http: FakeHTTPClient = field(default_factory=FakeHTTPClient)
    gpu_cleaner: FakeGpuCleaner = field(default_factory=FakeGpuCleaner)
    model_downloader: FakeModelDownloader = field(default_factory=FakeModelDownloader)
    gpu_info: FakeGpuInfo = field(default_factory=FakeGpuInfo)
    video_processor: FakeVideoProcessor = field(default_factory=FakeVideoProcessor)
    text_encoder: FakeTextEncoder = field(default_factory=FakeTextEncoder)
    task_runner: FakeTaskRunner = field(default_factory=FakeTaskRunner)
    ltx_api_client: FakeLTXAPIClient = field(default_factory=FakeLTXAPIClient)
    zit_api_client: FakeZitAPIClient = field(default_factory=FakeZitAPIClient)
    fast_video_pipeline: FakeFastVideoPipeline = field(default_factory=FakeFastVideoPipeline)
    image_generation_pipeline: FakeImageGenerationPipeline = field(default_factory=FakeImageGenerationPipeline)
    ic_lora_pipeline: FakeIcLoraPipeline = field(default_factory=FakeIcLoraPipeline)
    depth_processor_pipeline: FakeDepthProcessorPipeline = field(default_factory=FakeDepthProcessorPipeline)
    pose_processor_pipeline: FakePoseProcessorPipeline = field(default_factory=FakePoseProcessorPipeline)
    a2v_pipeline: FakeA2VPipeline = field(default_factory=FakeA2VPipeline)
    retake_pipeline: FakeRetakePipeline = field(default_factory=FakeRetakePipeline)
    image_edit_pipeline: FakeImageEditPipeline = field(default_factory=FakeImageEditPipeline)
    trainer_target: FakeTrainerTarget = field(default_factory=FakeTrainerTarget)
    video_captioner: FakeVideoCaptioner = field(default_factory=FakeVideoCaptioner)
    lora_prompt_profiler: FakeLoraPromptProfiler = field(default_factory=FakeLoraPromptProfiler)
    clip_processor: FakeClipProcessor = field(default_factory=FakeClipProcessor)
    image_editor: FakeImageEditor = field(default_factory=FakeImageEditor)
    video_restyler: FakeVideoRestyler = field(default_factory=FakeVideoRestyler)
    pexels_client: FakePexelsClient = field(default_factory=FakePexelsClient)

    def __post_init__(self) -> None:
        FakeFastVideoPipeline.bind_singleton(self.fast_video_pipeline)
        FakeImageGenerationPipeline.bind_singleton(self.image_generation_pipeline)
        FakeIcLoraPipeline.bind_singleton(self.ic_lora_pipeline)
        FakeDepthProcessorPipeline.bind_singleton(self.depth_processor_pipeline)
        FakePoseProcessorPipeline.bind_singleton(self.pose_processor_pipeline)
        FakeA2VPipeline.bind_singleton(self.a2v_pipeline)
        FakeRetakePipeline.bind_singleton(self.retake_pipeline)
        FakeImageEditPipeline.bind_singleton(self.image_edit_pipeline)

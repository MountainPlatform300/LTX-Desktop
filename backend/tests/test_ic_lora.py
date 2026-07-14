"""Integration-style tests for IC-LoRA endpoints."""

from __future__ import annotations

from pathlib import Path

from runtime_config.model_download_specs import (
    PERSON_DETECTOR_CP_ID,
    POSE_PROCESSOR_CP_ID,
    resolve_model_path,
)
from tests.http_error_assertions import assert_http_error
from tests.fakes import FakeCapture


def _create_pose_processor_files(test_state) -> None:
    """Pose conditioning needs the DW pose model + YOLOX person detector on disk."""
    for cp_id in (POSE_PROCESSOR_CP_ID, PERSON_DETECTOR_CP_ID):
        path = resolve_model_path(test_state.config.default_models_dir, cp_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00" * 1024)


class TestIcLoraExtractConditioning:
    def test_canny_extraction(self, client, test_state):
        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a"]))

        response = client.post(
            "/api/ic-lora/extract-conditioning",
            json={"video_path": str(video_path), "conditioning_type": "canny", "frame_time": 0},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["conditioning_type"] == "canny"
        assert payload["conditioning"].startswith("data:image/jpeg;base64,")

    def test_depth_extraction(self, client, test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files):
        create_fake_model_files()
        create_fake_ic_lora_files()
        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a"]))

        response = client.post(
            "/api/ic-lora/extract-conditioning",
            json={"video_path": str(video_path), "conditioning_type": "depth", "frame_time": 0},
        )
        assert response.status_code == 200
        assert response.json()["conditioning_type"] == "depth"
        assert fake_services.depth_processor_pipeline.apply_calls == ["frame-a"]

    def test_pose_extraction(self, client, test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files):
        create_fake_model_files()
        create_fake_ic_lora_files()
        _create_pose_processor_files(test_state)
        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a"]))

        response = client.post(
            "/api/ic-lora/extract-conditioning",
            json={"video_path": str(video_path), "conditioning_type": "pose", "frame_time": 0},
        )
        assert response.status_code == 200
        assert response.json()["conditioning_type"] == "pose"
        assert fake_services.pose_processor_pipeline.apply_calls == ["frame-a"]

    def test_depth_extraction_requires_downloaded_ltx_model(self, client, test_state):
        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a"]))

        response = client.post(
            "/api/ic-lora/extract-conditioning",
            json={"video_path": str(video_path), "conditioning_type": "depth", "frame_time": 0},
        )
        assert_http_error(response, status_code=409, code="NO_DOWNLOADED_LTX_MODEL")


class TestIcLoraGenerate:
    def test_happy_path(self, client, test_state, create_fake_model_files, create_fake_ic_lora_files):
        create_fake_model_files()
        create_fake_ic_lora_files()
        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [],
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "complete"
        assert Path(response.json()["video_path"]).exists()

    def test_cancel_during_pipeline_load_returns_cancelled(
        self, client, test_state, monkeypatch, create_fake_model_files, create_fake_ic_lora_files
    ):
        """A Stop clicked during the (cold) pipeline load must land — the run
        should return `cancelled`, not `complete`.

        Regression: start_generation used to run *after* load_ic_lora, so during
        the load there was no running generation and cancel_generation returned
        `no_active_generation` without setting the cancel flag — the run then
        completed despite Stop. Now start_generation runs before the load, so a
        cancel issued mid-load flips the state to cancelled and the post-load
        cancel check unwinds the run.
        """
        create_fake_model_files()
        create_fake_ic_lora_files()
        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        real_load = test_state.pipelines.load_ic_lora

        def load_and_cancel(*args, **kwargs):
            # Simulate the user clicking Stop while the pipeline is loading.
            # start_generation has already run, so this lands on a running
            # generation and flips it to cancelled.
            test_state.generation.cancel_generation()
            return real_load(*args, **kwargs)

        monkeypatch.setattr(test_state.pipelines, "load_ic_lora", load_and_cancel)

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "canny",
                "prompt": "test prompt",
                "images": [],
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"

    def test_pose_generate_runs_pose_processor_over_every_frame(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files
    ):
        create_fake_model_files()
        create_fake_ic_lora_files()
        _create_pose_processor_files(test_state)
        test_state.state.app_settings.use_local_text_encoder = True

        video_path = test_state.config.outputs_dir / "test_video.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(str(video_path), FakeCapture(frames=["frame-a", "frame-b"]))

        response = client.post(
            "/api/ic-lora/generate",
            json={
                "video_path": str(video_path),
                "conditioning_type": "pose",
                "prompt": "test prompt",
                "images": [],
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "complete"
        # The control video is built over the *standardized* frame stream
        # (24fps, 8k+1 frames). A 2-frame @ 24fps reference snaps to a 5s
        # duration -> 121 frames, resampled then freeze-padded with the last
        # frame. Pose processor runs once per standardized frame.
        apply_calls = fake_services.pose_processor_pipeline.apply_calls
        assert len(apply_calls) == 121
        assert apply_calls[0] == "frame-a"
        assert apply_calls[1] == "frame-b"
        assert apply_calls[2:] == ["frame-b"] * 119
        assert Path(response.json()["video_path"]).exists()

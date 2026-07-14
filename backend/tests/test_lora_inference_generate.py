"""Integration tests for in-app LoRA inference generate (Gen Space "Apply LoRA").

Covers all three `LoraGenerateRequest` variants through the synchronous
`/api/lora-inference/generate` route, the unknown-id 404 boundary, and the
queue integration (`kind="lora"` payload enqueues + dispatches through
`AppHandler._dispatch_queue_payload`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from api_types import GenerateVideoRequest, LoraQueuePayload, LoraStandardGenerateRequest
from runtime_config.model_download_specs import resolve_model_path
from state.queue_state import QueuePayload
from tests.fakes import FakeCapture

from tests.test_lora_inference_registry import _inject_completed_run, _write_union_checkpoint


def _standard_request(lora_id: str, *, lora_scale: float = 0.8) -> dict:
    return {
        "variant": "standard",
        "loraId": lora_id,
        "loraScale": lora_scale,
        "request": {
            "prompt": "a cinematic shot",
            "resolution": "540p",
            "duration": 5,
            "fps": 24,
            "aspectRatio": "16:9",
        },
    }


def _register_reference_video(test_state, *, frames=("frame-a", "frame-b")) -> Path:
    video_path = test_state.config.outputs_dir / "ref_video.mp4"
    video_path.write_bytes(b"\x00" * 100)
    test_state.video_processor.register_video(
        str(video_path), FakeCapture(frames=list(frames), width=64, height=64)
    )
    return video_path


# ------------------------------------------------------------------
# Synchronous /api/lora-inference/generate
# ------------------------------------------------------------------


class TestLoraInferenceGenerateRoute:
    def test_standard_variant_runs_fast_pipeline_with_adapter(
        self, client, test_state, fake_services, create_fake_model_files, tmp_path
    ):
        create_fake_model_files()
        job = _inject_completed_run(
            test_state.lora_training, tmp_path, job_id="gen-std", dataset_type="standard"
        )
        test_state.state.app_settings.use_local_text_encoder = True

        resp = client.post(
            "/api/lora-inference/generate",
            json=_standard_request("user-gen-std", lora_scale=0.8),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "complete"
        assert Path(body["videoPath"]).exists()
        # The adapter must be built into the fast pipeline (B2a): create() was
        # called with the user's lora_path + the requested scale.
        assert fake_services.fast_video_pipeline.last_lora_path == job.local_lora_path
        assert fake_services.fast_video_pipeline.last_lora_scale == 0.8
        assert (
            fake_services.fast_video_pipeline.generate_calls[-1]["prompt"]
            == "MYTOK a cinematic shot"
        )

    def test_union_control_variant_runs_ic_lora_pipeline(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files
    ):
        create_fake_model_files()
        create_fake_ic_lora_files()
        _write_union_checkpoint(test_state)
        test_state.state.app_settings.use_local_text_encoder = True
        video_path = _register_reference_video(test_state)

        resp = client.post(
            "/api/lora-inference/generate",
            json={
                "variant": "union_control",
                "loraId": "official-ic-lora-union",
                "request": {
                    "video_path": str(video_path),
                    "conditioning_type": "canny",
                    "prompt": "follow the reference",
                    "images": [],
                },
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "complete"
        assert Path(body["videoPath"]).exists()
        assert fake_services.ic_lora_pipeline.generate_calls, "IC-LoRA pipeline must run"
        # Union control is standardized too: 64x64 input -> 16:9 bucket (960x576),
        # 24fps, 8k+1 frames. The control signal is built on that same grid.
        # The pipeline is called with a 2x target so stage 1 lands at the bucket;
        # refine defaults to false so stage 2 is skipped (single high-res stage).
        # 540p bucket is 64-multiple so the stage-1 latent (30x18) is even and the
        # two-stage patchify (p1=p2=p3=2) splits cleanly.
        call = fake_services.ic_lora_pipeline.generate_calls[-1]
        assert call["width"] == 1920
        assert call["height"] == 1152
        assert call["skip_stage_2"] is True
        assert call["num_frames"] == 121
        assert call["frame_rate"] == 24
        assert len(call["video_conditioning"]) == 1

    def test_union_control_downscales_oversized_reference_before_preprocess(
        self,
        client,
        test_state,
        create_fake_model_files,
        create_fake_ic_lora_files,
    ):
        create_fake_model_files()
        create_fake_ic_lora_files()
        _write_union_checkpoint(test_state)
        test_state.state.app_settings.use_local_text_encoder = True
        video_path = test_state.config.outputs_dir / "union_4k.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(
            str(video_path),
            FakeCapture(frames=["a", "b"], width=3840, height=2160),
        )

        resp = client.post(
            "/api/lora-inference/generate",
            json={
                "variant": "union_control",
                "loraId": "official-ic-lora-union",
                "request": {
                    "video_path": str(video_path),
                    "conditioning_type": "canny",
                    "prompt": "follow the reference",
                    "images": [],
                },
            },
        )

        assert resp.status_code == 200, resp.text
        assert set(test_state.video_processor.resize_calls) == {(1024, 576)}
        assert len(test_state.video_processor.resize_calls) == 121

    def test_union_control_buckets_portrait_reference_to_9_16(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files
    ):
        create_fake_model_files()
        create_fake_ic_lora_files()
        _write_union_checkpoint(test_state)
        test_state.state.app_settings.use_local_text_encoder = True
        # Portrait reference (height > width) must bucket to 9:16 (576x960),
        # not the landscape bucket — this is the AR fix that prevents distortion.
        # Pipeline called with 2x target -> (1152, 1920).
        video_path = test_state.config.outputs_dir / "ref_portrait.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(
            str(video_path), FakeCapture(frames=["f0", "f1", "f2"], width=64, height=128)
        )

        resp = client.post(
            "/api/lora-inference/generate",
            json={
                "variant": "union_control",
                "loraId": "official-ic-lora-union",
                "request": {
                    "video_path": str(video_path),
                    "conditioning_type": "canny",
                    "prompt": "follow the reference",
                    "images": [],
                },
            },
        )
        assert resp.status_code == 200, resp.text
        call = fake_services.ic_lora_pipeline.generate_calls[-1]
        assert call["width"] == 1152
        assert call["height"] == 1920
        assert call["frame_rate"] == 24

    def test_union_control_resolution_override_to_720p(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files
    ):
        create_fake_model_files()
        create_fake_ic_lora_files()
        _write_union_checkpoint(test_state)
        test_state.state.app_settings.use_local_text_encoder = True
        video_path = _register_reference_video(test_state)

        resp = client.post(
            "/api/lora-inference/generate",
            json={
                "variant": "union_control",
                "loraId": "official-ic-lora-union",
                "request": {
                    "video_path": str(video_path),
                    "conditioning_type": "canny",
                    "prompt": "follow the reference",
                    "images": [],
                    "resolution": "720p",
                },
            },
        )
        assert resp.status_code == 200, resp.text
        # Stage-1 ALWAYS diffuses at the 540p bucket (960x576), so the pipeline
        # is called with the fixed 2x target (1920x1152) regardless of the
        # requested output. 720p engages the upsampler (skip_stage_2=False) and
        # is then downscaled from 1920x1152 to 1280x768 — it never diffuses
        # natively at 720p (which would cost more VRAM off-distribution).
        call = fake_services.ic_lora_pipeline.generate_calls[-1]
        assert call["width"] == 1920
        assert call["height"] == 1152
        assert call["skip_stage_2"] is False

    def test_union_control_resolution_override_to_1080p_portrait(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files
    ):
        create_fake_model_files()
        create_fake_ic_lora_files()
        _write_union_checkpoint(test_state)
        test_state.state.app_settings.use_local_text_encoder = True
        # Portrait reference buckets to 9:16; stage-1 is always the 540p 9:16
        # bucket (576x960), so the pipeline gets the fixed 2x target (1152x1920).
        # 1080p engages the upsampler (skip_stage_2=False) -> 1152x1920 output.
        video_path = test_state.config.outputs_dir / "ref_portrait.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(
            str(video_path), FakeCapture(frames=["f0", "f1", "f2"], width=64, height=128)
        )

        resp = client.post(
            "/api/lora-inference/generate",
            json={
                "variant": "union_control",
                "loraId": "official-ic-lora-union",
                "request": {
                    "video_path": str(video_path),
                    "conditioning_type": "canny",
                    "prompt": "follow the reference",
                    "images": [],
                    "resolution": "1080p",
                },
            },
        )
        assert resp.status_code == 200, resp.text
        call = fake_services.ic_lora_pipeline.generate_calls[-1]
        assert call["width"] == 1152
        assert call["height"] == 1920
        assert call["skip_stage_2"] is False

    def test_union_control_rejects_unsupported_resolution(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files
    ):
        create_fake_model_files()
        create_fake_ic_lora_files()
        _write_union_checkpoint(test_state)
        test_state.state.app_settings.use_local_text_encoder = True
        video_path = _register_reference_video(test_state)

        resp = client.post(
            "/api/lora-inference/generate",
            json={
                "variant": "union_control",
                "loraId": "official-ic-lora-union",
                "request": {
                    "video_path": str(video_path),
                    "conditioning_type": "canny",
                    "prompt": "follow the reference",
                    "images": [],
                    "resolution": "1440p",
                },
            },
        )
        # Literal["540p","720p","1080p"] is rejected at the pydantic boundary.
        assert resp.status_code == 422, resp.text

    def test_union_control_duration_override_and_control_sibling(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files
    ):
        create_fake_model_files()
        create_fake_ic_lora_files()
        _write_union_checkpoint(test_state)
        test_state.state.app_settings.use_local_text_encoder = True
        video_path = _register_reference_video(test_state)

        resp = client.post(
            "/api/lora-inference/generate",
            json={
                "variant": "union_control",
                "loraId": "official-ic-lora-union",
                "request": {
                    "video_path": str(video_path),
                    "conditioning_type": "canny",
                    "prompt": "follow the reference",
                    "images": [],
                    "duration": 8,
                },
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "complete"
        # 8s @ 24fps on the 8k+1 lattice -> 193 frames (override beats the
        # 2-frame reference's snap-to-5s default).
        call = fake_services.ic_lora_pipeline.generate_calls[-1]
        assert call["num_frames"] == 193
        assert call["frame_rate"] == 24
        # The control video is copied to a stable sibling next to the output so
        # the frontend result viewer can surface it (stem convention).
        output = Path(body["videoPath"])
        control_sibling = output.with_name(f"{output.stem}_control.mp4")
        assert control_sibling.exists(), "control video sibling must be copied beside output"

    def test_union_conditioning_cache_is_scoped_to_canonical_duration(
        self,
        client,
        test_state,
        fake_services,
        create_fake_model_files,
        create_fake_ic_lora_files,
    ):
        create_fake_model_files()
        create_fake_ic_lora_files()
        _write_union_checkpoint(test_state)
        test_state.state.app_settings.use_local_text_encoder = True
        video_path = _register_reference_video(test_state)

        def generate(duration: int):
            test_state.video_processor.register_video(
                str(video_path),
                FakeCapture(frames=["frame-a", "frame-b"], width=64, height=64),
            )
            return client.post(
                "/api/lora-inference/generate",
                json={
                    "variant": "union_control",
                    "loraId": "official-ic-lora-union",
                    "request": {
                        "video_path": str(video_path),
                        "conditioning_type": "canny",
                        "prompt": "follow the reference",
                        "images": [],
                        "duration": duration,
                    },
                },
            )

        first = generate(5)
        second = generate(8)

        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        assert [
            call["num_frames"]
            for call in fake_services.ic_lora_pipeline.generate_calls
        ] == [121, 193]

    def test_union_control_rejects_unsupported_duration(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files
    ):
        create_fake_model_files()
        create_fake_ic_lora_files()
        _write_union_checkpoint(test_state)
        test_state.state.app_settings.use_local_text_encoder = True
        video_path = _register_reference_video(test_state)

        resp = client.post(
            "/api/lora-inference/generate",
            json={
                "variant": "union_control",
                "loraId": "official-ic-lora-union",
                "request": {
                    "video_path": str(video_path),
                    "conditioning_type": "canny",
                    "prompt": "follow the reference",
                    "images": [],
                    "duration": 7,
                },
            },
        )
        assert resp.status_code == 400, resp.text

    def test_union_control_preserve_audio_degrades_gracefully_without_audio(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files
    ):
        create_fake_model_files()
        create_fake_ic_lora_files()
        _write_union_checkpoint(test_state)
        test_state.state.app_settings.use_local_text_encoder = True
        video_path = _register_reference_video(test_state)

        resp = client.post(
            "/api/lora-inference/generate",
            json={
                "variant": "union_control",
                "loraId": "official-ic-lora-union",
                "request": {
                    "video_path": str(video_path),
                    "conditioning_type": "canny",
                    "prompt": "follow the reference",
                    "images": [],
                    "duration": 5,
                    "preserve_audio": True,
                },
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "complete"
        # The fake reference has no real audio stream, so the muxer no-ops and
        # returns the original video-only output (no `_audio` suffix). This
        # verifies the preserve-audio path never blocks a successful generation
        # when the reference is silent / unreadable.
        assert "_audio" not in Path(body["videoPath"]).name

    def test_union_control_refine_enables_stage_2(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files
    ):
        create_fake_model_files()
        create_fake_ic_lora_files()
        _write_union_checkpoint(test_state)
        test_state.state.app_settings.use_local_text_encoder = True
        video_path = _register_reference_video(test_state)

        resp = client.post(
            "/api/lora-inference/generate",
            json={
                "variant": "union_control",
                "loraId": "official-ic-lora-union",
                "request": {
                    "video_path": str(video_path),
                    "conditioning_type": "canny",
                    "prompt": "follow the reference",
                    "images": [],
                    "refine": True,
                },
            },
        )
        assert resp.status_code == 200, resp.text
        # refine=True -> stage 2 runs (skip_stage_2=False). Stage 1 still lands
        # at the 960x576 bucket via the 2x target; stage 2 upsamples to 1920x1152.
        call = fake_services.ic_lora_pipeline.generate_calls[-1]
        assert call["skip_stage_2"] is False
        assert call["width"] == 1920
        assert call["height"] == 1152

    def _write_dev_quality_base_overlay(self, test_state) -> None:
        for cp_id in ("ltx-2.3-22b-dev", "ltx-2.3-22b-distilled-lora-384-1.1"):
            path = resolve_model_path(test_state.config.default_models_dir, cp_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x00" * 1024)

    def test_dev_quality_base_loads_dev_checkpoint_with_distilled_lora(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files
    ):
        create_fake_model_files()
        create_fake_ic_lora_files()
        _write_union_checkpoint(test_state)
        test_state.state.app_settings.use_local_text_encoder = True
        test_state.state.app_settings.use_dev_quality_base = True
        self._write_dev_quality_base_overlay(test_state)
        video_path = _register_reference_video(test_state)

        resp = client.post(
            "/api/lora-inference/generate",
            json={
                "variant": "union_control",
                "loraId": "official-ic-lora-union",
                "request": {
                    "video_path": str(video_path),
                    "conditioning_type": "canny",
                    "prompt": "follow the reference",
                    "images": [],
                },
            },
        )
        assert resp.status_code == 200, resp.text
        # With the setting on AND both overlay checkpoints downloaded, the
        # pipeline is built on the dev checkpoint with the distilled v1.1 LoRA
        # stacked @0.5 (the ComfyUI dev + distilled-LoRA flow).
        create_call = fake_services.ic_lora_pipeline.create_calls[-1]
        assert create_call["checkpoint_path"].endswith("ltx-2.3-22b-dev.safetensors")
        assert create_call["distilled_lora_path"] is not None
        assert create_call["distilled_lora_path"].endswith("ltx-2.3-22b-distilled-lora-384-1.1.safetensors")

    def test_dev_quality_base_falls_back_to_distilled_when_overlay_missing(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files
    ):
        create_fake_model_files()
        create_fake_ic_lora_files()
        _write_union_checkpoint(test_state)
        test_state.state.app_settings.use_local_text_encoder = True
        # Setting on but the overlay checkpoints are NOT downloaded: generation
        # must still succeed, falling back to the distilled base with no
        # distilled LoRA (never stacked on an already-distilled checkpoint).
        test_state.state.app_settings.use_dev_quality_base = True
        video_path = _register_reference_video(test_state)

        resp = client.post(
            "/api/lora-inference/generate",
            json={
                "variant": "union_control",
                "loraId": "official-ic-lora-union",
                "request": {
                    "video_path": str(video_path),
                    "conditioning_type": "canny",
                    "prompt": "follow the reference",
                    "images": [],
                },
            },
        )
        assert resp.status_code == 200, resp.text
        create_call = fake_services.ic_lora_pipeline.create_calls[-1]
        assert create_call["checkpoint_path"].endswith("ltx-2.3-22b-distilled.safetensors")
        assert create_call["distilled_lora_path"] is None

    def test_video_input_ic_lora_standardizes_reference_before_inference(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files, tmp_path
    ):
        create_fake_model_files()
        create_fake_ic_lora_files()
        job = _inject_completed_run(
            test_state.lora_training, tmp_path, job_id="gen-ic", dataset_type="ic_lora"
        )
        test_state.state.app_settings.use_local_text_encoder = True
        video_path = _register_reference_video(test_state)

        resp = client.post(
            "/api/lora-inference/generate",
            json={
                "variant": "video_input_ic_lora",
                "loraId": "user-gen-ic",
                "loraScale": 0.35,
                "prompt": "match the reference motion",
                "videoPath": str(video_path),
                "conditioningStrength": 0.7,
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "complete"
        assert Path(body["videoPath"]).exists()

        # The reference is standardized (24fps, AR-bucketed, 8k+1 frames) before
        # reaching the pipeline — the raw imported video is NOT fed directly.
        # This is what keeps an arbitrary imported reference on the adapter's
        # training distribution instead of distorting the output.
        call = fake_services.ic_lora_pipeline.generate_calls[-1]
        assert call["prompt"] == "MYTOK match the reference motion"
        assert len(call["video_conditioning"]) == 1
        standardized_path, strength = call["video_conditioning"][0]
        assert strength == 0.7
        assert standardized_path != str(video_path)
        assert Path(standardized_path).exists()
        assert call["images"] == []
        # 64x64 (square) input buckets to 16:9 -> (960, 576); the pipeline is
        # called with a 2x target -> (1920, 1152) so stage 1 lands at the bucket.
        # 2 frames @ 24fps snaps to a 5s duration -> 121 frames on the 8k+1
        # lattice @ 24fps. refine defaults to false -> stage 2 skipped.
        assert call["width"] == 1920
        assert call["height"] == 1152
        assert call["skip_stage_2"] is True
        assert call["num_frames"] == 121
        assert call["frame_rate"] == 24
        assert job.local_lora_path is not None
        create_call = fake_services.ic_lora_pipeline.create_calls[-1]
        assert create_call["lora_path"] == job.local_lora_path
        assert create_call["lora_scale"] == 0.35

    def test_video_input_ic_lora_scale_change_reloads_adapter(
        self,
        client,
        test_state,
        fake_services,
        create_fake_model_files,
        create_fake_ic_lora_files,
        tmp_path,
    ):
        create_fake_model_files()
        create_fake_ic_lora_files()
        _inject_completed_run(
            test_state.lora_training,
            tmp_path,
            job_id="gen-ic-scale",
            dataset_type="ic_lora",
        )
        test_state.state.app_settings.use_local_text_encoder = True
        video_path = _register_reference_video(test_state)

        def generate(scale: float):
            test_state.video_processor.register_video(
                str(video_path),
                FakeCapture(frames=["frame-a", "frame-b"], width=64, height=64),
            )
            return client.post(
                "/api/lora-inference/generate",
                json={
                    "variant": "video_input_ic_lora",
                    "loraId": "user-gen-ic-scale",
                    "loraScale": scale,
                    "prompt": "match the reference motion",
                    "videoPath": str(video_path),
                },
            )

        first = generate(0.4)
        second = generate(0.9)

        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        assert [
            call["lora_scale"]
            for call in fake_services.ic_lora_pipeline.create_calls
        ] == [0.4, 0.9]

    def test_video_input_ic_lora_reuses_standardized_reference(
        self,
        client,
        test_state,
        fake_services,
        create_fake_model_files,
        create_fake_ic_lora_files,
        tmp_path,
    ):
        create_fake_model_files()
        create_fake_ic_lora_files()
        _inject_completed_run(
            test_state.lora_training,
            tmp_path,
            job_id="ref-cache",
            dataset_type="ic_lora",
        )
        test_state.state.app_settings.use_local_text_encoder = True
        video_path = _register_reference_video(test_state)
        body = {
            "variant": "video_input_ic_lora",
            "loraId": "user-ref-cache",
            "prompt": "apply the effect",
            "videoPath": str(video_path),
        }

        first = client.post("/api/lora-inference/generate", json=body)
        assert first.status_code == 200, first.text
        writers_after_first = len(fake_services.video_processor.writers)

        second = client.post("/api/lora-inference/generate", json=body)
        assert second.status_code == 200, second.text
        assert len(fake_services.video_processor.writers) == writers_after_first
        first_reference = fake_services.ic_lora_pipeline.generate_calls[-2][
            "video_conditioning"
        ][0][0]
        second_reference = fake_services.ic_lora_pipeline.generate_calls[-1][
            "video_conditioning"
        ][0][0]
        assert second_reference == first_reference

    def test_video_input_ic_lora_resolution_override_to_720p(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files, tmp_path
    ):
        create_fake_model_files()
        create_fake_ic_lora_files()
        _inject_completed_run(
            test_state.lora_training, tmp_path, job_id="gen-ic-720", dataset_type="ic_lora"
        )
        test_state.state.app_settings.use_local_text_encoder = True
        video_path = _register_reference_video(test_state)

        resp = client.post(
            "/api/lora-inference/generate",
            json={
                "variant": "video_input_ic_lora",
                "loraId": "user-gen-ic-720",
                "loraScale": 1.0,
                "prompt": "match the reference motion",
                "videoPath": str(video_path),
                "conditioningStrength": 0.7,
                "resolution": "720p",
            },
        )
        assert resp.status_code == 200, resp.text
        # Stage-1 always diffuses at the 540p bucket, so the pipeline gets the
        # fixed 2x target (1920x1152); 720p engages the upsampler
        # (skip_stage_2=False) and is downscaled from there to 1280x768.
        call = fake_services.ic_lora_pipeline.generate_calls[-1]
        assert call["width"] == 1920
        assert call["height"] == 1152
        assert call["skip_stage_2"] is False

    def test_video_input_ic_lora_downscales_oversized_reference(
        self, client, test_state, fake_services, create_fake_model_files, create_fake_ic_lora_files, tmp_path
    ):
        """A 4K reference is downscaled toward the generation target before the
        VAE encode — the dominant cost of a video_input generation. The
        standardized clip is written at the cover-scaled size (never native 4K),
        so the pipeline's internal resize stays a downscale and framing is
        unchanged, but the encode runs on far fewer pixels.
        """
        create_fake_model_files()
        create_fake_ic_lora_files()
        _inject_completed_run(
            test_state.lora_training, tmp_path, job_id="gen-ic-4k", dataset_type="ic_lora"
        )
        test_state.state.app_settings.use_local_text_encoder = True
        video_path = test_state.config.outputs_dir / "ref_4k.mp4"
        video_path.write_bytes(b"\x00" * 100)
        test_state.video_processor.register_video(
            str(video_path), FakeCapture(frames=["a", "b"], width=3840, height=2160)
        )

        resp = client.post(
            "/api/lora-inference/generate",
            json={
                "variant": "video_input_ic_lora",
                "loraId": "user-gen-ic-4k",
                "loraScale": 1.0,
                "prompt": "match the reference motion",
                "videoPath": str(video_path),
                "conditioningStrength": 0.7,
            },
        )
        assert resp.status_code == 200, resp.text
        # The reference conditions stage 1, which runs at the 540p 16:9 bucket
        # (960x576) — NOT the 2x generation-latent target. 3840x2160 cover-scales
        # to the bucket by 0.2667 -> 1024x576 (covers the bucket, so the
        # pipeline's internal resize stays a downscale and framing is unchanged).
        assert test_state.video_processor.resize_calls, "reference was not downscaled"
        assert set(test_state.video_processor.resize_calls) == {(1024, 576)}
        # One resize per standardized frame (2-frame source snaps to 5s -> 121).
        assert len(test_state.video_processor.resize_calls) == 121

    def test_unknown_lora_id_returns_404(self, client, test_state):
        resp = client.post(
            "/api/lora-inference/generate",
            json=_standard_request("user-does-not-exist"),
        )
        assert resp.status_code == 404


# ------------------------------------------------------------------
# Queue integration (kind="lora")
# ------------------------------------------------------------------


class TestLoraQueueDispatch:
    def test_lora_payload_enqueues_and_dispatches(
        self, client, test_state, fake_services, create_fake_model_files, tmp_path
    ):
        create_fake_model_files()
        _inject_completed_run(
            test_state.lora_training, tmp_path, job_id="q-std", dataset_type="standard"
        )
        test_state.state.app_settings.use_local_text_encoder = True

        # Enqueue a kind="lora" payload through the queue route — exercises the
        # widened QueuePayloadApi discriminated union + boundary validation.
        enqueue_resp = client.post(
            "/api/queue/items",
            json={
                "payload": {
                    "kind": "lora",
                    "request": _standard_request("user-q-std", lora_scale=0.5),
                },
                "source": "genspace",
            },
        )
        assert enqueue_resp.status_code == 200, enqueue_resp.text
        assert enqueue_resp.json()["payload"]["kind"] == "lora"

        # Freeze the runner so it doesn't claim the enqueued item out from
        # under the direct dispatch below (the route's enqueue sets the
        # wakeup event, so without this the runner races us to start the
        # generation and the direct dispatch hits "already in progress").
        test_state.queue_runner.stop()

        # Dispatch the same payload shape directly through the composition root
        # (the runner calls this closure on claim). Avoids spinning the runner
        # thread in a unit test while still exercising the kind="lora" branch.
        payload: QueuePayload = LoraQueuePayload(
            request=LoraStandardGenerateRequest(
                loraId="user-q-std",
                loraScale=0.5,
                request=GenerateVideoRequest(prompt="a cinematic shot", resolution="540p"),
            )
        )
        result = test_state._dispatch_queue_payload(payload)
        assert result.status == "complete"
        assert result.output_path and Path(result.output_path).exists()
        assert fake_services.fast_video_pipeline.last_lora_scale == 0.5

    def test_standard_lora_payload_rejected_in_api_mode(
        self, client, test_state, create_fake_model_files, tmp_path
    ):
        create_fake_model_files()
        _inject_completed_run(
            test_state.lora_training, tmp_path, job_id="api-std", dataset_type="standard"
        )
        # Force the API path: a standard LoRA can't run against the API, so the
        # queue boundary validator rejects the enqueue.
        test_state.state.app_settings.ltx_api_key = "test-key"
        test_state.state.app_settings.user_prefers_ltx_api_video_generations = True

        resp = client.post(
            "/api/queue/items",
            json={
                "payload": {
                    "kind": "lora",
                    "request": _standard_request("user-api-std"),
                },
                "source": "genspace",
            },
        )
        assert resp.status_code >= 400


# ------------------------------------------------------------------
# Reference downscale sizing (pure helper)
# ------------------------------------------------------------------


class TestStandardizedWriteSize:
    """`_standardized_write_size` strips resolution the pipeline can't use
    (the video_input encode's dominant cost) while preserving output framing:
    it downscales AR-preserved by the smallest factor that still *covers* the
    generation target, and never upscales a small reference.
    """

    def test_downscales_4k_16_9_to_cover_target(self):
        from handlers.ic_lora_handler import _standardized_write_size

        # 540p 16:9 bucket -> 2x target 1920x1152. 3840x2160 -> 2048x1152.
        assert _standardized_write_size(3840, 2160, 1920, 1152) == (2048, 1152)

    def test_taller_ar_still_covers_target_both_axes(self):
        from handlers.ic_lora_handler import _standardized_write_size

        # 4:3 input (2880x2160) vs 5:3-ish target: scale must cover BOTH axes
        # (max ratio), so the long side isn't starved and the pipeline never
        # has to upscale to cover. cover = max(1920/2880, 1152/2160)=0.6667.
        w, h = _standardized_write_size(2880, 2160, 1920, 1152)
        assert w >= 1920 and h >= 1152
        assert (w, h) == (1920, 1440)

    def test_never_upscales_small_reference(self):
        from handlers.ic_lora_handler import _standardized_write_size

        # A reference already smaller than the target is left as-is (even dims).
        assert _standardized_write_size(64, 64, 1920, 1152) == (64, 64)

    def test_forces_even_dims(self):
        from handlers.ic_lora_handler import _standardized_write_size

        # Odd native dims (with no downscale) round up to even for the encoder.
        assert _standardized_write_size(101, 63, 1920, 1152) == (102, 64)

    def test_degenerate_inputs_fall_back_to_native(self):
        from handlers.ic_lora_handler import _standardized_write_size

        assert _standardized_write_size(0, 0, 1920, 1152) == (0, 0)
        assert _standardized_write_size(1920, 1080, 0, 0) == (1920, 1080)


# ------------------------------------------------------------------
# Resolution -> pipeline plan (pure helper)
# ------------------------------------------------------------------


class TestIcLoraRenderPlan:
    """`_ic_lora_render_plan` keeps stage-1 diffusion pinned to the 540p bucket
    and reaches 720p/1080p via the upsampler (+ downscale), so a high output
    resolution never triggers a native high-res first pass (the VRAM bomb).
    """

    def test_540p_is_native_stage1_no_upsampler(self):
        from handlers.ic_lora_handler import _ic_lora_render_plan

        # 540p: raw stage-1, upsampler skipped, no downscale.
        assert _ic_lora_render_plan("540p", False, "16:9") == (960, 576, False, None)
        assert _ic_lora_render_plan("540p", False, "9:16") == (576, 960, False, None)

    def test_540p_refine_forces_upsampler_to_1080p(self):
        from handlers.ic_lora_handler import _ic_lora_render_plan

        # refine upgrades a 540p request to the x2 (1920x1152) output, no downscale.
        assert _ic_lora_render_plan("540p", True, "16:9") == (960, 576, True, None)

    def test_1080p_upsamples_from_540p_no_downscale(self):
        from handlers.ic_lora_handler import _ic_lora_render_plan

        # Stage-1 stays 540p; the x2 upsampler lands exactly on 1080p (1920x1152).
        assert _ic_lora_render_plan("1080p", False, "16:9") == (960, 576, True, None)
        assert _ic_lora_render_plan("1080p", False, "9:16") == (576, 960, True, None)

    def test_720p_upsamples_then_downscales(self):
        from handlers.ic_lora_handler import _ic_lora_render_plan

        # Stage-1 stays 540p; x2 -> 1920x1152, then downscaled to 720p (1280x768).
        assert _ic_lora_render_plan("720p", False, "16:9") == (960, 576, True, (1280, 768))
        assert _ic_lora_render_plan("720p", False, "9:16") == (576, 960, True, (768, 1280))

    def test_unsupported_resolution_raises(self):
        from handlers.ic_lora_handler import _ic_lora_render_plan
        from _routes._errors import HTTPError

        with pytest.raises(HTTPError) as exc:
            _ic_lora_render_plan("1440p", False, "16:9")
        assert exc.value.status_code == 400

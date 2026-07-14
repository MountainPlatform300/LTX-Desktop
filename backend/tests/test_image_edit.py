"""Integration tests for /api/generate-image-edit (FLUX.2 [klein] 9B)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from api_types import GenerateImageEditRequest, ImageEditQueuePayload
from runtime_config.model_download_specs import resolve_model_path
from state.queue_state import QueuePayload
from tests.http_error_assertions import assert_http_error


def _create_klein_files(test_state) -> Path:
    klein_dir = resolve_model_path(test_state.config.default_models_dir, "flux-2-klein-9b")
    klein_dir.mkdir(parents=True, exist_ok=True)
    (klein_dir / "model.safetensors").write_bytes(b"\x00" * 1024)
    return klein_dir


class TestGenerateImageEdit:
    def test_text_to_image_no_references(self, client, test_state, fake_services):
        _create_klein_files(test_state)

        r = client.post(
            "/api/generate-image-edit",
            json={"prompt": "A red cube on a blue floor", "width": 1024, "height": 1024, "numSteps": 4},
        )

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "complete"
        assert len(data["image_paths"]) == 1
        assert Path(data["image_paths"][0]).exists()
        # txt2img path → generate(), not generate_with_references().
        assert len(fake_services.image_edit_pipeline.generate_calls) == 1
        assert len(fake_services.image_edit_pipeline.generate_with_references_calls) == 0
        assert fake_services.image_edit_pipeline.generate_calls[0]["prompt"] == "A red cube on a blue floor"

    def test_edit_with_reference_image(self, client, test_state, fake_services, make_test_image, tmp_path):
        _create_klein_files(test_state)
        ref_path = tmp_path / "input.png"
        ref_path.write_bytes(make_test_image().getvalue())

        r = client.post(
            "/api/generate-image-edit",
            json={
                "prompt": "make the sky sunset orange",
                "width": 1024,
                "height": 1024,
                "numSteps": 4,
                "referenceImages": [str(ref_path)],
            },
        )

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "complete"
        assert len(data["image_paths"]) == 1
        assert Path(data["image_paths"][0]).exists()
        assert len(fake_services.image_edit_pipeline.generate_with_references_calls) == 1
        assert len(fake_services.image_edit_pipeline.generate_calls) == 0
        call = fake_services.image_edit_pipeline.generate_with_references_calls[0]
        assert call["prompt"] == "make the sky sunset orange"
        assert len(call["reference_images"]) == 1

    def test_not_downloaded_returns_409(self, client, test_state):
        # No Klein files created → load_klein_to_gpu raises 409.
        r = client.post("/api/generate-image-edit", json={"prompt": "A cat"})
        assert_http_error(
            r,
            status_code=409,
            code="KLEIN_NOT_DOWNLOADED",
            message="FLUX.2 Klein 9B isn't downloaded. Download it from the Model Status menu.",
        )

    def test_force_api_returns_501(self, client, test_state, fake_services):
        _create_klein_files(test_state)
        # `force_api_generations` is derived from local_generations_mode.
        test_state.config.local_generations_mode = "unsupported"

        r = client.post("/api/generate-image-edit", json={"prompt": "A cat"})
        assert_http_error(
            r,
            status_code=501,
            code="KLEIN_UNAVAILABLE",
            message="Local image editing isn't available while generations are forced to the API.",
        )
        # No GPU work should have happened.
        assert len(fake_services.image_edit_pipeline.generate_calls) == 0

    def test_cancelled(self, client, test_state, fake_services):
        _create_klein_files(test_state)
        fake_services.image_edit_pipeline.raise_on_generate = RuntimeError("cancelled")

        r = client.post("/api/generate-image-edit", json={"prompt": "A cat"})
        assert r.status_code == 200
        assert r.json()["status"] == "cancelled"

    def test_dimension_clamping(self, client, test_state, fake_services):
        _create_klein_files(test_state)
        r = client.post(
            "/api/generate-image-edit",
            json={"prompt": "test", "width": 1023, "height": 1023},
        )
        assert r.status_code == 200
        call = fake_services.image_edit_pipeline.generate_calls[0]
        assert call["width"] == 1008
        assert call["height"] == 1008

    def test_high_resolution_generates_natively_then_upscales(
        self, client, test_state, fake_services
    ):
        _create_klein_files(test_state)
        r = client.post(
            "/api/generate-image-edit",
            json={"prompt": "test", "width": 3641, "height": 2048},
        )
        assert r.status_code == 200
        call = fake_services.image_edit_pipeline.generate_calls[0]
        assert call["width"] == 1024
        assert call["height"] == 576
        assert call["guidance_scale"] == 1.0

        output = Image.open(r.json()["image_paths"][0])
        assert output.size == (3632, 2048)


class TestKleinInImageCatalog:
    def test_klein_listed_as_available_and_gated(self, client):
        r = client.get("/api/generate/image-models-specs")
        assert r.status_code == 200
        by_id = {m["id"]: m for m in r.json()["models"]}
        assert "flux-2-klein-9b" in by_id
        assert by_id["flux-2-klein-9b"]["inference_status"] == "available"
        assert by_id["flux-2-klein-9b"]["gated"] is True
        assert by_id["flux-2-klein-9b"]["is_edit_model"] is True
        # Z-Image is the txt2img default, not an edit model.
        assert by_id["z-image-turbo"]["is_edit_model"] is False

    def test_klein_rejected_by_txt2img_endpoint(self, client, create_fake_model_files):
        create_fake_model_files(include_zit=True)
        r = client.post(
            "/api/generate-image",
            json={"prompt": "A cat", "model": "flux-2-klein-9b"},
        )
        assert_http_error(
            r,
            status_code=400,
            code="KLEIN_USE_EDIT_ENDPOINT",
            message="FLUX.2 [klein] 9B uses the image-edit endpoint (/api/generate-image-edit).",
        )


class TestKleinQueueDispatch:
    """Klein image edits route through the durable queue (kind 'image_edit')
    so the Generate button doesn't grey out / block and the edit shows in the
    queue panel. The runner dispatches via `AppHandler._dispatch_queue_payload`,
    which calls `ImageEditHandler.generate` and normalizes the response.
    """

    def test_enqueue_accepts_image_edit_payload(self, client, test_state):
        _create_klein_files(test_state)
        r = client.post(
            "/api/queue/items",
            json={
                "payload": {
                    "kind": "image_edit",
                    "request": {
                        "prompt": "a klein edit via the queue",
                        "width": 1024,
                        "height": 1024,
                        "numSteps": 4,
                        "numImages": 1,
                        "referenceImages": [],
                    },
                },
                "source": "genspace",
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["payload"]["kind"] == "image_edit"

    def test_dispatch_image_edit_payload_completes(
        self, client, test_state, fake_services
    ):
        _create_klein_files(test_state)
        # Freeze the runner so it doesn't claim a concurrent item out from
        # under the direct dispatch below.
        test_state.queue_runner.stop()

        payload: QueuePayload = ImageEditQueuePayload(
            request=GenerateImageEditRequest(
                prompt="a klein edit via the queue",
                width=1024,
                height=1024,
                numSteps=4,
                numImages=1,
                referenceImages=[],
            )
        )
        result = test_state._dispatch_queue_payload(payload)
        assert result.status == "complete"
        assert result.output_path and Path(result.output_path).exists()
        # Routed to the Klein pipeline's txt2img path (no references).
        assert len(fake_services.image_edit_pipeline.generate_calls) == 1
        assert len(fake_services.image_edit_pipeline.generate_with_references_calls) == 0

    def test_dispatch_image_edit_busy_on_in_progress(
        self, client, test_state, fake_services
    ):
        _create_klein_files(test_state)
        test_state.queue_runner.stop()
        # Occupy the single-flight generation slot so the edit handler raises
        # HTTPError(409) "Generation already in progress", which the dispatch
        # maps to `busy` (no retry consumed) instead of `failed`.
        from state.app_state_types import GpuSlot, VideoPipelineState
        from tests.fakes.services import FakeFastVideoPipeline

        test_state.state.gpu_slot = GpuSlot(
            active_pipeline=VideoPipelineState(
                pipeline=FakeFastVideoPipeline(),
                is_compiled=False,
            )
        )
        test_state.generation.start_generation("busy")
        try:
            payload: QueuePayload = ImageEditQueuePayload(
                request=GenerateImageEditRequest(prompt="a busy klein edit")
            )
            result = test_state._dispatch_queue_payload(payload)
            assert result.status == "busy"
        finally:
            test_state.generation.fail_generation("test cleanup")

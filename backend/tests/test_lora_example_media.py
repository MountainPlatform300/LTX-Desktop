"""Tests for LoRA Library example media (CivitAI-style "what does this LoRA do?").

Covers attach / replace / clear for both imported and user-trained LoRAs through
the real FastAPI app: an example image or video the user picks is copied into
app storage, surfaced as `exampleMediaType` on the registry entry, served back
through the secure `example-media` route, and removable. Official union LoRAs
can't have an example. Uses fake services (no mocks) per the backend boundary.
"""

from __future__ import annotations

from pathlib import Path

from api_types import LoraInferenceRegistryResponseApi
from state.lora_training_state import (
    LoraClip,
    LoraDataset,
    PreprocessedDataset,
    TrainingConfig,
    TrainingJob,
)


def _import(client, *, source_path: str, name: str) -> str:
    resp = client.post(
        "/api/lora-inference/import",
        json={"sourcePath": source_path, "name": name, "variant": "standard"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["entry"]["id"]


def _inject_completed_run(handler, tmp_path: Path, *, job_id: str) -> TrainingJob:
    clip = tmp_path / f"{job_id}-clip.mp4"
    clip.write_bytes(b"\x00\x01")
    weights = tmp_path / f"{job_id}.safetensors"
    weights.write_bytes(b"\x00" * 8)
    ds = LoraDataset(
        id=f"ds-{job_id}",
        name=f"Dataset {job_id}",
        created_at="2026-01-01T00:00:00Z",
        status="uploaded",
        trigger_word="MYTOK",
        remote_dataset_dir=f"/w/datasets/{job_id}",
        type="standard",
        clips=[LoraClip(id="c1", local_path=str(clip), caption="a cat", duration_seconds=3.0)],
    )
    pre = PreprocessedDataset(
        id=f"pre-{job_id}",
        dataset_id=ds.id,
        created_at="2026-01-01T00:00:00Z",
        status="ready",
        resolution_buckets="768x448x89",
        remote_precomputed_dir=f"/w/.precomputed-{job_id}",
    )
    job = TrainingJob(
        id=job_id,
        preprocessed_id=pre.id,
        name=f"Run {job_id}",
        created_at="2026-01-01T00:00:00Z",
        status="completed",
        config=TrainingConfig(rank=64),
        provider="runpod",
        local_lora_path=str(weights),
    )
    handler._datasets.datasets.append(ds)
    handler._preprocessed.items.append(pre)
    handler._training.items.append(job)
    return job


def _entry(client, lora_id: str) -> dict:
    reg = client.get("/api/lora-inference/registry")
    assert reg.status_code == 200, reg.text
    parsed = LoraInferenceRegistryResponseApi.model_validate(reg.json())
    for e in parsed.entries:
        if e.id == lora_id:
            return e.model_dump(mode="json")
    raise AssertionError(f"entry {lora_id} not in registry")


class TestImportedExample:
    def test_attach_image_surfaces_media_type_and_serves_bytes(
        self, client, test_state, tmp_path
    ):
        src = tmp_path / "external.safetensors"
        src.write_bytes(b"\x00" * 64)
        lora_id = _import(client, source_path=str(src), name="Style")

        img = tmp_path / "preview.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

        resp = client.post(
            f"/api/lora-inference/entries/{lora_id}/example",
            json={"sourcePath": str(img)},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["entry"]["exampleMediaType"] == "image"

        assert _entry(client, lora_id)["exampleMediaType"] == "image"

        media = client.get(f"/api/lora-inference/entries/{lora_id}/example-media")
        assert media.status_code == 200
        assert media.content.startswith(b"\x89PNG")

    def test_replace_image_with_video_swaps_media_type(self, client, test_state, tmp_path):
        src = tmp_path / "external.safetensors"
        src.write_bytes(b"\x00" * 64)
        lora_id = _import(client, source_path=str(src), name="Style")

        img = tmp_path / "p.png"
        img.write_bytes(b"\x00" * 16)
        client.post(
            f"/api/lora-inference/entries/{lora_id}/example",
            json={"sourcePath": str(img)},
        )
        assert _entry(client, lora_id)["exampleMediaType"] == "image"

        vid = tmp_path / "p.mp4"
        original_video = b"\x00\x00\x00\x18ftyp" + b"\x00" * 32
        vid.write_bytes(original_video)
        resp = client.post(
            f"/api/lora-inference/entries/{lora_id}/example",
            json={"sourcePath": str(vid)},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["entry"]["exampleMediaType"] == "video"
        served = client.get(f"/api/lora-inference/entries/{lora_id}/example-media")
        assert served.status_code == 200
        assert served.content == original_video
        # Only one example file lives on disk after a replace.
        assert _entry(client, lora_id)["exampleMediaType"] == "video"

    def test_clear_removes_example_and_media_route_404s(self, client, test_state, tmp_path):
        src = tmp_path / "external.safetensors"
        src.write_bytes(b"\x00" * 64)
        lora_id = _import(client, source_path=str(src), name="Style")
        img = tmp_path / "p.png"
        img.write_bytes(b"\x00" * 16)
        client.post(
            f"/api/lora-inference/entries/{lora_id}/example",
            json={"sourcePath": str(img)},
        )

        resp = client.delete(f"/api/lora-inference/entries/{lora_id}/example")
        assert resp.status_code == 204
        assert _entry(client, lora_id)["exampleMediaType"] is None

        media = client.get(f"/api/lora-inference/entries/{lora_id}/example-media")
        assert media.status_code == 404

    def test_unsupported_type_returns_400(self, client, test_state, tmp_path):
        src = tmp_path / "external.safetensors"
        src.write_bytes(b"\x00" * 64)
        lora_id = _import(client, source_path=str(src), name="Style")
        bad = tmp_path / "notes.txt"
        bad.write_text("not media")
        resp = client.post(
            f"/api/lora-inference/entries/{lora_id}/example",
            json={"sourcePath": str(bad)},
        )
        assert resp.status_code == 400


class TestTrainedExample:
    def test_attach_and_clear_example_on_trained_lora(
        self, client, test_state, tmp_path
    ):
        job = _inject_completed_run(test_state.lora_training, tmp_path, job_id="ex-1")
        lora_id = f"user-{job.id}"

        img = tmp_path / "trained.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)

        resp = client.post(
            f"/api/lora-inference/entries/{lora_id}/example",
            json={"sourcePath": str(img)},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["entry"]["exampleMediaType"] == "image"
        assert _entry(client, lora_id)["exampleMediaType"] == "image"

        media = client.get(f"/api/lora-inference/entries/{lora_id}/example-media")
        assert media.status_code == 200
        assert media.content.startswith(b"\x89PNG")

        cleared = client.delete(f"/api/lora-inference/entries/{lora_id}/example")
        assert cleared.status_code == 204
        assert _entry(client, lora_id)["exampleMediaType"] is None


class TestOfficialExample:
    def test_official_union_cannot_have_example(self, client, test_state, tmp_path):
        img = tmp_path / "p.png"
        img.write_bytes(b"\x00" * 16)
        resp = client.post(
            "/api/lora-inference/entries/official-ic-lora-union/example",
            json={"sourcePath": str(img)},
        )
        assert resp.status_code == 400

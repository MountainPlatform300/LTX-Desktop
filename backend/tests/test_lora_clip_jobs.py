"""Tests for the local clip-prep job ledger + runner (sprite generation).

The runner uses the fake clip-processor (no real ffmpeg). We drive a
single dispatch pass synchronously via `reconcile_once()` instead of
the background thread so assertions are deterministic.
"""

from __future__ import annotations

from pathlib import Path

from services.clip_processor.clip_processor import ClipProcessorError


def test_enqueue_creates_pending_jobs(test_state):
    jobs = test_state.lora_training.enqueue_clip_jobs(
        source_paths=["/clips/a.mp4", "/clips/b.mp4"], kind="sprite"
    )
    assert [j.source_path for j in jobs] == ["/clips/a.mp4", "/clips/b.mp4"]
    assert all(j.status == "pending" for j in jobs)
    assert all(j.kind == "sprite" for j in jobs)


def test_enqueue_is_idempotent_per_source_and_kind(test_state):
    first = test_state.lora_training.enqueue_clip_jobs(
        source_paths=["/clips/a.mp4"], kind="sprite"
    )
    again = test_state.lora_training.enqueue_clip_jobs(
        source_paths=["/clips/a.mp4"], kind="sprite"
    )
    assert first[0].id == again[0].id
    state = test_state.lora_training.get_clip_jobs_state()
    assert len(state.jobs) == 1


def test_runner_completes_sprite_job(test_state, fake_services):
    test_state.lora_training.enqueue_clip_jobs(
        source_paths=["/clips/a.mp4"], kind="sprite"
    )
    test_state.lora_clip_jobs_runner.reconcile_once()

    state = test_state.lora_training.get_clip_jobs_state()
    assert len(state.jobs) == 1
    job = state.jobs[0]
    assert job.status == "completed"
    assert job.sprite_path is not None and Path(job.sprite_path).exists()
    assert job.poster_path is not None and Path(job.poster_path).exists()
    assert job.sprite_tiles == fake_services.clip_processor.sprite_calls[0]["tile_count"]
    # Sprite + poster live under the app data thumbs dir.
    assert "thumbs" in job.sprite_path


def test_set_poster_publishes_early_without_completing(test_state):
    # The runner reports the poster mid-flight so the gallery can drop its
    # spinner before the (slow) sprite filmstrip finishes.
    jobs = test_state.lora_training.enqueue_clip_jobs(
        source_paths=["/clips/a.mp4"], kind="sprite"
    )
    test_state.lora_training.claim_pending_clip_jobs()
    test_state.lora_training.set_clip_job_poster(jobs[0].id, poster_path="/thumbs/a.png")

    job = test_state.lora_training.get_clip_jobs_state().jobs[0]
    assert job.poster_path == "/thumbs/a.png"
    assert job.status == "running"
    assert job.sprite_path is None


def test_runner_marks_job_failed_on_processor_error(test_state, fake_services):
    fake_services.clip_processor.error = ClipProcessorError("boom", status_code=400)
    test_state.lora_training.enqueue_clip_jobs(
        source_paths=["/clips/a.mp4"], kind="sprite"
    )
    test_state.lora_clip_jobs_runner.reconcile_once()

    job = test_state.lora_training.get_clip_jobs_state().jobs[0]
    assert job.status == "failed"
    assert job.error == "boom"


def test_recovery_resets_running_jobs(test_state):
    # Simulate a crash mid-run: claim marks the job running, then we
    # reload the ledger which should reset it to pending.
    test_state.lora_training.enqueue_clip_jobs(
        source_paths=["/clips/a.mp4"], kind="sprite"
    )
    test_state.lora_training.claim_pending_clip_jobs()
    assert test_state.lora_training.get_clip_jobs_state().jobs[0].status == "running"

    test_state.lora_training.load_state()
    assert test_state.lora_training.get_clip_jobs_state().jobs[0].status == "pending"


def test_enqueue_endpoint_returns_jobs(client):
    res = client.post(
        "/api/lora/clip-jobs",
        json={"sourcePaths": ["/clips/a.mp4", "/clips/b.mp4"]},
    )
    assert res.status_code == 200
    body = res.json()
    paths = {j["sourcePath"] for j in body["jobs"]}
    assert paths == {"/clips/a.mp4", "/clips/b.mp4"}
    assert all(j["kind"] == "sprite" for j in body["jobs"])

    listed = client.get("/api/lora/clip-jobs")
    assert listed.status_code == 200
    assert len(listed.json()["jobs"]) >= 2

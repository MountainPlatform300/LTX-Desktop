"""Tests for the target/variant derivation job ledger + runner.

The runner's Kling drive is fully remote (fake video-restyler), so we can
drive it end-to-end via `reconcile_once()` synchronously. The local
IC-LoRA drive needs GPU/models and is exercised only at the handler state
level (transitions, cancel, retry, recovery).
"""

from __future__ import annotations

from pathlib import Path

from api_types import CreateLoraDerivationJobRequest
from handlers.lora_derivation_runner import _is_transient
from services.video_restyler.video_restyler import VideoRestylerError
from state.lora_derivation_jobs_state import derivation_jobs_state_to_api


def _kling_req(driver_path: str, **overrides) -> CreateLoraDerivationJobRequest:
    base = dict(
        driverPath=driver_path,
        referencePath=driver_path,
        sourceClipId="clip-1",
        editPrompt="remove the person on the left",
        scenePrompt="a quiet street",
        engine="kling",
        caption="trigger, a quiet street",
        label="Target for clip-1",
    )
    base.update(overrides)
    return CreateLoraDerivationJobRequest(**base)


def _review_req(driver_path: str, **overrides) -> CreateLoraDerivationJobRequest:
    return _kling_req(driver_path, requireReview=True, **overrides)


def _frame_edit_req(driver_path: str, **overrides) -> CreateLoraDerivationJobRequest:
    """Edit-only job (frame-edit modal): direction `frame_edit`, no animate."""
    base = dict(
        driverPath=driver_path,
        sourceClipId="clip-1",
        editPrompt="change to winter",
        editEngine="fal",
        engine="kling",  # unused for frame_edit (no animate step)
        direction="frame_edit",
        frameEdited=True,
        caption="trigger, change to winter",
        label="Edited still for clip-1",
    )
    base.update(overrides)
    return CreateLoraDerivationJobRequest(**base)


def _make_driver(tmp_path: Path) -> str:
    driver = tmp_path / "driver.mp4"
    driver.write_bytes(b"fake-driver-mp4")
    return str(driver)


# ---------------------------------------------------------------------------
# Handler state transitions
# ---------------------------------------------------------------------------


def test_enqueue_creates_pending_job(test_state, tmp_path):
    job = test_state.lora_training.enqueue_derivation_job(_kling_req(_make_driver(tmp_path)))
    assert job.status == "pending"
    assert job.engine == "kling"
    # Default direction generates the target (legacy behavior).
    assert job.direction == "target"
    state = test_state.lora_training.get_derivation_jobs_state()
    assert len(state.jobs) == 1


def test_direction_reference_persists_and_serializes(test_state, tmp_path):
    job = test_state.lora_training.enqueue_derivation_job(
        _kling_req(_make_driver(tmp_path), direction="reference")
    )
    assert job.direction == "reference"
    # Survives the ledger round-trip and reaches the API model the frontend
    # reads when folding the finished clip back in.
    api = derivation_jobs_state_to_api(test_state.lora_training.get_derivation_jobs_state())
    assert api.jobs[0].direction == "reference"


def test_direction_frame_edit_persists_and_serializes(test_state, tmp_path):
    job = test_state.lora_training.enqueue_derivation_job(
        _frame_edit_req(_make_driver(tmp_path))
    )
    assert job.direction == "frame_edit"
    assert job.require_review is False
    api = derivation_jobs_state_to_api(test_state.lora_training.get_derivation_jobs_state())
    assert api.jobs[0].direction == "frame_edit"


def test_claim_marks_in_flight_and_serial(test_state, tmp_path):
    d = _make_driver(tmp_path)
    test_state.lora_training.enqueue_derivation_job(_kling_req(d))
    test_state.lora_training.enqueue_derivation_job(_kling_req(d))
    first = test_state.lora_training.claim_next_derivation_job()
    assert first is not None and first.status == "editing"
    second = test_state.lora_training.claim_next_derivation_job()
    # One still pending (the other), so a second claim returns it...
    assert second is not None and second.id != first.id
    # ...and a third claim finds nothing pending.
    assert test_state.lora_training.claim_next_derivation_job() is None


def test_claim_serializes_klein_edits_one_in_flight(test_state, tmp_path):
    """Local GPU (Klein) edits must be claimed one at a time, regardless of
    the Fal concurrency setting, so the ~32GB pipeline isn't loaded by
    several workers at once (which crashes the backend and strands
    cancel-all). With one Klein job in flight, a second Klein job stays
    pending; Fal jobs are unaffected.
    """
    d = _make_driver(tmp_path)
    test_state.lora_training.enqueue_derivation_job(
        _frame_edit_req(d, editEngine="klein")
    )
    test_state.lora_training.enqueue_derivation_job(
        _frame_edit_req(d, editEngine="klein")
    )
    first = test_state.lora_training.claim_next_derivation_job()
    assert first is not None and first.status == "editing"
    # Second Klein edit must NOT be claimed while the first is in flight.
    assert test_state.lora_training.claim_next_derivation_job() is None
    # The second job is still pending, so cancel-all drops it immediately
    # (this is the visible "cancel all works" behavior).
    affected = test_state.lora_training.cancel_all_derivation_jobs()
    assert affected == 2  # the in-flight (flagged) + the pending (cancelled)
    statuses = [j.status for j in test_state.lora_training.get_derivation_jobs_state().jobs]
    assert statuses.count("cancelled") == 1  # the pending one
    assert statuses.count("editing") == 1  # the in-flight one (flagged, not yet failed)


def test_claim_does_not_serialize_fal_edits(test_state, tmp_path):
    """Fal (remote) edits are NOT serialized by the Klein guard — the Fal
    concurrency limit still claims several at once."""
    d = _make_driver(tmp_path)
    test_state.lora_training.enqueue_derivation_job(_frame_edit_req(d, editEngine="fal"))
    test_state.lora_training.enqueue_derivation_job(_frame_edit_req(d, editEngine="fal"))
    first = test_state.lora_training.claim_next_derivation_job()
    second = test_state.lora_training.claim_next_derivation_job()
    assert first is not None and second is not None and first.id != second.id


def test_cancel_pending_marks_cancelled(test_state, tmp_path):
    job = test_state.lora_training.enqueue_derivation_job(_kling_req(_make_driver(tmp_path)))
    test_state.lora_training.cancel_derivation_job(job.id)
    cancelled = test_state.lora_training.get_derivation_jobs_state().jobs[0]
    assert cancelled.status == "cancelled"
    assert cancelled.cancel_requested is False


def test_cancel_all_aborts_active_jobs(test_state, tmp_path):
    d = _make_driver(tmp_path)
    # Two pending (queued) + one in-flight (claimed -> editing).
    test_state.lora_training.enqueue_derivation_job(_kling_req(d))
    test_state.lora_training.enqueue_derivation_job(_kling_req(d))
    test_state.lora_training.enqueue_derivation_job(_kling_req(d))
    claimed = test_state.lora_training.claim_next_derivation_job()
    assert claimed is not None and claimed.status == "editing"

    affected = test_state.lora_training.cancel_all_derivation_jobs()
    assert affected == 3
    jobs = test_state.lora_training.get_derivation_jobs_state().jobs
    # Queued jobs cancel outright; the in-flight one is flagged for the runner.
    statuses = {j.status for j in jobs}
    assert statuses <= {"cancelled", "editing"}
    in_flight = next(j for j in jobs if j.id == claimed.id)
    assert in_flight.cancel_requested is True


def test_cancel_all_scoped_to_dataset(test_state, tmp_path):
    d = _make_driver(tmp_path)
    a = test_state.lora_training.enqueue_derivation_job(_kling_req(d, datasetId="ds-a"))
    b = test_state.lora_training.enqueue_derivation_job(_kling_req(d, datasetId="ds-b"))

    affected = test_state.lora_training.cancel_all_derivation_jobs(dataset_id="ds-a")
    assert affected == 1
    by_id = {j.id: j for j in test_state.lora_training.get_derivation_jobs_state().jobs}
    assert by_id[a.id].status == "cancelled"
    assert by_id[b.id].status == "pending"


def test_cancel_all_route(client, test_state, tmp_path):
    test_state.lora_derivation_runner.stop()
    d = _make_driver(tmp_path)
    test_state.lora_training.enqueue_derivation_job(_kling_req(d))
    test_state.lora_training.enqueue_derivation_job(_kling_req(d))

    res = client.post("/api/lora/derivations/cancel-all", json={})
    assert res.status_code == 200, res.text
    assert all(j["status"] == "cancelled" for j in res.json()["jobs"])


def test_cancel_in_flight_flags_then_fail_reports_cancelled(test_state, tmp_path):
    job = test_state.lora_training.enqueue_derivation_job(_kling_req(_make_driver(tmp_path)))
    test_state.lora_training.claim_next_derivation_job()
    test_state.lora_training.cancel_derivation_job(job.id)
    assert test_state.lora_training.is_derivation_cancelled(job.id)
    # The runner reports a mid-flight cancel as a failure; the handler maps
    # it back to "cancelled" because the flag is set.
    test_state.lora_training.fail_derivation_job(job.id, "Cancelled")
    cancelled = test_state.lora_training.get_derivation_jobs_state().jobs[0]
    assert cancelled.status == "cancelled"
    assert cancelled.cancel_requested is False


def test_cancel_requested_wins_over_late_completion(test_state, fake_services, tmp_path):
    job = test_state.lora_training.enqueue_derivation_job(_kling_req(_make_driver(tmp_path)))
    test_state.lora_training.claim_next_derivation_job()
    test_state.lora_training.cancel_derivation_job(job.id)

    test_state.lora_training.complete_derivation_job(
        job.id,
        derived_path=str(tmp_path / "done.mp4"),
        probe=fake_services.clip_processor.result,
    )

    cancelled = test_state.lora_training.get_derivation_jobs_state().jobs[0]
    assert cancelled.status == "cancelled"
    assert cancelled.cancel_requested is False


def test_terminal_derivation_is_not_revived_by_late_worker_callbacks(
    test_state, fake_services, tmp_path
):
    job = test_state.lora_training.enqueue_derivation_job(
        _kling_req(_make_driver(tmp_path))
    )
    test_state.lora_training.cancel_derivation_job(job.id)

    test_state.lora_training.complete_derivation_job(
        job.id,
        derived_path=str(tmp_path / "late.mp4"),
        probe=fake_services.clip_processor.result,
    )
    test_state.lora_training.fail_derivation_job(job.id, "late failure")

    terminal = test_state.lora_training.get_derivation_jobs_state().jobs[0]
    assert terminal.status == "cancelled"
    assert terminal.derived_path is None
    assert terminal.error is None


def test_retry_resets_failed_job(test_state, tmp_path):
    job = test_state.lora_training.enqueue_derivation_job(_kling_req(_make_driver(tmp_path)))
    test_state.lora_training.claim_next_derivation_job()
    test_state.lora_training.fail_derivation_job(job.id, "boom")
    assert test_state.lora_training.get_derivation_jobs_state().jobs[0].status == "failed"
    test_state.lora_training.retry_derivation_job(job.id)
    j = test_state.lora_training.get_derivation_jobs_state().jobs[0]
    assert j.status == "pending" and j.error is None
    assert j.edited_frame_path is None


def test_dismiss_removes_terminal_job(test_state, tmp_path):
    job = test_state.lora_training.enqueue_derivation_job(_kling_req(_make_driver(tmp_path)))
    test_state.lora_training.cancel_derivation_job(job.id)
    test_state.lora_training.dismiss_derivation_job(job.id)
    assert test_state.lora_training.get_derivation_jobs_state().jobs == []


def test_recovery_resets_in_flight_jobs(test_state, tmp_path):
    test_state.lora_training.enqueue_derivation_job(_kling_req(_make_driver(tmp_path)))
    test_state.lora_training.claim_next_derivation_job()
    assert test_state.lora_training.get_derivation_jobs_state().jobs[0].status == "editing"
    test_state.lora_training.load_state()
    assert test_state.lora_training.get_derivation_jobs_state().jobs[0].status == "pending"


# ---------------------------------------------------------------------------
# Runner (Kling drive, fully faked)
# ---------------------------------------------------------------------------


def test_runner_completes_kling_job(test_state, fake_services, tmp_path):
    driver = _make_driver(tmp_path)
    test_state.lora_training.enqueue_derivation_job(_kling_req(driver))
    test_state.lora_derivation_runner.reconcile_once()

    job = test_state.lora_training.get_derivation_jobs_state().jobs[0]
    assert job.status == "completed", job.error
    assert job.derived_path is not None and Path(job.derived_path).exists()
    assert job.probe is not None and job.probe.width == 1280
    # The edited frame was produced (edit prompt present) and recorded.
    assert job.edited_frame_path is not None and Path(job.edited_frame_path).exists()
    # Kling drive went through motion_transfer with the requested orientation.
    assert len(fake_services.video_restyler.motion_transfer_calls) == 1


def test_runner_completes_kling_o3_job_with_edited_reference(
    test_state, fake_services, tmp_path
):
    driver = _make_driver(tmp_path)
    # An edit prompt is present, so the edited still rides along as @Image1.
    test_state.lora_training.enqueue_derivation_job(
        _kling_req(driver, engine="kling_o3", keepAudio=False, frameEdited=True)
    )
    test_state.lora_derivation_runner.reconcile_once()

    job = test_state.lora_training.get_derivation_jobs_state().jobs[0]
    assert job.status == "completed", job.error
    assert job.derived_path is not None and Path(job.derived_path).exists()
    # The O3 v2v-edit path went through kling_v2v_edit (not motion_transfer),
    # forwarding the prompt, the edited still as the appearance reference, and
    # the keep-audio flag.
    assert len(fake_services.video_restyler.motion_transfer_calls) == 0
    calls = fake_services.video_restyler.kling_v2v_edit_calls
    assert len(calls) == 1
    assert calls[0]["prompt"] == "a quiet street"
    assert calls[0]["keep_audio"] is False
    assert calls[0]["image_size"] is not None


def test_runner_kling_o3_without_edit_sends_no_image(
    test_state, fake_services, tmp_path
):
    driver = _make_driver(tmp_path)
    # No edit (blank prompt, frameEdited False) => pure video + prompt edit.
    test_state.lora_training.enqueue_derivation_job(
        _kling_req(driver, engine="kling_o3", editPrompt="", frameEdited=False)
    )
    test_state.lora_derivation_runner.reconcile_once()

    job = test_state.lora_training.get_derivation_jobs_state().jobs[0]
    assert job.status == "completed", job.error
    calls = fake_services.video_restyler.kling_v2v_edit_calls
    assert len(calls) == 1
    # No appearance reference image was uploaded for the pure prompt edit.
    assert calls[0]["image_size"] is None


def test_runner_skips_edit_when_no_prompt(test_state, fake_services, tmp_path):
    driver = _make_driver(tmp_path)
    # A pre-existing still as the anchor, no edit prompt => no image edit.
    still = tmp_path / "still.png"
    still.write_bytes(b"fake-still-png")
    test_state.lora_training.enqueue_derivation_job(
        _kling_req(driver, editPrompt="", framePath=str(still))
    )
    test_state.lora_derivation_runner.reconcile_once()

    job = test_state.lora_training.get_derivation_jobs_state().jobs[0]
    assert job.status == "completed", job.error
    assert len(fake_services.image_editor.calls) == 0
    assert job.edited_frame_path is None


def test_runner_edit_stage_uses_klein_pipeline(test_state, fake_services, tmp_path, make_test_image):
    from runtime_config.model_download_specs import resolve_model_path

    # Klein checkpoint must be present for load_klein_to_gpu.
    klein_dir = resolve_model_path(test_state.config.default_models_dir, "flux-2-klein-9b")
    klein_dir.mkdir(parents=True, exist_ok=True)
    (klein_dir / "model.safetensors").write_bytes(b"\x00" * 1024)
    # The fake clip processor returns dummy bytes; swap in a real PNG so the
    # Klein pipeline can load the extracted frame as a reference image.
    real_png = make_test_image().getvalue()
    fake_services.clip_processor.extract_frame = lambda *, video_path, time_seconds: real_png  # type: ignore[assignment]

    test_state.lora_training.enqueue_derivation_job(
        _kling_req(_make_driver(tmp_path), editEngine="klein")
    )
    test_state.lora_derivation_runner.reconcile_once()

    job = test_state.lora_training.get_derivation_jobs_state().jobs[0]
    assert job.status == "completed", job.error
    # The edit stage ran through the local Klein pipeline (reference edit),
    # and the remote Fal/Nano Banana editor was not touched.
    assert len(fake_services.image_edit_pipeline.generate_with_references_calls) == 1
    assert len(fake_services.image_edit_pipeline.generate_calls) == 0
    assert fake_services.image_editor.calls == []
    assert job.edited_frame_path is not None


def test_klein_dimensions_preserve_aspect_ratio(tmp_path, make_test_image):
    from handlers.lora_training_handler import LoraTrainingHandler

    # Portrait source frame -> portrait output, longest side capped at 1024,
    # both sides a multiple of 16.
    portrait = tmp_path / "portrait.png"
    portrait.write_bytes(make_test_image(w=720, h=1280).getvalue())
    w, h = LoraTrainingHandler._klein_dimensions_for(str(portrait))
    assert w < h
    assert max(w, h) <= 1024
    assert w % 16 == 0 and h % 16 == 0
    # 720x1280 scales to 576x1024 (longest=1024, rounded to 16).
    assert (w, h) == (576, 1024)

    # Landscape source frame -> landscape output.
    landscape = tmp_path / "landscape.png"
    landscape.write_bytes(make_test_image(w=1920, h=1080).getvalue())
    w, h = LoraTrainingHandler._klein_dimensions_for(str(landscape))
    assert w > h
    assert max(w, h) <= 1024
    assert w % 16 == 0 and h % 16 == 0

    # Missing/unreadable frame falls back to 1024x1024.
    assert LoraTrainingHandler._klein_dimensions_for(str(tmp_path / "nope.png")) == (1024, 1024)


def test_runner_marks_failed_on_restyler_error(test_state, fake_services, tmp_path):
    fake_services.video_restyler.error = VideoRestylerError("fal exploded")
    test_state.lora_training.enqueue_derivation_job(_kling_req(_make_driver(tmp_path)))
    test_state.lora_derivation_runner.reconcile_once()

    job = test_state.lora_training.get_derivation_jobs_state().jobs[0]
    assert job.status == "failed"
    assert job.error is not None and "fal exploded" in job.error


# ---------------------------------------------------------------------------
# Frame-edit-only jobs (frame-edit modal → LoRA Trainer queue)
# ---------------------------------------------------------------------------


def test_runner_completes_frame_edit_fal_job(test_state, fake_services, tmp_path):
    driver = _make_driver(tmp_path)
    test_state.lora_training.enqueue_derivation_job(_frame_edit_req(driver))
    test_state.lora_derivation_runner.reconcile_once()

    job = test_state.lora_training.get_derivation_jobs_state().jobs[0]
    assert job.status == "completed", job.error
    # The derived path is the edited still (a PNG), not a video clip.
    assert job.derived_path is not None and Path(job.derived_path).exists()
    assert job.edited_frame_path is not None and Path(job.edited_frame_path).exists()
    # The edit ran via the remote Nano Banana editor...
    assert len(fake_services.image_editor.calls) == 1
    # ...and no motion drive ran (frame_edit has no animate step).
    assert len(fake_services.video_restyler.motion_transfer_calls) == 0
    assert len(fake_services.video_restyler.kling_v2v_edit_calls) == 0


def test_runner_frame_edit_uses_klein_pipeline(test_state, fake_services, tmp_path, make_test_image):
    from runtime_config.model_download_specs import resolve_model_path

    klein_dir = resolve_model_path(test_state.config.default_models_dir, "flux-2-klein-9b")
    klein_dir.mkdir(parents=True, exist_ok=True)
    (klein_dir / "model.safetensors").write_bytes(b"\x00" * 1024)
    real_png = make_test_image().getvalue()
    fake_services.clip_processor.extract_frame = lambda *, video_path, time_seconds: real_png  # type: ignore[assignment]

    test_state.lora_training.enqueue_derivation_job(
        _frame_edit_req(_make_driver(tmp_path), editEngine="klein")
    )
    test_state.lora_derivation_runner.reconcile_once()

    job = test_state.lora_training.get_derivation_jobs_state().jobs[0]
    assert job.status == "completed", job.error
    assert job.derived_path is not None and Path(job.derived_path).exists()
    # Klein reference-edit ran; the remote editor and motion drive did not.
    assert len(fake_services.image_edit_pipeline.generate_with_references_calls) == 1
    assert fake_services.image_editor.calls == []
    assert len(fake_services.video_restyler.motion_transfer_calls) == 0


# ---------------------------------------------------------------------------
# Bounded concurrency + transient-failure retry
# ---------------------------------------------------------------------------


def test_is_transient_classifies_fal_errors():
    # Rate limit / server-side / network → retry.
    assert _is_transient(VideoRestylerError("Fal video job failed (429): slow down"))
    assert _is_transient(VideoRestylerError("Fal video job failed (503): unavailable"))
    assert _is_transient(VideoRestylerError("Fal video job failed (500): oops"))
    assert _is_transient(VideoRestylerError("request timed out"))
    # Permanent client errors → fail fast, no retry.
    assert not _is_transient(
        VideoRestylerError("Fal video job failed (422): video too large")
    )
    assert not _is_transient(VideoRestylerError("Fal video job failed (400): bad input"))
    assert not _is_transient(VideoRestylerError("fal exploded"))


def test_runner_retries_transient_fal_error_then_succeeds(
    test_state, fake_services, tmp_path
):
    runner = test_state.lora_derivation_runner
    # No real backoff in the test.
    runner._retry_base_seconds = 0.0
    runner._retry_cap_seconds = 0.0
    # First two Kling drive calls hit a 429; the third succeeds.
    fake_services.video_restyler.fail_times = 2
    test_state.lora_training.enqueue_derivation_job(_kling_req(_make_driver(tmp_path)))
    runner.reconcile_once()

    job = test_state.lora_training.get_derivation_jobs_state().jobs[0]
    assert job.status == "completed", job.error
    # Only the successful attempt records a call (failures raise before append).
    assert len(fake_services.video_restyler.motion_transfer_calls) == 1
    assert fake_services.video_restyler.fail_times == 0


def test_runner_does_not_retry_permanent_fal_error(test_state, fake_services, tmp_path):
    runner = test_state.lora_derivation_runner
    runner._retry_base_seconds = 0.0
    # A 422 ("video too large") is permanent — fail fast without retrying.
    fake_services.video_restyler.error = VideoRestylerError(
        "Fal video job failed (422): video too large"
    )
    test_state.lora_training.enqueue_derivation_job(_kling_req(_make_driver(tmp_path)))
    runner.reconcile_once()

    job = test_state.lora_training.get_derivation_jobs_state().jobs[0]
    assert job.status == "failed"
    assert job.error is not None and "422" in job.error


def test_derivation_concurrency_reads_and_clamps_setting(test_state):
    assert test_state.lora_training.derivation_concurrency() == 20
    test_state.state.app_settings.lora_fal_concurrency = 12
    assert test_state.lora_training.derivation_concurrency() == 12
    # Clamped to [1, 20] (validated on assignment + defended in the getter).
    test_state.state.app_settings.lora_fal_concurrency = 999
    assert test_state.lora_training.derivation_concurrency() == 20
    test_state.state.app_settings.lora_fal_concurrency = 0
    assert test_state.lora_training.derivation_concurrency() == 1


# ---------------------------------------------------------------------------
# Review gate (pause after edit, approve / regenerate before motion)
# ---------------------------------------------------------------------------


def test_runner_persists_source_frame_for_review(test_state, fake_services, tmp_path):
    test_state.lora_training.enqueue_derivation_job(_review_req(_make_driver(tmp_path)))
    test_state.lora_derivation_runner.reconcile_once()

    job = test_state.lora_training.get_derivation_jobs_state().jobs[0]
    assert job.status == "review"
    # The exact extracted source frame is persisted and distinct from the edit,
    # so the review UI can show a true before/after of the same frame.
    assert job.source_frame_path is not None and Path(job.source_frame_path).exists()
    assert job.source_frame_path != job.edited_frame_path
    # And it reaches the API the review modal reads.
    api = derivation_jobs_state_to_api(test_state.lora_training.get_derivation_jobs_state())
    assert api.jobs[0].sourceFramePath == job.source_frame_path


def test_runner_source_frame_is_the_still_when_no_edit(test_state, fake_services, tmp_path):
    still = tmp_path / "still.png"
    still.write_bytes(b"fake-still-png")
    test_state.lora_training.enqueue_derivation_job(
        _kling_req(_make_driver(tmp_path), editPrompt="", framePath=str(still))
    )
    test_state.lora_derivation_runner.reconcile_once()

    job = test_state.lora_training.get_derivation_jobs_state().jobs[0]
    assert job.status == "completed", job.error
    # No edit ran: the source frame is the still itself; no edited frame.
    assert job.source_frame_path == str(still)
    assert job.edited_frame_path is None


def test_runner_pauses_for_review(test_state, fake_services, tmp_path):
    test_state.lora_training.enqueue_derivation_job(_review_req(_make_driver(tmp_path)))
    test_state.lora_derivation_runner.reconcile_once()

    job = test_state.lora_training.get_derivation_jobs_state().jobs[0]
    # Edit ran and the job paused awaiting review — no motion drive yet.
    assert job.status == "review"
    assert job.edited_frame_path is not None and Path(job.edited_frame_path).exists()
    assert len(fake_services.image_editor.calls) == 1
    assert len(fake_services.video_restyler.motion_transfer_calls) == 0


def test_approve_then_drive_reuses_edited_frame(test_state, fake_services, tmp_path):
    job = test_state.lora_training.enqueue_derivation_job(_review_req(_make_driver(tmp_path)))
    test_state.lora_derivation_runner.reconcile_once()
    edited = test_state.lora_training.get_derivation_jobs_state().jobs[0].edited_frame_path

    approved = test_state.lora_training.approve_derivation_job(job.id)
    assert approved is not None and approved.status == "approved"

    test_state.lora_derivation_runner.reconcile_once()
    done = test_state.lora_training.get_derivation_jobs_state().jobs[0]
    assert done.status == "completed", done.error
    # The motion phase reused the approved still — no second edit call.
    assert len(fake_services.image_editor.calls) == 1
    assert done.edited_frame_path == edited
    assert len(fake_services.video_restyler.motion_transfer_calls) == 1


def test_retry_motion_failure_reuses_approved_frame(test_state, fake_services, tmp_path):
    job = test_state.lora_training.enqueue_derivation_job(
        _review_req(_make_driver(tmp_path))
    )
    test_state.lora_derivation_runner.reconcile_once()
    reviewed = test_state.lora_training.get_derivation_jobs_state().jobs[0]
    edited = reviewed.edited_frame_path
    assert edited is not None

    test_state.lora_training.approve_derivation_job(job.id)
    claimed = test_state.lora_training.claim_next_derivation_job()
    assert claimed is not None and claimed.status == "generating"
    test_state.lora_training.fail_derivation_job(job.id, "motion failed")

    retried = test_state.lora_training.retry_derivation_job(job.id)
    assert retried is not None
    assert retried.status == "approved"
    assert retried.edited_frame_path == edited

    test_state.lora_derivation_runner.reconcile_once()
    completed = test_state.lora_training.get_derivation_jobs_state().jobs[0]
    assert completed.status == "completed"
    assert completed.edited_frame_path == edited
    assert len(fake_services.image_editor.calls) == 1


def test_regenerate_edit_reruns_edit(test_state, fake_services, tmp_path):
    job = test_state.lora_training.enqueue_derivation_job(_review_req(_make_driver(tmp_path)))
    test_state.lora_derivation_runner.reconcile_once()

    regenerated = test_state.lora_training.regenerate_derivation_edit(
        job.id, edit_prompt="add a red hat"
    )
    assert regenerated is not None and regenerated.status == "pending"
    assert regenerated.edit_prompt == "add a red hat"
    assert regenerated.edited_frame_path is None

    # Re-runs the edit and pauses for review again (second edit call).
    test_state.lora_derivation_runner.reconcile_once()
    job2 = test_state.lora_training.get_derivation_jobs_state().jobs[0]
    assert job2.status == "review"
    assert len(fake_services.image_editor.calls) == 2


def test_approve_rejects_non_review_job(test_state, tmp_path):
    job = test_state.lora_training.enqueue_derivation_job(_review_req(_make_driver(tmp_path)))
    # Still pending — cannot approve.
    assert test_state.lora_training.approve_derivation_job(job.id) is None
    assert test_state.lora_training.regenerate_derivation_edit(job.id) is None


def test_recovery_resumes_approved_motion_phase(test_state, fake_services, tmp_path):
    job = test_state.lora_training.enqueue_derivation_job(_review_req(_make_driver(tmp_path)))
    test_state.lora_derivation_runner.reconcile_once()
    test_state.lora_training.approve_derivation_job(job.id)
    # Simulate a crash mid-drive: claim moves approved -> generating.
    claimed = test_state.lora_training.claim_next_derivation_job()
    assert claimed is not None and claimed.status == "generating"

    test_state.lora_training.load_state()
    recovered = test_state.lora_training.get_derivation_jobs_state().jobs[0]
    # Keeps the approved still and resumes the motion-only phase (no re-edit).
    assert recovered.status == "approved"
    assert recovered.edited_frame_path is not None


def test_recovery_reuses_non_review_anchor_after_crash(test_state, tmp_path):
    job = test_state.lora_training.enqueue_derivation_job(
        _kling_req(_make_driver(tmp_path))
    )
    test_state.lora_training.claim_next_derivation_job()
    edited = tmp_path / "approved-edit.png"
    edited.write_bytes(b"png")
    test_state.lora_training.mark_derivation_generating(
        job.id, edited_frame_path=str(edited)
    )

    test_state.lora_training.load_state()
    recovered = test_state.lora_training.get_derivation_jobs_state().jobs[0]
    assert recovered.status == "approved"
    assert recovered.edited_frame_path == str(edited)


def test_cancel_review_job_marks_cancelled(test_state, tmp_path):
    job = test_state.lora_training.enqueue_derivation_job(_review_req(_make_driver(tmp_path)))
    test_state.lora_derivation_runner.reconcile_once()
    assert test_state.lora_training.get_derivation_jobs_state().jobs[0].status == "review"
    test_state.lora_training.cancel_derivation_job(job.id)
    assert test_state.lora_training.get_derivation_jobs_state().jobs[0].status == "cancelled"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def test_create_and_list_derivation_routes(client, test_state, tmp_path):
    # Stop the background runner so cancel/dismiss assertions are not racing
    # the worker thread that the app lifespan started.
    test_state.lora_derivation_runner.stop()
    driver = _make_driver(tmp_path)
    res = client.post(
        "/api/lora/derivations",
        json={
            "driverPath": driver,
            "referencePath": driver,
            "engine": "kling",
            "editPrompt": "remove the sign",
            "label": "Target",
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["status"] == "pending"
    job_id = body["id"]

    listed = client.get("/api/lora/derivations")
    assert listed.status_code == 200
    assert any(j["id"] == job_id for j in listed.json()["jobs"])

    cancelled = client.post(f"/api/lora/derivations/{job_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"

    dismissed = client.post(f"/api/lora/derivations/{job_id}/dismiss")
    assert dismissed.status_code == 200
    assert all(j["id"] != job_id for j in dismissed.json()["jobs"])


def test_review_routes_reject_non_review_job(client, test_state, tmp_path):
    test_state.lora_derivation_runner.stop()
    driver = _make_driver(tmp_path)
    res = client.post(
        "/api/lora/derivations",
        json={"driverPath": driver, "engine": "kling", "editPrompt": "x", "requireReview": True},
    )
    assert res.status_code == 200, res.text
    job_id = res.json()["id"]

    # Job is still pending (runner stopped), so approve/regenerate are invalid.
    approve = client.post(f"/api/lora/derivations/{job_id}/approve")
    assert approve.status_code == 409
    regen = client.post(
        f"/api/lora/derivations/{job_id}/regenerate-edit", json={"editPrompt": None}
    )
    assert regen.status_code == 409

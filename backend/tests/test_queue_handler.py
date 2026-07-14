"""Unit tests for `handlers/queue_handler.py`.

Exercises the durable ledger mechanics: enqueue (with boundary
validation), batch, cancel, edit, remove, reorder, pause/resume, clear,
the runner-facing transitions (claim/complete/fail/cancel/requeue),
crash recovery, and atomic persistence. The state machine is driven
directly through a freshly constructed `QueueHandler` (no runner, no
FastAPI) so assertions are deterministic.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from handlers.queue_handler import (
    QueueHandler,
    QueueItemNotFoundError,
    QueueItemTransitionError,
    QueuePayloadValidationError,
)
from state.queue_state import (
    EnqueueRequest,
    ImageQueuePayload,
    QueueItem,
    VideoQueuePayload,
)
from api_types import GenerateImageRequest, GenerateVideoRequest


def _video_payload(prompt: str = "a cat") -> VideoQueuePayload:
    return VideoQueuePayload(kind="video", request=GenerateVideoRequest(prompt=prompt))


def _image_payload(prompt: str = "a dog") -> ImageQueuePayload:
    return ImageQueuePayload(kind="image", request=GenerateImageRequest(prompt=prompt))


def _video_draft(prompt: str = "a cat") -> EnqueueRequest:
    return EnqueueRequest(payload=_video_payload(prompt))


def _make_queue(test_state, validate=None) -> QueueHandler:
    # Fresh handler against the test config's app_data_dir with an
    # isolated lock. The conftest `test_state` fixture doesn't start the
    # runner, so there's no background claimer to race.
    return QueueHandler(
        state=test_state.state,
        lock=threading.RLock(),
        config=test_state.config,
        validate_payload=validate,
    )


# =====================================================================
# Enqueue / persistence
# =====================================================================


class TestEnqueue:
    def test_enqueue_creates_pending_item_and_persists(self, test_state):
        q = _make_queue(test_state)
        item = q.enqueue(_video_draft("a cat"))
        assert item.status == "pending"
        assert item.payload.kind == "video"
        assert item.payload.request.prompt == "a cat"
        assert item.source == "genspace"
        assert item.created_at
        # Persisted to disk.
        assert (test_state.config.app_data_dir / "queue.json").is_file()

    def test_enqueue_sets_wakeup(self, test_state):
        q = _make_queue(test_state)
        assert not q.wakeup_event.is_set()
        q.enqueue(_video_draft())
        assert q.wakeup_event.is_set()

    def test_enqueue_records_project_and_source(self, test_state):
        q = _make_queue(test_state)
        item = q.enqueue(
            EnqueueRequest(
                payload=_image_payload(),
                originating_project_id="proj-1",
                source="queue_manual",
            )
        )
        assert item.originating_project_id == "proj-1"
        assert item.source == "queue_manual"
        assert item.payload.kind == "image"

    def test_enqueue_batch_appends_all_and_single_write(self, test_state):
        q = _make_queue(test_state)
        items = q.enqueue_batch([_video_draft("a"), _video_draft("b"), _video_draft("c")])
        assert [i.payload.request.prompt for i in items] == ["a", "b", "c"]
        assert all(i.status == "pending" for i in items)
        assert len(q.get_state().items) == 3

    def test_enqueue_batch_empty_returns_empty(self, test_state):
        q = _make_queue(test_state)
        assert q.enqueue_batch([]) == []

    def test_enqueue_batch_validates_all_before_appending(self, test_state):
        def reject_third(payload):
            if payload.kind == "video" and payload.request.prompt == "c":
                raise ValueError("rejected c")

        q = _make_queue(test_state, validate=reject_third)
        with pytest.raises(QueuePayloadValidationError, match="rejected c"):
            q.enqueue_batch([_video_draft("a"), _video_draft("b"), _video_draft("c")])
        # All-or-nothing: nothing landed.
        assert q.get_state().items == []

    def test_enqueue_validation_rejects_bad_payload(self, test_state):
        def reject(payload):
            if payload.request.prompt == "BAD":
                raise ValueError("bad prompt")

        q = _make_queue(test_state, validate=reject)
        with pytest.raises(QueuePayloadValidationError, match="bad prompt"):
            q.enqueue(_video_draft("BAD"))
        assert q.get_state().items == []


# =====================================================================
# Read API
# =====================================================================


class TestReadApi:
    def test_get_state_returns_deep_copy(self, test_state):
        q = _make_queue(test_state)
        q.enqueue(_video_draft("a"))
        snap = q.get_state()
        snap.items[0].status = "completed"
        # Mutating the snapshot must not touch the in-memory ledger.
        assert q.get_state().items[0].status == "pending"

    def test_get_item_missing_returns_none(self, test_state):
        q = _make_queue(test_state)
        assert q.get_item("nope") is None

    def test_list_items_filtered_by_status(self, test_state):
        q = _make_queue(test_state)
        q.enqueue(_video_draft("a"))
        q.enqueue(_video_draft("b"))
        # Claim one so it's running.
        q.claim_next_pending()
        assert len(q.list_items()) == 2
        assert len(q.list_items(status="pending")) == 1
        assert len(q.list_items(status="running")) == 1
        assert q.list_items(status="completed") == []


# =====================================================================
# User-facing mutations
# =====================================================================


class TestMutations:
    def test_cancel_pending(self, test_state):
        q = _make_queue(test_state)
        item = q.enqueue(_video_draft())
        cancelled = q.cancel_item(item.id)
        assert cancelled.status == "cancelled"
        assert cancelled.completed_at

    def test_cancel_non_pending_rejected(self, test_state):
        q = _make_queue(test_state)
        item = q.enqueue(_video_draft())
        q.claim_next_pending()
        with pytest.raises(QueueItemTransitionError):
            q.cancel_item(item.id)

    def test_update_pending_payload(self, test_state):
        q = _make_queue(test_state)
        item = q.enqueue(_video_draft("a"))
        updated = q.update_pending_item(item.id, _video_payload("b"))
        assert updated.payload.request.prompt == "b"

    def test_update_pending_validates(self, test_state):
        def reject(payload):
            if payload.request.prompt == "BAD":
                raise ValueError("nope")

        q = _make_queue(test_state, validate=reject)
        item = q.enqueue(_video_draft("a"))
        with pytest.raises(QueuePayloadValidationError):
            q.update_pending_item(item.id, _video_payload("BAD"))

    def test_update_running_rejected(self, test_state):
        q = _make_queue(test_state)
        item = q.enqueue(_video_draft())
        q.claim_next_pending()
        with pytest.raises(QueueItemTransitionError):
            q.update_pending_item(item.id, _video_payload("b"))

    def test_remove_item(self, test_state):
        q = _make_queue(test_state)
        item = q.enqueue(_video_draft())
        q.remove_item(item.id)
        assert q.get_state().items == []

    def test_remove_running_rejected(self, test_state):
        q = _make_queue(test_state)
        item = q.enqueue(_video_draft())
        q.claim_next_pending()
        with pytest.raises(QueueItemTransitionError):
            q.remove_item(item.id)

    def test_remove_missing_raises(self, test_state):
        q = _make_queue(test_state)
        with pytest.raises(QueueItemNotFoundError):
            q.remove_item("nope")

    def test_reorder_pending_valid_permutation(self, test_state):
        q = _make_queue(test_state)
        a = q.enqueue(_video_draft("a"))
        b = q.enqueue(_video_draft("b"))
        c = q.enqueue(_video_draft("c"))
        # Reverse the pending order.
        q.reorder_pending([c.id, b.id, a.id])
        pending = [i for i in q.get_state().items if i.status == "pending"]
        assert [i.id for i in pending] == [c.id, b.id, a.id]

    def test_reorder_invalid_permutation_rejected(self, test_state):
        q = _make_queue(test_state)
        a = q.enqueue(_video_draft("a"))
        b = q.enqueue(_video_draft("b"))
        with pytest.raises(QueueItemTransitionError):
            q.reorder_pending([a.id])  # missing b
        with pytest.raises(QueueItemTransitionError):
            q.reorder_pending([a.id, b.id, a.id])  # duplicate

    def test_pause_then_resume_sets_wakeup(self, test_state):
        q = _make_queue(test_state)
        q.pause()
        assert q.get_state().paused is True
        q.wakeup_event.clear()
        q.resume()
        assert q.get_state().paused is False
        assert q.wakeup_event.is_set()

    def test_clear_completed_and_failed(self, test_state):
        q = _make_queue(test_state)
        a = q.enqueue(_video_draft("a"))
        b = q.enqueue(_video_draft("b"))
        # Drive a to completed, b to failed.
        q.claim_next_pending()
        q.complete_running(a.id, "/tmp/a.mp4")
        q.claim_next_pending()
        q.fail_running(b.id, "boom")  # first fail -> pending (retry)
        # Claim again and fail the retry.
        q.claim_next_pending()
        q.fail_running(b.id, "boom again")
        assert q.clear_completed() == 1
        assert q.clear_failed() == 1
        assert q.get_state().items == []


# =====================================================================
# Runner-facing transitions
# =====================================================================


class TestRunnerTransitions:
    def test_claim_pops_oldest_pending_and_marks_running(self, test_state):
        q = _make_queue(test_state)
        a = q.enqueue(_video_draft("a"))
        q.enqueue(_video_draft("b"))
        claimed = q.claim_next_pending()
        assert claimed is not None
        assert claimed.id == a.id
        assert claimed.status == "running"
        assert claimed.started_at

    def test_claim_paused_returns_none(self, test_state):
        q = _make_queue(test_state)
        q.enqueue(_video_draft())
        q.pause()
        assert q.claim_next_pending() is None

    def test_claim_empty_returns_none(self, test_state):
        q = _make_queue(test_state)
        assert q.claim_next_pending() is None

    def test_complete_running(self, test_state):
        q = _make_queue(test_state)
        item = q.enqueue(_video_draft())
        q.claim_next_pending()
        done = q.complete_running(item.id, "/tmp/out.mp4")
        assert done.status == "completed"
        assert done.output_path == "/tmp/out.mp4"
        assert done.error is None

    def test_complete_non_running_rejected(self, test_state):
        q = _make_queue(test_state)
        item = q.enqueue(_video_draft())
        with pytest.raises(QueueItemTransitionError):
            q.complete_running(item.id, "/tmp/out.mp4")

    def test_fail_first_attempt_reprepends_without_error(self, test_state):
        q = _make_queue(test_state)
        item = q.enqueue(_video_draft("a"))
        q.enqueue(_video_draft("b"))
        q.claim_next_pending()
        result = q.fail_running(item.id, "transient")
        assert result.status == "pending"
        assert result.retry_count == 1
        assert result.error is None  # first-attempt error discarded
        # Re-prepended at head: next claim is the retried item, not b.
        assert q.claim_next_pending().id == item.id

    def test_fail_second_attempt_marks_failed(self, test_state):
        q = _make_queue(test_state)
        item = q.enqueue(_video_draft())
        q.claim_next_pending()
        q.fail_running(item.id, "first")
        q.claim_next_pending()
        result = q.fail_running(item.id, "second")
        assert result.status == "failed"
        assert result.error == "second"
        assert result.retry_count == 1

    def test_cancel_running(self, test_state):
        q = _make_queue(test_state)
        item = q.enqueue(_video_draft())
        q.claim_next_pending()
        result = q.cancel_running(item.id)
        assert result.status == "cancelled"

    def test_cancel_running_item_cancels_the_running_one(self, test_state):
        q = _make_queue(test_state)
        item = q.enqueue(_video_draft("a"))
        q.enqueue(_video_draft("b"))  # pending, must be left alone
        q.claim_next_pending()
        result = q.cancel_running_item()
        assert result is not None
        assert result.id == item.id
        assert result.status == "cancelled"
        # The pending item is untouched.
        assert q.get_item(item.id).status == "cancelled"
        pending = [i for i in q.get_state().items if i.status == "pending"]
        assert len(pending) == 1 and pending[0].status == "pending"

    def test_cancel_running_item_noop_when_none_running(self, test_state):
        q = _make_queue(test_state)
        q.enqueue(_video_draft())  # pending only
        assert q.cancel_running_item() is None
        # Pending item left intact.
        assert q.get_state().items[0].status == "pending"

    def test_requeue_blocked_does_not_consume_retry(self, test_state):
        q = _make_queue(test_state)
        item = q.enqueue(_video_draft("a"))
        q.enqueue(_video_draft("b"))
        q.claim_next_pending()
        result = q.requeue_blocked(item.id)
        assert result.status == "pending"
        assert result.retry_count == 0  # untouched
        assert result.error is None
        # Item goes back to the head of pending.
        assert q.claim_next_pending().id == item.id

    def test_requeue_blocked_non_running_rejected(self, test_state):
        q = _make_queue(test_state)
        item = q.enqueue(_video_draft())
        with pytest.raises(QueueItemTransitionError):
            q.requeue_blocked(item.id)


# =====================================================================
# Crash recovery + persistence
# =====================================================================


class TestPersistenceAndRecovery:
    def test_running_item_remarked_pending_on_load(self, test_state, tmp_path):
        q = _make_queue(test_state)
        item = q.enqueue(_video_draft())
        q.claim_next_pending()  # -> running
        # A new handler instance simulates a restart, reading queue.json.
        q2 = QueueHandler(
            state=test_state.state,
            lock=threading.RLock(),
            config=test_state.config,
        )
        q2.load_queue()
        loaded = q2.get_item(item.id)
        assert loaded is not None
        assert loaded.status == "pending"
        assert loaded.started_at is None
        # A crash consumes the first retry (mirrors `fail_running`), so a
        # subsequent crash marks the item failed instead of looping forever.
        assert loaded.retry_count == 1

    def test_second_crash_marks_running_item_failed(self, test_state, tmp_path):
        q = _make_queue(test_state)
        item = q.enqueue(_video_draft())
        q.claim_next_pending()  # -> running, retry_count 0
        # First crash: re-queued as pending, retry_count 0 -> 1.
        q2 = QueueHandler(
            state=test_state.state,
            lock=threading.RLock(),
            config=test_state.config,
        )
        q2.load_queue()
        q2.claim_next_pending()  # -> running again, retry_count 1
        # Second crash: retry_count already 1 -> marked failed.
        q3 = QueueHandler(
            state=test_state.state,
            lock=threading.RLock(),
            config=test_state.config,
        )
        q3.load_queue()
        loaded = q3.get_item(item.id)
        assert loaded is not None
        assert loaded.status == "failed"
        assert loaded.retry_count == 1
        assert loaded.error is not None
        assert "crashed" in loaded.error.lower()

    def test_corrupt_file_backed_up_and_state_empties(self, test_state):
        q = _make_queue(test_state)
        # Write garbage to queue.json.
        qf = test_state.config.app_data_dir / "queue.json"
        qf.parent.mkdir(parents=True, exist_ok=True)
        qf.write_text("{not valid json", encoding="utf-8")
        q2 = QueueHandler(
            state=test_state.state,
            lock=threading.RLock(),
            config=test_state.config,
        )
        q2.load_queue()
        assert q2.get_state().items == []
        # A corrupt backup was preserved.
        backups = list(test_state.config.app_data_dir.glob("queue.corrupt-*.json"))
        assert len(backups) == 1

    def test_missing_file_is_empty_queue(self, test_state):
        q = _make_queue(test_state)
        # Never wrote anything; load is a no-op.
        q.load_queue()
        assert q.get_state().items == []

    def test_persist_round_trips_payload(self, test_state):
        q = _make_queue(test_state)
        q.enqueue(
            EnqueueRequest(
                payload=_image_payload("round trip"),
                originating_project_id="p",
            )
        )
        qf = test_state.config.app_data_dir / "queue.json"
        data = json.loads(qf.read_text(encoding="utf-8"))
        assert data["items"][0]["payload"]["kind"] == "image"
        assert data["items"][0]["payload"]["request"]["prompt"] == "round trip"
        assert data["items"][0]["originating_project_id"] == "p"

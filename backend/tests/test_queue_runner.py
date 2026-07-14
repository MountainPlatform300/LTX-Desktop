"""Unit tests for `handlers/queue_runner.py`.

The runner is tested with a stub `dispatch_fn` + `is_slot_free` against a
real `QueueHandler`, so the integration between the loop and the ledger
is exercised without spinning up the generation pipelines. Covers the
`QueueRunResult` -> transition mapping, the cooperative single-flight
back-off, retry-once, and lifecycle (start/stop/shutdown wake-up).
"""

from __future__ import annotations

import threading
import time

import pytest

from handlers.queue_handler import QueueHandler
from handlers.queue_runner import QueueRunner, QueueRunResult
from state.queue_state import EnqueueRequest, VideoQueuePayload
from api_types import GenerateVideoRequest


def _video_draft(prompt: str = "a cat") -> EnqueueRequest:
    return EnqueueRequest(
        payload=VideoQueuePayload(kind="video", request=GenerateVideoRequest(prompt=prompt))
    )


def _make_queue(test_state) -> QueueHandler:
    return QueueHandler(
        state=test_state.state,
        lock=threading.RLock(),
        config=test_state.config,
    )


def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


# =====================================================================
# _process_one: QueueRunResult -> transition mapping
# =====================================================================


class TestProcessOneMapping:
    def test_complete_marks_completed_with_output(self, test_state):
        q = _make_queue(test_state)
        item = q.enqueue(_video_draft())
        q.claim_next_pending()
        runner = QueueRunner(
            queue_handler=q,
            dispatch_fn=lambda payload: QueueRunResult(
                status="complete", output_path="/tmp/out.mp4"
            ),
            is_slot_free=lambda: True,
        )
        runner._process_one(item)
        assert q.get_item(item.id).status == "completed"
        assert q.get_item(item.id).output_path == "/tmp/out.mp4"

    def test_cancelled_marks_cancelled(self, test_state):
        q = _make_queue(test_state)
        item = q.enqueue(_video_draft())
        q.claim_next_pending()
        runner = QueueRunner(
            queue_handler=q,
            dispatch_fn=lambda payload: QueueRunResult(status="cancelled"),
            is_slot_free=lambda: True,
        )
        runner._process_one(item)
        assert q.get_item(item.id).status == "cancelled"

    def test_failed_first_attempt_retries(self, test_state):
        q = _make_queue(test_state)
        item = q.enqueue(_video_draft())
        q.claim_next_pending()
        runner = QueueRunner(
            queue_handler=q,
            dispatch_fn=lambda payload: QueueRunResult(
                status="failed", error="transient"
            ),
            is_slot_free=lambda: True,
        )
        runner._process_one(item)
        loaded = q.get_item(item.id)
        assert loaded.status == "pending"
        assert loaded.retry_count == 1

    def test_busy_requeues_without_retry(self, test_state):
        q = _make_queue(test_state)
        item = q.enqueue(_video_draft())
        q.claim_next_pending()
        runner = QueueRunner(
            queue_handler=q,
            dispatch_fn=lambda payload: QueueRunResult(status="busy"),
            is_slot_free=lambda: True,
        )
        runner._process_one(item)
        loaded = q.get_item(item.id)
        assert loaded.status == "pending"
        assert loaded.retry_count == 0  # busy never consumes a retry
        assert loaded.error is None

    def test_dispatch_raising_treated_as_failed(self, test_state):
        q = _make_queue(test_state)
        item = q.enqueue(_video_draft())
        q.claim_next_pending()

        def boom(_payload):
            raise RuntimeError("dispatch blew up")

        runner = QueueRunner(
            queue_handler=q, dispatch_fn=boom, is_slot_free=lambda: True
        )
        runner._process_one(item)
        loaded = q.get_item(item.id)
        assert loaded.status == "pending"  # first failure -> retry
        assert loaded.retry_count == 1


# =====================================================================
# Cooperative single-flight + loop
# =====================================================================


class TestLoopAndSingleFlight:
    def test_runner_processes_enqueued_item_end_to_end(self, test_state):
        q = _make_queue(test_state)
        calls: list[str] = []

        def dispatch(payload):
            calls.append(payload.request.prompt)
            return QueueRunResult(status="complete", output_path="/tmp/x.mp4")

        runner = QueueRunner(
            queue_handler=q,
            dispatch_fn=dispatch,
            is_slot_free=lambda: True,
            idle_poll_seconds=0.05,
            slot_busy_poll_seconds=0.05,
        )
        runner.start()
        try:
            item = q.enqueue(_video_draft("hello"))
            assert _wait_until(
                lambda: q.get_item(item.id).status == "completed"
            ), f"item never completed; state={q.get_state().items!r}"
            assert calls == ["hello"]
        finally:
            runner.stop()

    def test_busy_slot_prevents_claim(self, test_state):
        q = _make_queue(test_state)
        dispatch_calls: list[str] = []
        slot_free = {"value": False}

        def dispatch(payload):
            dispatch_calls.append(payload.request.prompt)
            return QueueRunResult(status="complete", output_path="/tmp/x.mp4")

        runner = QueueRunner(
            queue_handler=q,
            dispatch_fn=dispatch,
            is_slot_free=lambda: slot_free["value"],
            idle_poll_seconds=0.05,
            slot_busy_poll_seconds=0.05,
        )
        runner.start()
        try:
            q.enqueue(_video_draft("pending while busy"))
            time.sleep(0.2)
            # Slot busy: nothing claimed, nothing dispatched.
            assert dispatch_calls == []
            assert q.get_state().items[0].status == "pending"
            # Free the slot; the runner should pick it up.
            slot_free["value"] = True
            assert _wait_until(lambda: len(dispatch_calls) == 1)
        finally:
            runner.stop()

    def test_busy_outcome_requeues_and_backs_off(self, test_state):
        q = _make_queue(test_state)
        attempts: list[str] = []

        def dispatch(payload):
            attempts.append(payload.request.prompt)
            # First attempt busy, second complete.
            if len(attempts) == 1:
                return QueueRunResult(status="busy")
            return QueueRunResult(status="complete", output_path="/tmp/x.mp4")

        runner = QueueRunner(
            queue_handler=q,
            dispatch_fn=dispatch,
            is_slot_free=lambda: True,
            idle_poll_seconds=0.05,
            slot_busy_poll_seconds=0.02,
        )
        runner.start()
        try:
            item = q.enqueue(_video_draft("one"))
            assert _wait_until(
                lambda: q.get_item(item.id).status == "completed"
            ), f"state={q.get_state().items!r}"
            assert len(attempts) == 2  # busy then complete
            assert q.get_item(item.id).retry_count == 0
        finally:
            runner.stop()

    def test_paused_runner_does_not_claim(self, test_state):
        q = _make_queue(test_state)
        q.pause()
        dispatch_calls: list[str] = []

        runner = QueueRunner(
            queue_handler=q,
            dispatch_fn=lambda payload: (
                dispatch_calls.append(payload.request.prompt),
                QueueRunResult(status="complete", output_path="/tmp/x.mp4"),
            )[1],
            is_slot_free=lambda: True,
            idle_poll_seconds=0.05,
            slot_busy_poll_seconds=0.05,
        )
        runner.start()
        try:
            q.enqueue(_video_draft("paused work"))
            time.sleep(0.15)
            assert dispatch_calls == []
            assert q.get_state().items[0].status == "pending"
            # Resume -> processed.
            q.resume()
            assert _wait_until(lambda: len(dispatch_calls) == 1)
        finally:
            runner.stop()


# =====================================================================
# Lifecycle
# =====================================================================


class TestLifecycle:
    def test_start_is_idempotent(self, test_state):
        q = _make_queue(test_state)
        runner = QueueRunner(
            queue_handler=q,
            dispatch_fn=lambda payload: QueueRunResult(status="complete"),
            is_slot_free=lambda: True,
        )
        runner.start()
        first = runner.is_running
        runner.start()  # no-op
        assert runner.is_running is first
        runner.stop()
        assert not runner.is_running

    def test_stop_wakes_blocked_runner(self, test_state):
        q = _make_queue(test_state)
        runner = QueueRunner(
            queue_handler=q,
            dispatch_fn=lambda payload: QueueRunResult(status="complete"),
            is_slot_free=lambda: True,
            idle_poll_seconds=30.0,  # would block a long time if not woken
        )
        runner.start()
        time.sleep(0.05)  # let it block on the wakeup event
        # stop() should return well under the idle timeout because it
        # sets the wakeup event.
        t0 = time.monotonic()
        runner.stop()
        assert time.monotonic() - t0 < 5.0
        assert not runner.is_running

    def test_stop_when_never_started_is_noop(self, test_state):
        q = _make_queue(test_state)
        runner = QueueRunner(
            queue_handler=q,
            dispatch_fn=lambda payload: QueueRunResult(status="complete"),
            is_slot_free=lambda: True,
        )
        runner.stop()  # must not raise
        assert not runner.is_running

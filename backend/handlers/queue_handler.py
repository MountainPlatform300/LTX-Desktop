"""Durable batch generation queue handler.

Owns `APP_DATA_DIR/queue.json`, the on-disk ledger of pending / running /
completed / failed / cancelled queue items. Persistence is the whole
point: an overnight queue must survive an OS sleep, a backend crash, or
an Electron restart and pick up exactly where it left off.

Two consumer surfaces:

1. **User-facing CRUD** — enqueue, batch enqueue, list, cancel pending,
   remove, edit pending payload, reorder, pause, resume, clear-completed,
   clear-failed. These are wrapped 1:1 by the HTTP routes in
   `_routes/queue.py`.
2. **Runner-facing transitions** — `claim_next_pending`,
   `complete_running`, `fail_running`, `cancel_running`,
   `requeue_blocked`. The `QueueRunner` background thread is the only
   caller; the user-facing API never moves items into or out of
   `running` directly.

Enqueue-time validation:
- A `validate_payload` callable (wired by `AppHandler` to the video
  spec validator; a no-op for image, whose fields are already
  pydantic-constrained) is invoked for every enqueue and pending-edit.
  Rejecting a malformed request at the boundary — before it occupies a
  queue slot — is both a UX win (the user gets an immediate 422 instead
  of a silent failure two attempts later) and a security/robustness win
  (the runner never feeds a bad snapshot to the GPU pipeline). The
  callable raises `ValueError` on invalid; the route layer maps that to
  a 422.

Crash recovery:
- Any item left in `running` at boot is a leftover from a process death
  (the runner is the only thing that can write `running`, and only while
  it's actively generating). On `load_queue()` we flip it back to
  `pending` so the runner re-attempts on restart, preserving
  `retry_count` so a hard-stuck item still hits the retry cap.
- Corrupt or unparseable `queue.json` is moved aside to a `.corrupt-*`
  backup file (so we don't lose the user's queue if they ask) and the
  in-memory state defaults to empty. Pre-existing valid items on disk
  are never destroyed silently.

Persistence is atomic via temp-file + `os.replace`. The whole
`QueueState` is rewritten on every mutation — the file is small
(< 1 MB even for hundreds of items because payloads are mostly short
strings + ints) so partial writes aren't worth the complexity.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from handlers.base import StateHandlerBase, with_state_lock
from state.app_state_types import AppState
from state.queue_state import (
    EnqueueRequest,
    QueueItem,
    QueueItemStatus,
    QueuePayload,
    QueueState,
)

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)

# Mutable status set used for cleanup / removal validation. Kept as a
# module-level constant so the typing-Literal-vs-runtime gap doesn't
# leak into every method that needs to decide "is this a terminal
# state?".
_TERMINAL_STATUSES: frozenset[QueueItemStatus] = frozenset(
    ("completed", "failed", "cancelled")
)

# A payload validator returns None on success or raises `ValueError`
# with a user-facing message. `None` means "no validation configured"
# (e.g., in tests that exercise the ledger mechanics directly); in that
# case the handler accepts any pydantic-valid payload.
ValidatePayload = Callable[[QueuePayload], None]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class QueueItemNotFoundError(LookupError):
    """Raised when a queue item id doesn't exist."""


class QueueItemTransitionError(ValueError):
    """Raised when a state transition is rejected (e.g. cancelling a
    completed item, removing a running item)."""


class QueuePayloadValidationError(ValueError):
    """Raised when an enqueue/edit payload fails boundary validation.
    Distinct from `QueueItemTransitionError` so the route layer can map
    it to 422 rather than 409."""


class QueueHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        config: "RuntimeConfig",
        validate_payload: ValidatePayload | None = None,
    ) -> None:
        super().__init__(state, lock, config)
        self._queue_file: Path = config.app_data_dir / "queue.json"
        self._queue: QueueState = QueueState()
        self._validate_payload: ValidatePayload | None = validate_payload
        # Set whenever new pending work appears (enqueue / enqueue_batch
        # / resume). The QueueRunner waits on this so the runner thread
        # blocks while idle instead of polling. See `wakeup_event`.
        self._wakeup_event: threading.Event = threading.Event()

    @property
    def wakeup_event(self) -> threading.Event:
        """Event the runner blocks on while the queue is idle.

        Set by mutations that may produce claimable work (`enqueue`,
        `enqueue_batch`, `resume`). The runner clears the event after
        waking — we don't clear here so a "sleeping wakeup" race can't
        drop a notification when the runner hasn't blocked yet.
        Callers other than the runner should treat this as opaque.
        """
        return self._wakeup_event

    # ------------------------------------------------------------------
    # Persistence + crash recovery
    # ------------------------------------------------------------------

    @with_state_lock
    def load_queue(self) -> None:
        """Read `queue.json` if present and apply crash recovery.

        Called once at AppHandler boot from `load_persistent_state`. A
        missing file is normal (first run) and produces an empty queue.
        A malformed file is preserved as a `.corrupt-<ts>` backup and
        the in-memory state defaults to empty so the user doesn't get
        wedged at boot.
        """
        if not self._queue_file.exists():
            return

        try:
            with open(self._queue_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            loaded = QueueState.model_validate(payload)
        except Exception as exc:
            backup = self._queue_file.with_name(
                f"{self._queue_file.stem}.corrupt-{int(time.time())}.json"
            )
            try:
                shutil.copy2(self._queue_file, backup)
                logger.warning(
                    "queue.json could not be parsed; backed up to %s and "
                    "starting with empty queue: %s",
                    backup,
                    exc,
                )
            except Exception:
                logger.warning(
                    "queue.json could not be parsed and backup also failed; "
                    "starting with empty queue: %s",
                    exc,
                )
            return

        requeued = 0
        marked_failed = 0
        for item in loaded.items:
            if item.status != "running":
                continue
            # The runner is the only writer of "running"; if we see one
            # here, the process died mid-generation. Treat the crash as a
            # failed attempt and consume a retry, mirroring `fail_running`:
            # the first crash re-queues the item, a second crash (or a
            # crash after a real failure already consumed the retry) marks
            # it `failed`. This is what stops a hard-stuck generation —
            # one whose inference call hangs so `fail_running` is never
            # reached — from looping forever across restarts. (The previous
            # behavior left `retry_count` untouched on recovery, so the cap
            # never triggered and the stuck item re-ran on every boot.)
            if item.retry_count >= 1:
                item.status = "failed"
                item.started_at = None
                item.completed_at = _now_iso()
                item.error = (
                    "Generation crashed mid-run and failed to recover "
                    "after retry (likely hung or ran out of GPU memory)."
                )
                marked_failed += 1
            else:
                item.retry_count = 1
                item.status = "pending"
                item.started_at = None
                requeued += 1

        self._queue = loaded
        if requeued or marked_failed:
            logger.info(
                "Queue recovered %d running item(s) from prior crash; "
                "%d re-queued as pending, %d marked failed",
                requeued + marked_failed,
                requeued,
                marked_failed,
            )
            self._persist_unlocked()

    def _persist_unlocked(self) -> None:
        """Atomic rewrite of `queue.json`. Caller must hold `self.lock`.

        Writes to a sibling `.tmp` then `os.replace` — atomic on POSIX
        and same-volume Windows. fsync the temp file so the rename
        operates on durable bytes (a power loss between rename and flush
        would otherwise leave us with a renamed-but-empty file).
        """
        self._queue_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._queue_file.with_name(self._queue_file.name + ".tmp")
        payload = self._queue.model_dump(mode="json")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._queue_file)

    def _validate(self, payload: QueuePayload) -> None:
        if self._validate_payload is not None:
            try:
                self._validate_payload(payload)
            except ValueError as exc:
                raise QueuePayloadValidationError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    @with_state_lock
    def get_state(self) -> QueueState:
        """Deep-copied snapshot of the queue state.

        Deep copy is important: callers (route handlers, tests) must not
        be able to mutate the in-memory ledger directly.
        """
        return self._queue.model_copy(deep=True)

    @with_state_lock
    def get_item(self, item_id: str) -> QueueItem | None:
        item = self._find_unlocked(item_id)
        return None if item is None else item.model_copy(deep=True)

    @with_state_lock
    def list_items(
        self,
        *,
        status: QueueItemStatus | None = None,
    ) -> list[QueueItem]:
        if status is None:
            return [item.model_copy(deep=True) for item in self._queue.items]
        return [
            item.model_copy(deep=True)
            for item in self._queue.items
            if item.status == status
        ]

    # ------------------------------------------------------------------
    # User-facing mutations
    # ------------------------------------------------------------------

    @with_state_lock
    def enqueue(self, draft: EnqueueRequest) -> QueueItem:
        self._validate(draft.payload)
        item = self._build_item(draft)
        self._queue.items.append(item)
        self._persist_unlocked()
        self._wakeup_event.set()
        return item.model_copy(deep=True)

    @with_state_lock
    def enqueue_batch(self, drafts: list[EnqueueRequest]) -> list[QueueItem]:
        if not drafts:
            return []
        # Validate the whole batch up front so a single bad payload
        # rejects the entire batch atomically — the caller (manual
        # multi-prompt / brainstorm enqueue) doesn't end up with a
        # partially-landed batch and a confusing partial UI.
        for draft in drafts:
            self._validate(draft.payload)
        appended: list[QueueItem] = []
        for draft in drafts:
            item = self._build_item(draft)
            self._queue.items.append(item)
            appended.append(item)
        self._persist_unlocked()
        self._wakeup_event.set()
        return [item.model_copy(deep=True) for item in appended]

    @with_state_lock
    def cancel_item(self, item_id: str) -> QueueItem:
        """Cancel a *pending* item. Running items must go through the
        existing `/api/generate/cancel` flow (observed by the runner,
        which calls `cancel_running`)."""
        item = self._require_item(item_id)
        if item.status != "pending":
            raise QueueItemTransitionError(
                f"Cannot cancel item {item_id} in status {item.status!r}; "
                "only pending items can be cancelled via this API"
            )
        item.status = "cancelled"
        item.completed_at = _now_iso()
        self._persist_unlocked()
        return item.model_copy(deep=True)

    @with_state_lock
    def update_pending_item(
        self,
        item_id: str,
        new_payload: QueuePayload,
    ) -> QueueItem:
        """Replace the payload of a *pending* item.

        Editing a running item would be racy (the runner has already
        snapshotted the payload and started rendering against it), so we
        restrict edits to pending only. Terminal items (completed /
        failed / cancelled) likewise can't be edited — re-enqueueing is
        the right path for those.
        """
        self._validate(new_payload)
        item = self._require_item(item_id)
        if item.status != "pending":
            raise QueueItemTransitionError(
                f"Cannot edit item {item_id} in status {item.status!r}; "
                "only pending items are editable"
            )
        item.payload = new_payload
        self._persist_unlocked()
        return item.model_copy(deep=True)

    @with_state_lock
    def remove_item(self, item_id: str) -> None:
        item = self._require_item(item_id)
        if item.status == "running":
            raise QueueItemTransitionError(
                f"Cannot remove running item {item_id}; cancel it first"
            )
        self._queue.items.remove(item)
        self._persist_unlocked()

    @with_state_lock
    def reorder_pending(self, item_ids: list[str]) -> list[QueueItem]:
        """Apply a new ordering to *all* pending items at once.

        `item_ids` must be a permutation of the currently-pending item
        ids. Reordering doesn't move the running, completed, failed, or
        cancelled items (they stay where they are in the underlying
        list); only the relative order of pending items changes.

        We accept the full ordering rather than swap-based or
        index-based moves because dragging in the UI naturally produces
        "here's the new full ordering" — and validating one permutation
        is cleaner than reasoning about partial moves.
        """
        pending = [item for item in self._queue.items if item.status == "pending"]
        existing_ids = {item.id for item in pending}
        requested_ids = set(item_ids)
        if existing_ids != requested_ids or len(item_ids) != len(pending):
            raise QueueItemTransitionError(
                "reorder_pending requires a permutation of the current "
                f"pending ids; got {sorted(requested_ids)}, "
                f"have {sorted(existing_ids)}"
            )

        by_id = {item.id: item for item in pending}
        reordered = [by_id[item_id] for item_id in item_ids]

        # Walk the underlying list, replacing pending entries in
        # encountered order with the new permutation. Non-pending
        # entries are left untouched and keep their relative position
        # (so completed history stays in chronological order).
        new_items: list[QueueItem] = []
        cursor = iter(reordered)
        for item in self._queue.items:
            if item.status == "pending":
                new_items.append(next(cursor))
            else:
                new_items.append(item)
        self._queue.items = new_items
        self._persist_unlocked()
        return [item.model_copy(deep=True) for item in reordered]

    @with_state_lock
    def pause(self) -> QueueState:
        if not self._queue.paused:
            self._queue.paused = True
            self._persist_unlocked()
        return self._queue.model_copy(deep=True)

    @with_state_lock
    def resume(self) -> QueueState:
        if self._queue.paused:
            self._queue.paused = False
            self._persist_unlocked()
            # Existing pending items become claimable again — wake the
            # runner so we don't wait for the next enqueue or for the
            # idle-poll timeout to fire.
            self._wakeup_event.set()
        return self._queue.model_copy(deep=True)

    @with_state_lock
    def clear_completed(self) -> int:
        return self._clear_status("completed")

    @with_state_lock
    def clear_failed(self) -> int:
        return self._clear_status("failed")

    # ------------------------------------------------------------------
    # Runner-facing transitions (called by QueueRunner)
    # ------------------------------------------------------------------

    @with_state_lock
    def claim_next_pending(self) -> QueueItem | None:
        """Pop the oldest pending item, mark it running, return a copy.

        Returns None if the queue is paused or empty. Caller (the
        runner) is responsible for calling `complete_running` /
        `fail_running` / `cancel_running` / `requeue_blocked` once the
        work finishes.

        Returning a deep copy intentionally — the runner shouldn't be
        mutating the ledger directly. All updates flow through the
        explicit transition methods so persistence stays atomic.
        """
        if self._queue.paused:
            return None
        for item in self._queue.items:
            if item.status == "pending":
                item.status = "running"
                item.started_at = _now_iso()
                self._persist_unlocked()
                return item.model_copy(deep=True)
        return None

    @with_state_lock
    def complete_running(self, item_id: str, output_path: str) -> QueueItem:
        item = self._require_item(item_id)
        if item.status != "running":
            raise QueueItemTransitionError(
                f"Cannot complete item {item_id} in status {item.status!r}; "
                "expected 'running'"
            )
        item.status = "completed"
        item.completed_at = _now_iso()
        item.output_path = output_path
        item.error = None
        self._persist_unlocked()
        return item.model_copy(deep=True)

    @with_state_lock
    def fail_running(self, item_id: str, error: str) -> QueueItem:
        """Apply the retry-once-immediately policy.

        - First failure (`retry_count == 0`): re-prepend as pending so
          the runner picks this exact item back up next, before any
          other pending work. The first-attempt error is intentionally
          discarded — if the retry succeeds, the user shouldn't see a
          stale red flag.
        - Second failure (`retry_count == 1`): mark `failed` with the
          second-attempt error message and move on.
        """
        item = self._require_item(item_id)
        if item.status != "running":
            raise QueueItemTransitionError(
                f"Cannot fail item {item_id} in status {item.status!r}; "
                "expected 'running'"
            )

        if item.retry_count == 0:
            item.retry_count = 1
            item.status = "pending"
            item.started_at = None
            # Move the item to the head of the list so it's the next
            # `claim_next_pending` result. The list mixes statuses, so
            # "head of list" approximates "head of pending queue"
            # accurately whenever no other pending item precedes it
            # already; in practice the failed item *was* running, so
            # by definition every pending item came after it in
            # creation order — we just put it back ahead of them.
            self._queue.items.remove(item)
            self._queue.items.insert(0, item)
            logger.info(
                "Queue: item %s failed first attempt; retrying immediately "
                "(error: %s)",
                item_id,
                error,
            )
        else:
            item.status = "failed"
            item.completed_at = _now_iso()
            item.error = error
            logger.info(
                "Queue: item %s failed twice; marking failed (error: %s)",
                item_id,
                error,
            )

        self._persist_unlocked()
        return item.model_copy(deep=True)

    @with_state_lock
    def cancel_running(self, item_id: str) -> QueueItem:
        """Mark a running item as cancelled (called by the runner when
        the user hits cancel during generation)."""
        item = self._require_item(item_id)
        if item.status != "running":
            raise QueueItemTransitionError(
                f"Cannot cancel-running item {item_id} in status {item.status!r}; "
                "expected 'running'"
            )
        item.status = "cancelled"
        item.completed_at = _now_iso()
        self._persist_unlocked()
        return item.model_copy(deep=True)

    @with_state_lock
    def cancel_running_item(self) -> QueueItem | None:
        """Mark the currently-running queue item as cancelled.

        Called from the generation cancel flow so a user-initiated Stop on
        a queued generation also cancels the ledger entry directly — even
        when the underlying inference call is stuck (e.g. an OOM'd CUDA
        kernel that never returns) and the runner can't observe the cancel
        to call `cancel_running` itself. Without this, force-closing the
        app would leave the item `running` on disk and crash recovery would
        re-queue it, re-running the stuck generation on every restart.

        Safe to call when no queue item is running (the active generation
        is a non-queue retake / editor regen, or nothing is running at
        all): returns None. There is at most one running item because the
        generation slot is single-flight.
        """
        for item in self._queue.items:
            if item.status == "running":
                item.status = "cancelled"
                item.completed_at = _now_iso()
                self._persist_unlocked()
                return item.model_copy(deep=True)
        return None

    @with_state_lock
    def requeue_blocked(self, item_id: str) -> QueueItem:
        """Return a running item to pending without consuming a retry.

        Called by the runner when the single-flight generation slot was
        already occupied by a non-queue surface (retake / IC-LoRA /
        editor regen) at the moment the runner tried to start. This is
        not a failure of the item, so `retry_count` is untouched and no
        error is recorded — the item just waits its turn at the head of
        pending. The runner follows up with a short back-off sleep so it
        doesn't spin against an occupied slot.
        """
        item = self._require_item(item_id)
        if item.status != "running":
            raise QueueItemTransitionError(
                f"Cannot requeue-blocked item {item_id} in status "
                f"{item.status!r}; expected 'running'"
            )
        item.status = "pending"
        item.started_at = None
        self._queue.items.remove(item)
        self._queue.items.insert(0, item)
        self._persist_unlocked()
        return item.model_copy(deep=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_item(self, draft: EnqueueRequest) -> QueueItem:
        return QueueItem(
            id=uuid.uuid4().hex,
            status="pending",
            created_at=_now_iso(),
            originating_project_id=draft.originating_project_id,
            payload=draft.payload,
            source=draft.source,
        )

    def _find_unlocked(self, item_id: str) -> QueueItem | None:
        for item in self._queue.items:
            if item.id == item_id:
                return item
        return None

    def _require_item(self, item_id: str) -> QueueItem:
        item = self._find_unlocked(item_id)
        if item is None:
            raise QueueItemNotFoundError(f"Queue item not found: {item_id}")
        return item

    def _clear_status(self, status: QueueItemStatus) -> int:
        before = len(self._queue.items)
        self._queue.items = [
            item for item in self._queue.items if item.status != status
        ]
        removed = before - len(self._queue.items)
        if removed:
            self._persist_unlocked()
        return removed


__all__ = [
    "QueueHandler",
    "QueueItemNotFoundError",
    "QueueItemTransitionError",
    "QueuePayloadValidationError",
    "ValidatePayload",
]

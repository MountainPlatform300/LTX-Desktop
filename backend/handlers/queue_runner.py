"""Background runner thread that drives the durable queue ledger.

The runner is the bridge between the persistent `QueueHandler` and the
existing single-flight generation handlers:

  ┌──────────────┐ enqueue / resume   ┌──────────────┐
  │  HTTP route  │ ─────────────────▶ │ QueueHandler │ ──▶ queue.json
  └──────────────┘                    └──────┬───────┘
                                             │ wakeup_event
                                             ▼
                                       ┌─────────────┐
                                       │ QueueRunner │ ──┐
                                       └─────────────┘   │ dispatch(payload)
                                                         ▼
                                          ┌──────────────────────────────┐
                                          │ video / image generation     │
                                          │ (single-flight GPU/API slot) │
                                          └──────────────────────────────┘

Loop shape:

  1. If the queue is paused or empty, block on `wakeup_event` until a
     producer signals (or an idle-poll sanity timeout fires).
  2. If the single-flight generation slot is currently occupied by a
     non-queue surface (retake / IC-LoRA / editor regen), back off
     briefly and re-check — see "Cooperative single-flight" below.
  3. `claim_next_pending` — handler atomically pops the oldest pending
     item, marks it `running`, persists, returns a snapshot.
  4. Invoke `dispatch_fn(item.payload)`, which routes to the right
     generation handler by `payload.kind` and normalizes the response
     into a `QueueRunResult` (`complete` / `cancelled` / `failed` /
     `busy`).
  5. Map the result:
       - `complete`   → `complete_running(item.id, output_path)`
       - `cancelled`  → `cancel_running(item.id)`
       - `failed`     → `fail_running(item.id, error)` (applies the
         retry-once-immediately policy itself)
       - `busy`       → `requeue_blocked(item.id)` + short back-off
  6. Loop. After a successful complete the next iteration immediately
     re-claims; after a fail-and-retry the same item gets re-prepended
     by `fail_running` and the next claim picks it up again.

Cooperative single-flight:
- The generation handlers (`VideoGenerationHandler`,
  `ImageGenerationHandler`) share a single in-flight slot with other
  surfaces (retake, IC-LoRA, editor regen, gap generation). The queue
  does NOT take ownership of that slot — it cooperates. Before claiming
  it calls `is_slot_free()` to avoid churning `queue.json` while a
  non-queue generation is running; the authoritative gate is the
  generation handler's own `start_generation` guard, so a race between
  the check and the claim is recovered by `dispatch_fn` returning
  `busy`, which re-queues the item (no retry consumed) and backs off.

Threading rules:
- One runner thread per `QueueRunner`. `start()` is idempotent. `stop()`
  sets the shutdown event, signals the wakeup event so a blocked runner
  wakes, and joins.
- The runner never holds the queue handler's lock across the call to
  `dispatch_fn` — `claim_next_pending` and the transition methods
  acquire+release the lock per-call.
- The handler-side mutations are robust against missing IDs (raise
  cleanly), so a well-behaved runner can't corrupt state by getting out
  of sync with the on-disk ledger.

Cancellation:
- The user-facing `/api/generate/cancel` route flips the existing
  `GenerationHandler` state to `Cancelled`. The local generation loop
  polls that flag and raises; the generation handler returns a
  `*CancelledResponse`, which `dispatch_fn` normalizes to
  `QueueRunResult.cancelled`. The runner observes that and calls
  `cancel_running(item.id)`. No new cancel plumbing is needed.

Shutdown:
- `stop()` is called from the FastAPI lifespan finalizer. It flips
  `_shutdown`, sets `_wakeup_event`, and joins with a generous timeout.
  If a long-running generation is in-flight at shutdown, we wait for it
  to complete naturally — we don't kill it mid-step because that would
  leave temp files, half-written outputs, and dirty ML state. The next
  process boot picks up a `running` item via crash recovery anyway, so
  even a hard kill self-heals.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Literal, Protocol

from logging_policy import log_background_exception
from state.queue_state import QueueItem, QueuePayload

logger = logging.getLogger(__name__)

_BACKGROUND_TASK_NAME = "queue-runner"

# Default sanity-timeout for the wait. We rely on the wakeup event for
# normal liveness, but a periodic re-check guards against a missed-notify
# bug that would otherwise wedge the queue. 10s is imperceptible for an
# overnight queue.
DEFAULT_IDLE_POLL_SECONDS: float = 10.0
# When the single-flight slot is occupied by a non-queue surface, we
# back off briefly instead of spinning. Short enough that the queue
# picks up work promptly once the slot frees; long enough that we don't
# burn CPU pegging `is_slot_free` against a multi-minute retake.
DEFAULT_SLOT_BUSY_POLL_SECONDS: float = 2.0
# How long `stop()` waits for the runner thread to exit. The runner
# blocks at most for the in-flight generation, which on a 1080p render
# can be a couple of minutes on Apple Silicon. We err generous rather
# than truncate; the alternative is killing a half-written output.
DEFAULT_STOP_JOIN_TIMEOUT_SECONDS: float = 600.0


QueueRunStatus = Literal["complete", "cancelled", "failed", "busy"]


@dataclass(frozen=True)
class QueueRunResult:
    """Normalized outcome of dispatching one queue item.

    The runner depends only on this shape, not on the video/image
    response unions, so the AppHandler-provided `dispatch_fn` is the
    single place that knows how to translate generation-handler
    responses (and the "slot already in progress" race) into a queue
    transition.
    """

    status: QueueRunStatus
    # Populated for `complete` — the primary output path (video path or
    # first image path). The asset router resolves project routing from
    # the originating project id; the queue only records a breadcrumb.
    output_path: str | None = None
    # Populated for `failed` — the user-facing error message.
    error: str | None = None


class QueueDispatchCallable(Protocol):
    """Narrow callable contract: run one queued payload.

    In production this is wired by `AppHandler` to call
    `VideoGenerationHandler.generate` / `ImageGenerationHandler.generate`
    by `payload.kind` and normalize the response. The Protocol exists so
    tests can plug in a stub without spinning up pipelines.
    """

    def __call__(self, payload: QueuePayload) -> QueueRunResult: ...


class QueueSlotFreeCallable(Protocol):
    """Returns True when the single-flight generation slot is idle.

    Wired to `GenerationHandler.is_generation_running` inverted. Used as
    a cheap pre-check so the runner doesn't claim (and persist) an item
    it can't yet start. Authoritative gating still happens inside the
    generation handler's `start_generation`.
    """

    def __call__(self) -> bool: ...


class QueueRunner:
    def __init__(
        self,
        *,
        queue_handler: "QueueHandlerLike",
        dispatch_fn: QueueDispatchCallable,
        is_slot_free: QueueSlotFreeCallable,
        idle_poll_seconds: float = DEFAULT_IDLE_POLL_SECONDS,
        slot_busy_poll_seconds: float = DEFAULT_SLOT_BUSY_POLL_SECONDS,
        stop_join_timeout_seconds: float = DEFAULT_STOP_JOIN_TIMEOUT_SECONDS,
    ) -> None:
        self._queue = queue_handler
        self._dispatch_fn = dispatch_fn
        self._is_slot_free = is_slot_free
        self._idle_poll_seconds = idle_poll_seconds
        self._slot_busy_poll_seconds = slot_busy_poll_seconds
        self._stop_join_timeout = stop_join_timeout_seconds
        self._wakeup = queue_handler.wakeup_event
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        # Guards `start` / `stop` against concurrent invocation. Distinct
        # from the queue handler's lock so we never inherit lock-order
        # constraints from there.
        self._lifecycle_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the runner thread. Idempotent."""
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._shutdown.clear()
            # Daemon=True so a forced process exit doesn't hang on the
            # thread (graceful shutdown still goes through `stop()`,
            # which is what we expect from the FastAPI lifespan).
            self._thread = threading.Thread(
                target=self._run_loop,
                name="queue-runner",
                daemon=True,
            )
            self._thread.start()
            logger.info("Queue runner started")

    def stop(self) -> None:
        """Signal shutdown and join. Safe to call from any thread.

        Waits up to `stop_join_timeout_seconds` for an in-flight
        generation to complete. If the join times out, we log and
        return — a daemon thread doesn't block process exit, and the
        next boot's crash recovery flips any leftover running item back
        to pending.
        """
        with self._lifecycle_lock:
            thread = self._thread
            if thread is None:
                return
            self._shutdown.set()
            self._wakeup.set()  # wake a blocked runner so it sees shutdown
            self._thread = None
        thread.join(timeout=self._stop_join_timeout)
        if thread.is_alive():
            logger.warning(
                "Queue runner did not stop within %.0fs; leaving daemon "
                "thread to be reaped at process exit (next boot will "
                "recover any 'running' item via crash recovery)",
                self._stop_join_timeout,
            )
        else:
            logger.info("Queue runner stopped")

    @property
    def is_running(self) -> bool:
        with self._lifecycle_lock:
            return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Loop body
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        try:
            while not self._shutdown.is_set():
                # Cheap pre-check: don't churn the ledger while a
                # non-queue surface is using the single-flight slot.
                if not self._is_slot_free():
                    self._sleep_shutdown_aware(self._slot_busy_poll_seconds)
                    continue

                item = self._queue.claim_next_pending()
                if item is None:
                    # Queue is empty or paused. Block until a producer
                    # signals or the sanity timeout fires.
                    self._wakeup.wait(timeout=self._idle_poll_seconds)
                    self._wakeup.clear()
                    continue

                logger.info(
                    "Queue runner: processing item %s (retry=%d, kind=%s)",
                    item.id,
                    item.retry_count,
                    item.payload.kind,
                )
                self._process_one(item)
        except Exception as exc:
            # The loop body should swallow per-item failures, so
            # reaching here means something pathological. Route through
            # the centralized background-error logger (boundary policy)
            # so the traceback shows up exactly once.
            log_background_exception(_BACKGROUND_TASK_NAME, exc)

    def _sleep_shutdown_aware(self, seconds: float) -> None:
        """Sleep that wakes immediately on shutdown."""
        self._shutdown.wait(timeout=seconds)

    def _process_one(self, item: QueueItem) -> None:
        """Run one generation and apply the resulting state transition.

        Wrapped in a per-item try/except so any single-item failure
        (including unexpected exceptions from inside the dispatch
        wrapper) doesn't take down the whole runner. The transition
        methods are themselves no-ops on stale state (they'd raise),
        which we catch and log so the runner never abandons the loop.
        """
        try:
            result = self._dispatch_fn(item.payload)
        except Exception as exc:
            # The dispatch wrapper is expected to normalize every outcome
            # (including its own errors) into a QueueRunResult; reaching
            # here means the wrapper itself blew up. Treat it as a
            # failed item so the user sees something rather than a
            # silently-stuck running one.
            self._safe_fail(item.id, f"Queue dispatch raised: {exc}")
            return

        if result.status == "busy":
            # Slot was taken between our pre-check and start. Put the
            # item back at the head of pending (no retry consumed) and
            # back off so we don't spin.
            self._safe_requeue_blocked(item.id)
            self._sleep_shutdown_aware(self._slot_busy_poll_seconds)
        elif result.status == "complete":
            output = result.output_path or ""
            self._safe_complete(item.id, output)
        elif result.status == "cancelled":
            self._safe_cancel(item.id)
        else:  # "failed"
            self._safe_fail(item.id, result.error or "Generation failed")

    def _safe_complete(self, item_id: str, output_path: str) -> None:
        try:
            self._queue.complete_running(item_id, output_path)
        except Exception as exc:
            log_background_exception(
                f"{_BACKGROUND_TASK_NAME}:complete[{item_id}]", exc
            )

    def _safe_fail(self, item_id: str, error: str) -> None:
        try:
            self._queue.fail_running(item_id, error)
        except Exception as exc:
            log_background_exception(
                f"{_BACKGROUND_TASK_NAME}:fail[{item_id}]", exc
            )

    def _safe_cancel(self, item_id: str) -> None:
        try:
            self._queue.cancel_running(item_id)
        except Exception as exc:
            log_background_exception(
                f"{_BACKGROUND_TASK_NAME}:cancel[{item_id}]", exc
            )

    def _safe_requeue_blocked(self, item_id: str) -> None:
        try:
            self._queue.requeue_blocked(item_id)
        except Exception as exc:
            log_background_exception(
                f"{_BACKGROUND_TASK_NAME}:requeue[{item_id}]", exc
            )


# A minimal structural protocol so `QueueRunner` can depend on the
# handler surface it uses (`wakeup_event` + the runner-facing
# transitions) without importing the concrete `QueueHandler` — keeps
# the seam testable and avoids an import cycle.
class QueueHandlerLike(Protocol):
    @property
    def wakeup_event(self) -> threading.Event: ...

    def claim_next_pending(self) -> QueueItem | None: ...

    def complete_running(self, item_id: str, output_path: str) -> QueueItem: ...

    def fail_running(self, item_id: str, error: str) -> QueueItem: ...

    def cancel_running(self, item_id: str) -> QueueItem: ...

    def requeue_blocked(self, item_id: str) -> QueueItem: ...


__all__ = [
    "QueueDispatchCallable",
    "QueueHandlerLike",
    "QueueRunResult",
    "QueueRunStatus",
    "QueueRunner",
    "QueueSlotFreeCallable",
]

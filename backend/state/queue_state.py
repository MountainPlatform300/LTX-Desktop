"""Persistent queue state schema.

The queue is a durable batch generation ledger persisted to
`APP_DATA_DIR/queue.json`. Each item is a fully self-contained snapshot
of a discriminated `payload` (video or image `Generate*Request`) plus
tracking metadata (status, timestamps, errors, retry count, originating
project, source).

Mutation rules (enforced by `QueueHandler`, documented here):
  - Pending items may be reordered, edited (payload), cancelled, or
    removed.
  - A running item is owned by the `QueueRunner`; only the runner-side
    handler entry points (`claim_next_pending`, `complete_running`,
    `fail_running`, `cancel_running`, `requeue_blocked`) move an item
    out of `running`.
  - Completed / failed / cancelled items are immutable; the only valid
    transition is removal via `clear_completed` / `clear_failed` /
    `remove_item`.

Retry policy:
  - On the first failure of a running item, `retry_count` goes 0 -> 1
    and the item is re-prepended to pending so it runs immediately
    before any other pending work (per-user UX choice — flaky transient
    OOM should self-heal without operator intervention).
  - On the second failure (`retry_count == 1`), the item is marked
    `failed` with the second-attempt error message; the first-attempt
    error is intentionally not stored so items that succeed on retry
    don't show a stale red flag in the UI.
  - A "busy" outcome (the single-flight GPU slot was taken by a
    non-queue surface like retake / IC-LoRA when the runner tried to
    start) is NOT a failure: `requeue_blocked` puts the item back to
    pending at the head without consuming a retry.

Everything in this module is JSON-serializable so the on-disk format
matches the in-memory model 1:1 — no separate persistence DTO. The
`payload` field reuses the canonical discriminated `QueuePayloadApi`
from `api_types` so the on-disk, in-memory, and HTTP shapes of a
request snapshot can never drift.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from api_types import (
    EnqueueQueueItemRequest,
    ImageQueuePayload,
    QueueItemApi,
    QueueItemKindApi,
    QueuePayloadApi,
    QueueStateResponse,
    VideoQueuePayload,
)


QueueItemStatus = Literal["pending", "running", "completed", "failed", "cancelled"]

# Where the item was authored. Drives the source filter in the queue UI
# and is useful for telemetry on which authoring path users prefer.
# `genspace` covers the GenSpace Generate flow (after the queue
# migration, every Generate click enqueues here). `queue_manual` is the
# in-panel "Add prompts" flow. `gemini_brainstorm` items come from a
# brainstorm-driven prompt generator.
QueueItemSource = Literal["genspace", "queue_manual", "gemini_brainstorm"]

# Re-exported so the runner / handler can pattern-match on the same
# literal the API uses without importing the `*Api` alias name.
QueueItemKind = QueueItemKindApi

# Internal aliases mirroring the API discriminated union. They are the
# same pydantic models — aliases exist so internal code reads as
# snake_case-adjacent Python while the wire format stays camelCase.
QueuePayload = QueuePayloadApi


class QueueItem(BaseModel):
    model_config = ConfigDict(strict=True)

    id: str
    status: QueueItemStatus
    # ISO 8601 UTC timestamps. Stored as strings (not datetime) so the
    # JSON file is human-readable and round-trips through any future
    # tooling without a Pydantic-aware reader.
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    # Project that owns this item's output. May be None if the item was
    # authored without an active project (manual queue entry from the
    # side panel before picking a project) — such items still render but
    # the asset router leaves them in the global outputs dir.
    originating_project_id: str | None = None
    # Fully self-contained, discriminated request snapshot at enqueue
    # time. Per-item params (resolution, duration, fps, turbo, image,
    # width/height/steps, etc.) live here; there are no queue-level
    # shared params.
    payload: QueuePayload
    output_path: str | None = None
    error: str | None = None
    retry_count: int = Field(default=0, ge=0, le=1)
    source: QueueItemSource = "genspace"


class QueueState(BaseModel):
    model_config = ConfigDict(strict=True)

    items: list[QueueItem] = Field(default_factory=list[QueueItem])
    # When True, the runner stops claiming new pending items. The
    # currently-running item (if any) finishes naturally. Drained from
    # disk on boot so an overnight queue paused before a crash stays
    # paused after recovery.
    paused: bool = False
    # Bumped only on breaking schema changes. P1 ships at 1; future
    # changes either include a migration or mark old items as
    # `failed("schema_incompatible")` at load time.
    schema_version: int = 1


class EnqueueRequest(BaseModel):
    """Typed input contract for `QueueHandler.enqueue` / `enqueue_batch`.

    Snake-case to match the rest of the persistence layer; the HTTP
    routes convert from the camel-case `EnqueueQueueItemRequest` via
    `enqueue_request_from_api` below.
    """

    model_config = ConfigDict(strict=True)

    payload: QueuePayload
    originating_project_id: str | None = None
    source: QueueItemSource = "genspace"


# ----------------------------------------------------------------
# Persistence <-> API boundary converters
# ----------------------------------------------------------------
# Two-layer model: snake_case here for on-disk + Python internals,
# camelCase in `api_types` for the HTTP surface. Converting at the
# route layer keeps a rename on either side a one-file change.


def queue_item_to_api(item: QueueItem) -> QueueItemApi:
    return QueueItemApi(
        id=item.id,
        status=item.status,
        createdAt=item.created_at,
        startedAt=item.started_at,
        completedAt=item.completed_at,
        originatingProjectId=item.originating_project_id,
        payload=item.payload,
        outputPath=item.output_path,
        error=item.error,
        retryCount=item.retry_count,
        source=item.source,
    )


def queue_state_to_api(state: QueueState) -> QueueStateResponse:
    return QueueStateResponse(
        items=[queue_item_to_api(item) for item in state.items],
        paused=state.paused,
        schemaVersion=state.schema_version,
    )


def enqueue_request_from_api(req: EnqueueQueueItemRequest) -> EnqueueRequest:
    return EnqueueRequest(
        payload=req.payload,
        originating_project_id=req.originatingProjectId,
        source=req.source,
    )


__all__ = [
    "EnqueueRequest",
    "ImageQueuePayload",
    "QueueItem",
    "QueueItemKind",
    "QueueItemSource",
    "QueueItemStatus",
    "QueuePayload",
    "QueueState",
    "VideoQueuePayload",
    "enqueue_request_from_api",
    "queue_item_to_api",
    "queue_state_to_api",
]

"""Route handlers for the durable batch generation queue.

Endpoints:
    GET    /api/queue                       — full queue state snapshot
    POST   /api/queue/items                 — enqueue one item
    POST   /api/queue/items/batch           — enqueue many items at once
    GET    /api/queue/items                 — list items (optional ?status=)
    GET    /api/queue/items/{id}            — fetch one item
    PATCH  /api/queue/items/{id}            — edit a pending item's payload
    DELETE /api/queue/items/{id}            — remove non-running item
    POST   /api/queue/items/{id}/cancel     — cancel pending item
    POST   /api/queue/reorder               — replace pending order with permutation
    POST   /api/queue/pause                 — pause runner claims
    POST   /api/queue/resume                — resume runner claims
    POST   /api/queue/clear-completed       — purge completed items
    POST   /api/queue/clear-failed          — purge failed items

The routes are intentionally thin: state-shape conversion happens at the
boundary (`queue_item_to_api` / `enqueue_request_from_api`), typed-error
translation maps `QueueItemNotFoundError` / `QueueItemTransitionError` /
`QueuePayloadValidationError` to `HTTPError(404|409|422, code=...)`, and
everything else is `QueueHandler` calls. See `handlers/queue_handler.py`
for the mutation rules and `state/queue_state.py` for the schema.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from _routes._errors import HTTPError
from api_types import (
    ClearQueueResponse,
    EnqueueQueueBatchRequest,
    EnqueueQueueItemRequest,
    QueueItemApi,
    QueueItemStatusApi,
    QueueStateResponse,
    ReorderQueueRequest,
    UpdateQueueItemRequest,
)
from app_handler import AppHandler
from handlers.queue_handler import (
    QueueItemNotFoundError,
    QueueItemTransitionError,
    QueuePayloadValidationError,
)
from state import get_state_service
from state.queue_state import (
    enqueue_request_from_api,
    queue_item_to_api,
    queue_state_to_api,
)

router = APIRouter(prefix="/api/queue", tags=["queue"])


# ----------------------------------------------------------------
# Read endpoints
# ----------------------------------------------------------------


@router.get("", response_model=QueueStateResponse)
def route_get_queue(
    handler: AppHandler = Depends(get_state_service),
) -> QueueStateResponse:
    """GET /api/queue — full queue state snapshot.

    Returned shape mirrors `queue.json` 1:1 (camelCased) so the side
    panel can drive its entire render off a single payload. Polling
    callers should hit this rather than `/api/queue/items` to avoid
    racing the `paused` flag against the items list.
    """
    return queue_state_to_api(handler.queue.get_state())


@router.get("/items", response_model=list[QueueItemApi])
def route_list_queue_items(
    status: QueueItemStatusApi | None = Query(default=None),
    handler: AppHandler = Depends(get_state_service),
) -> list[QueueItemApi]:
    """GET /api/queue/items — list items, optionally filtered by status.

    The frontend uses the unfiltered `/api/queue` response for the
    main panel render; this endpoint exists mostly for diagnostics
    and status-bucket-specific UI affordances (e.g., a "Failed
    items" admin view).
    """
    items = handler.queue.list_items(status=status)
    return [queue_item_to_api(item) for item in items]


@router.get("/items/{item_id}", response_model=QueueItemApi)
def route_get_queue_item(
    item_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> QueueItemApi:
    """GET /api/queue/items/{id} — fetch one item.

    Used by the generation hook to track an enqueued item's state from
    pending -> running -> completed / failed / cancelled without having
    to scan the full queue payload on every poll tick.
    """
    item = handler.queue.get_item(item_id)
    if item is None:
        raise HTTPError(
            404,
            f"Queue item not found: {item_id}",
            code="QUEUE_ITEM_NOT_FOUND",
        )
    return queue_item_to_api(item)


# ----------------------------------------------------------------
# Enqueue / mutate
# ----------------------------------------------------------------


@router.post("/items", response_model=QueueItemApi)
def route_enqueue_item(
    req: EnqueueQueueItemRequest,
    handler: AppHandler = Depends(get_state_service),
) -> QueueItemApi:
    """POST /api/queue/items — enqueue a single item.

    The Generate button in GenSpace lands here; the renderer keeps the
    returned item's id and polls `/api/queue/items/{id}` for status. The
    runner picks the item up via the wakeup event set inside
    `QueueHandler.enqueue`. The payload is validated at the boundary so
    malformed requests get an immediate 422 instead of failing two
    attempts later inside the runner.
    """
    try:
        item = handler.queue.enqueue(enqueue_request_from_api(req))
    except QueuePayloadValidationError as exc:
        raise HTTPError(422, str(exc), code="QUEUE_PAYLOAD_INVALID") from None
    return queue_item_to_api(item)


@router.post("/items/batch", response_model=list[QueueItemApi])
def route_enqueue_batch(
    req: EnqueueQueueBatchRequest,
    handler: AppHandler = Depends(get_state_service),
) -> list[QueueItemApi]:
    """POST /api/queue/items/batch — enqueue many items in one call.

    A single transaction (one `queue.json` write, one wakeup event)
    makes landing 20-50 prompts at once cheap enough that the manual
    multi-prompt entry flow and the brainstorm auto-enqueue flow can
    both enqueue everything synchronously without a noticeable delay.
    Validation is all-or-nothing: one bad payload rejects the batch.
    """
    drafts = [enqueue_request_from_api(item) for item in req.items]
    try:
        items = handler.queue.enqueue_batch(drafts)
    except QueuePayloadValidationError as exc:
        raise HTTPError(422, str(exc), code="QUEUE_PAYLOAD_INVALID") from None
    return [queue_item_to_api(item) for item in items]


@router.patch("/items/{item_id}", response_model=QueueItemApi)
def route_update_queue_item(
    item_id: str,
    body: UpdateQueueItemRequest,
    handler: AppHandler = Depends(get_state_service),
) -> QueueItemApi:
    """PATCH /api/queue/items/{id} — edit a pending item's payload.

    The side panel uses this to let users tweak a queued prompt
    (typo fix, parameter change) without losing its place in the
    queue. Editing is only valid while the item is pending: once
    the runner has claimed it, the snapshot is in flight and we
    return 409. Other terminal states (completed / failed /
    cancelled) likewise reject — re-enqueue is the right path.
    """
    try:
        item = handler.queue.update_pending_item(item_id, body.payload)
    except QueuePayloadValidationError as exc:
        raise HTTPError(422, str(exc), code="QUEUE_PAYLOAD_INVALID") from None
    except QueueItemNotFoundError:
        raise HTTPError(
            404,
            f"Queue item not found: {item_id}",
            code="QUEUE_ITEM_NOT_FOUND",
        ) from None
    except QueueItemTransitionError as exc:
        raise HTTPError(
            409,
            str(exc),
            code="QUEUE_ITEM_INVALID_TRANSITION",
        ) from None
    return queue_item_to_api(item)


@router.delete("/items/{item_id}", status_code=204)
def route_remove_queue_item(
    item_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> None:
    """DELETE /api/queue/items/{id} — remove a non-running item.

    Running items must be cancelled first (via /api/generate/cancel,
    which the runner observes and routes through `cancel_running`).
    The 409 here protects against a race where the user hits Delete
    on a row that just transitioned from pending to running between
    panel render and click.
    """
    try:
        handler.queue.remove_item(item_id)
    except QueueItemNotFoundError:
        raise HTTPError(
            404,
            f"Queue item not found: {item_id}",
            code="QUEUE_ITEM_NOT_FOUND",
        ) from None
    except QueueItemTransitionError as exc:
        raise HTTPError(
            409,
            str(exc),
            code="QUEUE_ITEM_INVALID_TRANSITION",
        ) from None


@router.post("/items/{item_id}/cancel", response_model=QueueItemApi)
def route_cancel_queue_item(
    item_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> QueueItemApi:
    """POST /api/queue/items/{id}/cancel — cancel a pending item.

    Only valid for `pending` items; running items go through
    `/api/generate/cancel` (existing route) which lets the inference
    loop unwind cleanly. Returns 409 if the item is already running
    or terminal.
    """
    try:
        item = handler.queue.cancel_item(item_id)
    except QueueItemNotFoundError:
        raise HTTPError(
            404,
            f"Queue item not found: {item_id}",
            code="QUEUE_ITEM_NOT_FOUND",
        ) from None
    except QueueItemTransitionError as exc:
        raise HTTPError(
            409,
            str(exc),
            code="QUEUE_ITEM_INVALID_TRANSITION",
        ) from None
    return queue_item_to_api(item)


@router.post("/reorder", response_model=list[QueueItemApi])
def route_reorder_queue(
    req: ReorderQueueRequest,
    handler: AppHandler = Depends(get_state_service),
) -> list[QueueItemApi]:
    """POST /api/queue/reorder — replace pending order with a permutation.

    Body is the full new ordering of currently-pending item ids; the
    handler validates that the request is exactly a permutation
    (same set of ids, same length) and rejects partial moves so the
    frontend doesn't have to reason about intermediate states.
    """
    try:
        items = handler.queue.reorder_pending(req.itemIds)
    except QueueItemTransitionError as exc:
        raise HTTPError(
            409,
            str(exc),
            code="QUEUE_REORDER_INVALID",
        ) from None
    return [queue_item_to_api(item) for item in items]


# ----------------------------------------------------------------
# Pause / resume / clear
# ----------------------------------------------------------------


@router.post("/pause", response_model=QueueStateResponse)
def route_pause_queue(
    handler: AppHandler = Depends(get_state_service),
) -> QueueStateResponse:
    """POST /api/queue/pause — block the runner from claiming new items.

    Idempotent. The currently-running item (if any) finishes
    naturally; pause just prevents the next claim. Survives across
    process restarts because the flag lives in queue.json.
    """
    return queue_state_to_api(handler.queue.pause())


@router.post("/resume", response_model=QueueStateResponse)
def route_resume_queue(
    handler: AppHandler = Depends(get_state_service),
) -> QueueStateResponse:
    """POST /api/queue/resume — let the runner claim again.

    Sets the wakeup event so the runner doesn't wait for the next
    enqueue or for the idle-poll timeout to fire.
    """
    return queue_state_to_api(handler.queue.resume())


@router.post("/clear-completed", response_model=ClearQueueResponse)
def route_clear_completed(
    handler: AppHandler = Depends(get_state_service),
) -> ClearQueueResponse:
    """POST /api/queue/clear-completed — purge completed items."""
    return ClearQueueResponse(cleared=handler.queue.clear_completed())


@router.post("/clear-failed", response_model=ClearQueueResponse)
def route_clear_failed(
    handler: AppHandler = Depends(get_state_service),
) -> ClearQueueResponse:
    """POST /api/queue/clear-failed — purge failed items.

    Failed items are kept around until explicitly cleared so the
    user can see what went wrong and decide whether to re-enqueue
    them manually (the auto-retry policy already had its one chance
    on each).
    """
    return ClearQueueResponse(cleared=handler.queue.clear_failed())

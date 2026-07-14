"""Integration tests for /api/queue/* endpoints.

Scope is the route layer: shape of requests and responses, status codes
for typed errors, and the snake_case <-> camelCase boundary conversion
via `queue_item_to_api` / `enqueue_request_from_api`. The underlying
state machine and persistence are exhaustively covered in
`tests/test_queue_handler.py`; these tests don't re-prove the
model-level invariants — just that the HTTP surface mirrors them.

The conftest `client` fixture fires the FastAPI lifespan and starts the
QueueRunner. Tests that observe pre-claim state call `_stop_runner` to
freeze the queue; tests that need a `running` item claim it directly on
the handler (bypassing the runner) so the assertion is deterministic.
"""

from __future__ import annotations

from typing import Any

import pytest

from handlers.queue_handler import QueuePayloadValidationError
from state import get_state_service


def _video_body(prompt: str = "test", **overrides: Any) -> dict:
    body: dict = {"payload": {"kind": "video", "request": {"prompt": prompt}}}
    body.update(overrides)
    return body


def _image_body(prompt: str = "test") -> dict:
    return {"payload": {"kind": "image", "request": {"prompt": prompt}}}


def _stop_runner(test_state) -> None:
    """Freeze the queue so pre-claim state can be snapshot deterministically."""
    test_state.queue_runner.stop()


def _claim_first(test_state) -> str:
    """Mark the oldest pending item running (bypassing the runner)."""
    item = test_state.queue.claim_next_pending()
    assert item is not None, "nothing pending to claim"
    return item.id


# =====================================================================
# GET /api/queue
# =====================================================================


class TestGetQueue:
    def test_empty_state(self, client):
        r = client.get("/api/queue")
        assert r.status_code == 200
        assert r.json() == {"items": [], "paused": False, "schemaVersion": 1}

    def test_after_enqueue(self, client, test_state):
        _stop_runner(test_state)
        r = client.post("/api/queue/items", json=_video_body("a"))
        assert r.status_code == 200

        r = client.get("/api/queue")
        data = r.json()
        assert len(data["items"]) == 1
        assert data["items"][0]["status"] == "pending"
        assert data["items"][0]["payload"]["kind"] == "video"
        assert data["items"][0]["payload"]["request"]["prompt"] == "a"


# =====================================================================
# Enqueue / batch / get item
# =====================================================================


class TestEnqueueRoutes:
    def test_enqueue_image_payload(self, client, test_state):
        _stop_runner(test_state)
        r = client.post("/api/queue/items", json=_image_body("a dog"))
        assert r.status_code == 200
        item = r.json()
        assert item["payload"]["kind"] == "image"
        assert item["payload"]["request"]["prompt"] == "a dog"

    def test_enqueue_records_project_and_source(self, client, test_state):
        _stop_runner(test_state)
        r = client.post(
            "/api/queue/items",
            json={
                "payload": {"kind": "video", "request": {"prompt": "a"}},
                "originatingProjectId": "proj-1",
                "source": "queue_manual",
            },
        )
        assert r.status_code == 200
        item = r.json()
        assert item["originatingProjectId"] == "proj-1"
        assert item["source"] == "queue_manual"

    def test_enqueue_invalid_prompt_returns_422(self, client):
        # Empty prompt violates NonEmptyPrompt at the pydantic boundary.
        r = client.post(
            "/api/queue/items",
            json={"payload": {"kind": "video", "request": {"prompt": "   "}}},
        )
        assert r.status_code == 422

    def test_enqueue_payload_validation_maps_to_422(self, client, test_state):
        # Override the handler dependency with one whose queue raises the
        # handler-level validation error, proving the route maps it to 422
        # (distinct from the 409 transition mapping and the pydantic 422).
        class _BadQueue:
            def enqueue(self, _draft):
                raise QueuePayloadValidationError("spec rejected")

        class _StubHandler:
            queue = _BadQueue()

        app = client.app
        app.dependency_overrides[get_state_service] = lambda: _StubHandler()
        try:
            r = client.post("/api/queue/items", json=_video_body("a"))
            assert r.status_code == 422
            assert r.json()["code"] == "QUEUE_PAYLOAD_INVALID"
        finally:
            app.dependency_overrides.pop(get_state_service, None)

    def test_batch_enqueue(self, client, test_state):
        _stop_runner(test_state)
        r = client.post(
            "/api/queue/items/batch",
            json={
                "items": [
                    _video_body("a"),
                    _image_body("b"),
                ]
            },
        )
        assert r.status_code == 200
        items = r.json()
        assert [i["payload"]["kind"] for i in items] == ["video", "image"]

    def test_batch_empty_rejected(self, client):
        r = client.post("/api/queue/items/batch", json={"items": []})
        assert r.status_code == 422  # min_length=1

    def test_get_item_and_missing(self, client, test_state):
        _stop_runner(test_state)
        created = client.post("/api/queue/items", json=_video_body("a")).json()
        r = client.get(f"/api/queue/items/{created['id']}")
        assert r.status_code == 200
        assert r.json()["id"] == created["id"]
        r = client.get("/api/queue/items/no-such-item")
        assert r.status_code == 404
        assert r.json()["code"] == "QUEUE_ITEM_NOT_FOUND"

    def test_list_items_filtered_by_status(self, client, test_state):
        _stop_runner(test_state)
        client.post("/api/queue/items", json=_video_body("a"))
        client.post("/api/queue/items", json=_video_body("b"))
        _claim_first(test_state)  # one running
        r = client.get("/api/queue/items", params={"status": "pending"})
        assert r.status_code == 200
        assert len(r.json()) == 1
        r = client.get("/api/queue/items", params={"status": "running"})
        assert len(r.json()) == 1


# =====================================================================
# Mutations
# =====================================================================


class TestMutationRoutes:
    def test_patch_pending(self, client, test_state):
        _stop_runner(test_state)
        item = client.post("/api/queue/items", json=_video_body("a")).json()
        r = client.patch(
            f"/api/queue/items/{item['id']}",
            json={"payload": {"kind": "video", "request": {"prompt": "b"}}},
        )
        assert r.status_code == 200
        assert r.json()["payload"]["request"]["prompt"] == "b"

    def test_patch_running_returns_409(self, client, test_state):
        _stop_runner(test_state)
        item = client.post("/api/queue/items", json=_video_body("a")).json()
        _claim_first(test_state)
        r = client.patch(
            f"/api/queue/items/{item['id']}",
            json={"payload": {"kind": "video", "request": {"prompt": "b"}}},
        )
        assert r.status_code == 409
        assert r.json()["code"] == "QUEUE_ITEM_INVALID_TRANSITION"

    def test_delete_pending(self, client, test_state):
        _stop_runner(test_state)
        item = client.post("/api/queue/items", json=_video_body("a")).json()
        r = client.delete(f"/api/queue/items/{item['id']}")
        assert r.status_code == 204
        assert client.get("/api/queue").json()["items"] == []

    def test_delete_running_returns_409(self, client, test_state):
        _stop_runner(test_state)
        item = client.post("/api/queue/items", json=_video_body("a")).json()
        _claim_first(test_state)
        r = client.delete(f"/api/queue/items/{item['id']}")
        assert r.status_code == 409

    def test_cancel_pending(self, client, test_state):
        _stop_runner(test_state)
        item = client.post("/api/queue/items", json=_video_body("a")).json()
        r = client.post(f"/api/queue/items/{item['id']}/cancel")
        assert r.status_code == 200
        assert r.json()["status"] == "cancelled"

    def test_cancel_running_returns_409(self, client, test_state):
        _stop_runner(test_state)
        item = client.post("/api/queue/items", json=_video_body("a")).json()
        _claim_first(test_state)
        r = client.post(f"/api/queue/items/{item['id']}/cancel")
        assert r.status_code == 409

    def test_reorder_valid(self, client, test_state):
        _stop_runner(test_state)
        a = client.post("/api/queue/items", json=_video_body("a")).json()
        b = client.post("/api/queue/items", json=_video_body("b")).json()
        c = client.post("/api/queue/items", json=_video_body("c")).json()
        r = client.post("/api/queue/reorder", json={"itemIds": [c["id"], b["id"], a["id"]]})
        assert r.status_code == 200
        assert [i["id"] for i in r.json()] == [c["id"], b["id"], a["id"]]

    def test_reorder_invalid_returns_409(self, client, test_state):
        _stop_runner(test_state)
        a = client.post("/api/queue/items", json=_video_body("a")).json()
        client.post("/api/queue/items", json=_video_body("b")).json()
        r = client.post("/api/queue/reorder", json={"itemIds": [a["id"]]})
        assert r.status_code == 409
        assert r.json()["code"] == "QUEUE_REORDER_INVALID"


# =====================================================================
# Pause / resume / clear
# =====================================================================


class TestPauseResumeClear:
    def test_pause_then_resume(self, client):
        r = client.post("/api/queue/pause")
        assert r.status_code == 200
        assert r.json()["paused"] is True
        r = client.post("/api/queue/resume")
        assert r.status_code == 200
        assert r.json()["paused"] is False

    def test_clear_completed_and_failed(self, client, test_state):
        _stop_runner(test_state)
        a = client.post("/api/queue/items", json=_video_body("a")).json()
        b = client.post("/api/queue/items", json=_video_body("b")).json()
        # Drive a -> completed, b -> failed via the handler directly.
        test_state.queue.claim_next_pending()
        test_state.queue.complete_running(a["id"], "/tmp/a.mp4")
        test_state.queue.claim_next_pending()
        test_state.queue.fail_running(b["id"], "boom")
        test_state.queue.claim_next_pending()
        test_state.queue.fail_running(b["id"], "boom2")

        r = client.post("/api/queue/clear-completed")
        assert r.status_code == 200
        assert r.json()["cleared"] == 1
        r = client.post("/api/queue/clear-failed")
        assert r.status_code == 200
        assert r.json()["cleared"] == 1
        assert client.get("/api/queue").json()["items"] == []

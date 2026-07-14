"""Tests for /health and /api/gpu-info endpoints."""

from starlette.testclient import TestClient

from _routes import health as health_routes
from app_factory import create_app
from state.app_state_types import GpuSlot, VideoPipelineState
from tests.fakes.services import FakeFastVideoPipeline
from tests.http_error_assertions import assert_http_error


def _set_video_pipeline(state):
    state.state.gpu_slot = GpuSlot(
        active_pipeline=VideoPipelineState(
            pipeline=FakeFastVideoPipeline(),
            is_compiled=False,
        ),
    )


class TestHealth:
    def test_no_models_loaded(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["models_loaded"] is False
        assert data["active_model"] is None

    def test_fast_model_loaded(self, client, test_state):
        _set_video_pipeline(test_state)
        r = client.get("/health")
        data = r.json()
        assert data["models_loaded"] is True
        assert data["active_model"] == "fast"
        assert data["models_loaded"] is True

    def test_models_downloaded(self, client, create_fake_model_files):
        create_fake_model_files()
        r = client.get("/health")
        data = r.json()
        assert len(data["models_status"]) == 1
        assert data["models_status"][0]["downloaded"] is True

    def test_cors_header(self, client):
        r = client.get("/health", headers={"Origin": "http://localhost:5173"})
        assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


class TestGpuInfo:
    def test_no_gpu(self, client, test_state):
        test_state.gpu_info.cuda_available = False
        test_state.gpu_info.mps_available = False
        test_state.gpu_info.gpu_name = None
        test_state.gpu_info.vram_gb = None
        test_state.gpu_info.gpu_info = {"name": "Unknown", "vram": 0, "vramUsed": 0}

        r = client.get("/api/gpu-info")
        assert r.status_code == 200
        data = r.json()
        assert data["cuda_available"] is False
        assert data["mps_available"] is False
        assert data["gpu_available"] is False
        assert data["gpu_name"] is None
        assert data["vram_gb"] is None

    def test_with_cuda(self, client, test_state):
        test_state.gpu_info.cuda_available = True
        test_state.gpu_info.mps_available = False
        test_state.gpu_info.gpu_name = "RTX 5090"
        test_state.gpu_info.vram_gb = 32
        test_state.gpu_info.gpu_info = {"name": "Test GPU", "vram": 8192, "vramUsed": 1024}

        r = client.get("/api/gpu-info")
        assert r.status_code == 200
        data = r.json()
        assert data["cuda_available"] is True
        assert data["mps_available"] is False
        assert data["gpu_available"] is True
        assert data["gpu_name"] == "RTX 5090"
        assert data["vram_gb"] == 32

    def test_with_mps(self, client, test_state):
        test_state.gpu_info.cuda_available = False
        test_state.gpu_info.mps_available = True
        test_state.gpu_info.gpu_name = "Apple Silicon (MPS)"
        test_state.gpu_info.vram_gb = 36
        test_state.gpu_info.gpu_info = {"name": "Apple Silicon (MPS)", "vram": 36864, "vramUsed": 0}

        r = client.get("/api/gpu-info")
        assert r.status_code == 200
        data = r.json()
        assert data["cuda_available"] is False
        assert data["mps_available"] is True
        assert data["gpu_available"] is True
        assert data["gpu_name"] == "Apple Silicon (MPS)"
        assert data["vram_gb"] == 36


class TestShutdown:
    def test_managed_backend_requires_admin_token(self, test_state, monkeypatch):
        called = False

        def fake_shutdown() -> None:
            nonlocal called
            called = True

        monkeypatch.setattr(health_routes, "_shutdown_process", fake_shutdown)
        app = create_app(handler=test_state, admin_token="test-admin-token")

        with TestClient(app, client=("127.0.0.1", 50000)) as local_client:
            response = local_client.post("/api/system/shutdown")

        assert_http_error(
            response,
            status_code=403,
            code="HTTP_403",
            message="Admin token required",
        )
        assert called is False

    def test_managed_backend_accepts_admin_token(self, test_state, monkeypatch):
        called = False

        def fake_shutdown() -> None:
            nonlocal called
            called = True

        monkeypatch.setattr(health_routes, "_shutdown_process", fake_shutdown)
        app = create_app(handler=test_state, admin_token="test-admin-token")

        with TestClient(app, client=("127.0.0.1", 50000)) as local_client:
            response = local_client.post(
                "/api/system/shutdown",
                headers={"X-Admin-Token": "test-admin-token"},
            )

        assert response.status_code == 200
        assert response.json() == {"status": "shutting_down"}
        assert called is True

    def test_explicit_insecure_backend_remains_compatible(self, test_state, monkeypatch):
        called = False

        def fake_shutdown() -> None:
            nonlocal called
            called = True

        monkeypatch.setattr(health_routes, "_shutdown_process", fake_shutdown)
        app = create_app(handler=test_state)

        with TestClient(app, client=("127.0.0.1", 50000)) as local_client:
            response = local_client.post("/api/system/shutdown")

        assert response.status_code == 200
        assert called is True

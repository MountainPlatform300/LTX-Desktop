"""FastAPI app factory decoupled from runtime bootstrap side effects."""

from __future__ import annotations

import base64
import hmac
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response as StarletteResponse

from _routes._errors import HTTPError, build_http_error_response
from _routes.generation import router as generation_router
from _routes.hf_auth import router as hf_auth_router
from _routes.health import router as health_router
from _routes.ic_lora import router as ic_lora_router
from _routes.image_gen import router as image_gen_router
from _routes.image_edit import router as image_edit_router
from _routes.models import router as models_router
from _routes.media import router as media_router
from _routes.lora_training import router as lora_training_router
from _routes.lora_inference import router as lora_inference_router
from _routes.queue import router as queue_router
from _routes.suggest_gap_prompt import router as suggest_gap_prompt_router
from _routes.retake import router as retake_router
from _routes.runtime_policy import router as runtime_policy_router
from _routes.settings import router as settings_router
from api_types import HTTPErrorResponse
from logging_policy import log_http_error, log_unhandled_exception
from state import init_state_service

if TYPE_CHECKING:
    from app_handler import AppHandler

logger = logging.getLogger(__name__)

DEFAULT_ALLOWED_ORIGINS: list[str] = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

DEFAULT_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    "4XX": {
        "model": HTTPErrorResponse,
        "description": "Client Error",
    },
    "5XX": {
        "model": HTTPErrorResponse,
        "description": "Server Error",
    },
}


def create_app(
    *,
    handler: "AppHandler",
    allowed_origins: list[str] | None = None,
    title: str = "LTX-2 Video Generation Server",
    auth_token: str = "",
    admin_token: str = "",
) -> FastAPI:
    """Create a configured FastAPI app bound to the provided handler."""
    init_state_service(handler)

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        # The LoRA-trainer reconciler polls remote GPU jobs. The handler
        # builds the runners in its constructor but defers `start()` to
        # here so a bare AppHandler used outside a server (CLI tools,
        # schema exporters) doesn't leak a daemon thread. Lifespan
        # shutdown below joins the threads cleanly before the process exits.
        handler.lora_training_runner.start()
        # Local sprite/preview generation for the curation gallery; same
        # deferral rationale (no daemon thread for bare AppHandler use).
        handler.lora_clip_jobs_runner.start()
        # Background target/variant derivation pipeline (frame edit -> local
        # IC-LoRA drive or remote Kling); same deferral rationale.
        handler.lora_derivation_runner.start()
        # Durable batch generation queue. The handler loaded `queue.json`
        # (with crash recovery) during AppHandler init; the runner thread
        # is the sole claimer of pending items and cooperates with the
        # other single-flight generation surfaces via the slot-free probe.
        handler.queue_runner.start()
        try:
            yield
        finally:
            try:
                handler.queue_runner.stop()
            except Exception:
                logger.exception("Failed to stop queue runner cleanly")
            try:
                handler.lora_training_runner.stop()
            except Exception:
                logger.exception("Failed to stop LoRA trainer runner cleanly")
            try:
                handler.lora_clip_jobs_runner.stop()
            except Exception:
                logger.exception("Failed to stop LoRA clip-jobs runner cleanly")
            try:
                handler.lora_derivation_runner.stop()
            except Exception:
                logger.exception("Failed to stop LoRA derivation runner cleanly")

    app = FastAPI(title=title, responses=DEFAULT_ERROR_RESPONSES, lifespan=_lifespan)
    app.state.admin_token = admin_token  # type: ignore[attr-defined]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins or DEFAULT_ALLOWED_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _auth_middleware(  # pyright: ignore[reportUnusedFunction]
        request: Request,
        call_next: Callable[[Request], Awaitable[StarletteResponse]],
    ) -> StarletteResponse:
        if not auth_token:
            return await call_next(request)
        if request.method == "OPTIONS":
            return await call_next(request)
        if request.url.path == "/api/auth/huggingface/callback":
            return await call_next(request)
        def _token_matches(candidate: str) -> bool:
            return hmac.compare_digest(candidate, auth_token)

        # WebSocket: check query param
        if request.headers.get("upgrade", "").lower() == "websocket":
            if _token_matches(request.query_params.get("token", "")):
                return await call_next(request)
            return JSONResponse(
                status_code=401,
                content=build_http_error_response(401, "Unauthorized").model_dump(),
            )
        # HTTP: Bearer or Basic auth
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer ") and _token_matches(auth_header[7:]):
            return await call_next(request)
        if auth_header.startswith("Basic "):
            try:
                decoded = base64.b64decode(auth_header[6:]).decode()
                _, _, password = decoded.partition(":")
                if _token_matches(password):
                    return await call_next(request)
            except Exception:
                pass
        return JSONResponse(
            status_code=401,
            content=build_http_error_response(401, "Unauthorized").model_dump(),
        )

    async def _route_http_error_handler(request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, HTTPError):
            log_http_error(request, exc)
            return JSONResponse(status_code=exc.status_code, content=exc.response.model_dump())
        return JSONResponse(
            status_code=500,
            content=build_http_error_response(500, str(exc)).model_dump(),
        )

    async def _starlette_http_error_handler(request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, StarletteHTTPException):
            return JSONResponse(
                status_code=exc.status_code,
                content=build_http_error_response(exc.status_code, exc.detail).model_dump(),
            )
        return JSONResponse(
            status_code=500,
            content=build_http_error_response(500, str(exc)).model_dump(),
        )

    async def _validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, RequestValidationError):
            return JSONResponse(
                status_code=422,
                content=build_http_error_response(422, str(exc)).model_dump(),
            )
        return JSONResponse(
            status_code=422,
            content=build_http_error_response(422, str(exc)).model_dump(),
        )

    async def _route_generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        log_unhandled_exception(request, exc)
        return JSONResponse(
            status_code=500,
            content=build_http_error_response(500, str(exc)).model_dump(),
        )

    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(HTTPError, _route_http_error_handler)
    app.add_exception_handler(StarletteHTTPException, _starlette_http_error_handler)
    app.add_exception_handler(Exception, _route_generic_error_handler)

    app.include_router(health_router)
    app.include_router(generation_router)
    app.include_router(models_router)
    app.include_router(settings_router)
    app.include_router(image_gen_router)
    app.include_router(image_edit_router)
    app.include_router(suggest_gap_prompt_router)
    app.include_router(retake_router)
    app.include_router(ic_lora_router)
    app.include_router(runtime_policy_router)
    app.include_router(hf_auth_router)
    app.include_router(media_router)
    app.include_router(lora_training_router)
    app.include_router(lora_inference_router)
    app.include_router(queue_router)

    return app

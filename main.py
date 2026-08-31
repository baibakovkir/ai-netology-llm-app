import logging
import uuid
from contextlib import asynccontextmanager
from time import perf_counter
from typing import AsyncIterator, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.routes import InputValidationError, create_router, validation_error_body
from cache.ttl_cache import TTLCache
from config.logging_config import configure_logging
from config.settings import Settings
from llm.client import LLMClient
from llm.exceptions import LLMUnavailableError
from services.chat_service import ChatService


FALLBACK_MESSAGE = "Сервис временно недоступен, попробуйте позже"


def create_app(
    settings: Optional[Settings] = None,
    llm_client: Optional[LLMClient] = None,
    cache: Optional[TTLCache[str]] = None,
) -> FastAPI:
    app_settings = settings or Settings.from_env()
    configure_logging(app_settings.log_level, app_settings.log_file)
    logger = logging.getLogger(__name__)
    client = llm_client or LLMClient(app_settings)
    result_cache = cache or TTLCache[str](app_settings.cache_ttl_seconds)
    service = ChatService(app_settings, client, result_cache)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "service_started",
            extra={"model": app_settings.model, "base_url": app_settings.base_url},
        )
        yield
        await client.close()
        logger.info("service_stopped")

    app = FastAPI(
        title="Minimal LLM Service",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = app_settings
    app.state.llm_client = client
    app.state.cache = result_cache

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        started = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "unhandled_request_error",
                extra={"request_id": request_id, "path": request.url.path},
            )
            response = JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "error": {"code": "internal_error", "message": "Внутренняя ошибка сервиса"},
                    "request_id": request_id,
                },
            )
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "http_request_completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round((perf_counter() - started) * 1000, 2),
            },
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        logger.warning(
            "request_validation_error",
            extra={"request_id": request_id, "errors": exc.errors()},
        )
        return JSONResponse(status_code=400, content=validation_error_body(exc.errors(), request_id))

    @app.exception_handler(InputValidationError)
    async def input_exception_handler(request: Request, exc: InputValidationError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        logger.warning("request_validation_error", extra={"request_id": request_id, "error": str(exc)})
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "error": {"code": "validation_error", "message": str(exc)},
                "request_id": request_id,
            },
        )

    @app.exception_handler(LLMUnavailableError)
    async def llm_exception_handler(request: Request, exc: LLMUnavailableError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        logger.error(
            "fallback_returned",
            extra={"request_id": request_id, "error": str(exc)},
        )
        return JSONResponse(
            status_code=503,
            content={
                "status": "fallback",
                "answer": FALLBACK_MESSAGE,
                "cached": False,
                "request_id": request_id,
            },
        )

    app.include_router(create_router(service, app_settings))
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=app.state.settings.host,
        port=app.state.settings.port,
        log_config=None,
    )

import logging
from typing import Any, Dict

from fastapi import APIRouter, Request

from api.schemas import ChatRequest, ChatResponse, HealthResponse
from config.settings import Settings
from services.chat_service import ChatService


logger = logging.getLogger(__name__)


class InputValidationError(ValueError):
    pass


def create_router(service: ChatService, settings: Settings) -> APIRouter:
    router = APIRouter()

    @router.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    @router.post("/chat", response_model=ChatResponse)
    async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        request_id = request.state.request_id
        message = str(payload.message)
        if len(message) > settings.max_message_length:
            raise InputValidationError(f"message: maximum length is {settings.max_message_length}")

        logger.info(
            "incoming_chat_request",
            extra={"request_id": request_id, "user_message": message},
        )
        result = await service.answer(message, request_id)
        logger.info(
            "user_response_formed",
            extra={"request_id": request_id, "cached": result.cached},
        )
        return ChatResponse(answer=result.answer, cached=result.cached, request_id=request_id)

    return router


def validation_error_body(errors: Any, request_id: str) -> Dict[str, Any]:
    messages = []
    for error in errors:
        location = ".".join(str(part) for part in error.get("loc", []) if part != "body")
        description = error.get("msg", "invalid value")
        messages.append(f"{location}: {description}" if location else str(description))
    return {
        "status": "error",
        "error": {
            "code": "validation_error",
            "message": "; ".join(messages) or "Некорректный JSON-запрос",
        },
        "request_id": request_id,
    }

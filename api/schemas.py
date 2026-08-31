from typing import Literal

from pydantic import BaseModel, constr


class ChatRequest(BaseModel):
    message: constr(strict=True, strip_whitespace=True, min_length=1, max_length=1000)  # type: ignore[valid-type]


class ChatResponse(BaseModel):
    status: Literal["ok"] = "ok"
    answer: str
    cached: bool
    request_id: str


class FallbackResponse(BaseModel):
    status: Literal["fallback"] = "fallback"
    answer: str
    cached: bool = False
    request_id: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"

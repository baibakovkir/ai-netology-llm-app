import asyncio
import logging
from time import perf_counter
from typing import Any, Dict, List, Optional

import httpx

from config.settings import Settings
from llm.exceptions import LLMUnavailableError


logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, settings: Settings, http_client: Optional[httpx.AsyncClient] = None) -> None:
        self._settings = settings
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(timeout=settings.timeout_seconds)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def generate(self, messages: List[Dict[str, str]], request_id: str) -> str:
        if not self._settings.api_key:
            logger.error("llm_api_key_missing", extra={"request_id": request_id})
            raise LLMUnavailableError("LLM API key is not configured")

        payload: Dict[str, Any] = {
            "model": self._settings.model,
            "temperature": self._settings.temperature,
            "messages": messages,
        }
        headers = {
            "Authorization": f"Bearer {self._settings.api_key}",
            "Content-Type": "application/json",
        }
        endpoint = f"{self._settings.base_url.rstrip('/')}/chat/completions"

        for attempt in range(1, self._settings.max_attempts + 1):
            started = perf_counter()
            try:
                logger.info(
                    "llm_request_started",
                    extra={"request_id": request_id, "attempt": attempt, "model": self._settings.model},
                )
                response = await self._client.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self._settings.timeout_seconds,
                )
                duration_ms = round((perf_counter() - started) * 1000, 2)

                if response.status_code == 429 or response.status_code >= 500:
                    logger.warning(
                        "llm_temporary_http_error",
                        extra={
                            "request_id": request_id,
                            "attempt": attempt,
                            "status_code": response.status_code,
                            "duration_ms": duration_ms,
                        },
                    )
                    if attempt < self._settings.max_attempts:
                        await self._backoff(attempt)
                        continue
                    raise LLMUnavailableError(f"Temporary provider error: HTTP {response.status_code}")

                if response.status_code >= 400:
                    logger.error(
                        "llm_non_retryable_http_error",
                        extra={
                            "request_id": request_id,
                            "attempt": attempt,
                            "status_code": response.status_code,
                            "duration_ms": duration_ms,
                        },
                    )
                    raise LLMUnavailableError(f"Provider rejected request: HTTP {response.status_code}")

                answer = self._extract_answer(response, request_id)
                logger.info(
                    "llm_response_received",
                    extra={
                        "request_id": request_id,
                        "attempt": attempt,
                        "duration_ms": duration_ms,
                        "model_response": answer,
                    },
                )
                return answer
            except httpx.RequestError as exc:
                duration_ms = round((perf_counter() - started) * 1000, 2)
                logger.warning(
                    "llm_network_error",
                    extra={
                        "request_id": request_id,
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                        "duration_ms": duration_ms,
                    },
                )
                if attempt < self._settings.max_attempts:
                    await self._backoff(attempt)
                    continue
                raise LLMUnavailableError("Provider network error") from exc
            except LLMUnavailableError:
                raise
            except (ValueError, KeyError, TypeError) as exc:
                logger.error(
                    "llm_response_parse_error",
                    extra={"request_id": request_id, "attempt": attempt, "error_type": type(exc).__name__},
                )
                raise LLMUnavailableError("Invalid provider response") from exc

        raise LLMUnavailableError("Provider is unavailable")

    async def _backoff(self, attempt: int) -> None:
        delay = self._settings.retry_base_delay_seconds * (2 ** (attempt - 1))
        if delay:
            await asyncio.sleep(delay)

    @staticmethod
    def _extract_answer(response: httpx.Response, request_id: str) -> str:
        try:
            data = response.json()
            answer = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            logger.error(
                "llm_invalid_json_structure",
                extra={"request_id": request_id, "provider_body": response.text[:2000]},
            )
            raise ValueError("Invalid chat completion response") from exc
        if not isinstance(answer, str):
            raise TypeError("Response content must be a string")
        return answer

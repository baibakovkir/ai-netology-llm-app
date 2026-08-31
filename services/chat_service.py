import hashlib
import json
import logging
from dataclasses import dataclass
from time import perf_counter

from cache.ttl_cache import TTLCache
from config.settings import Settings
from llm.client import LLMClient
from llm.prompts import build_messages


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatResult:
    answer: str
    cached: bool


class ChatService:
    def __init__(self, settings: Settings, client: LLMClient, cache: TTLCache[str]) -> None:
        self._settings = settings
        self._client = client
        self._cache = cache

    async def answer(self, message: str, request_id: str) -> ChatResult:
        stage_started = perf_counter()
        normalized = message.strip()
        cache_key = self._cache_key(normalized)
        cached_answer = self._cache.get(cache_key)
        if cached_answer is not None:
            logger.info(
                "cache_hit",
                extra={"request_id": request_id, "cache_key": cache_key, "duration_ms": self._ms(stage_started)},
            )
            return ChatResult(answer=cached_answer, cached=True)

        logger.info(
            "cache_miss",
            extra={"request_id": request_id, "cache_key": cache_key, "duration_ms": self._ms(stage_started)},
        )
        messages = build_messages(self._settings.system_prompt, normalized)
        logger.info(
            "prompt_built",
            extra={"request_id": request_id, "prompt": messages, "duration_ms": self._ms(stage_started)},
        )
        raw_answer = await self._client.generate(messages, request_id)
        answer = " ".join(raw_answer.split())
        if not answer:
            logger.error("empty_model_answer", extra={"request_id": request_id})
            from llm.exceptions import LLMUnavailableError

            raise LLMUnavailableError("Model returned an empty answer")

        self._cache.set(cache_key, answer)
        logger.info(
            "pipeline_completed",
            extra={"request_id": request_id, "answer": answer, "duration_ms": self._ms(stage_started)},
        )
        return ChatResult(answer=answer, cached=False)

    def _cache_key(self, message: str) -> str:
        value = {
            "message": message,
            "model": self._settings.model,
            "temperature": self._settings.temperature,
            "system_prompt": self._settings.system_prompt,
        }
        canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _ms(started: float) -> float:
        return round((perf_counter() - started) * 1000, 2)


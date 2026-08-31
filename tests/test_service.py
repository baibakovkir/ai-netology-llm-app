import pytest

from cache.ttl_cache import TTLCache
from config.settings import Settings
from llm.exceptions import LLMUnavailableError
from services.chat_service import ChatService


class FakeClient:
    def __init__(self, answer: str = "answer") -> None:
        self.answer = answer
        self.calls = 0

    async def generate(self, messages, request_id: str) -> str:
        self.calls += 1
        return self.answer


def settings(**changes) -> Settings:
    values = {"api_key": "key", "log_file": "/tmp/ai-llm-service-test.log"}
    values.update(changes)
    return Settings(**values)


@pytest.mark.asyncio
async def test_cache_key_changes_with_model_temperature_and_prompt() -> None:
    cache = TTLCache[str](60)

    first_client = FakeClient("first")
    first = ChatService(settings(), first_client, cache)  # type: ignore[arg-type]
    assert (await first.answer("same", "one")).answer == "first"

    for changed in (
        {"model": "another-model"},
        {"temperature": 0.7},
        {"system_prompt": "another prompt"},
    ):
        client = FakeClient("changed")
        service = ChatService(settings(**changed), client, cache)  # type: ignore[arg-type]
        result = await service.answer("same", "changed")
        assert result.cached is False
        assert client.calls == 1


@pytest.mark.asyncio
async def test_empty_postprocessed_answer_is_unavailable() -> None:
    service = ChatService(settings(), FakeClient(" \n\t "), TTLCache[str](60))  # type: ignore[arg-type]
    with pytest.raises(LLMUnavailableError):
        await service.answer("message", "request")


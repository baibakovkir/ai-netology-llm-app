from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio

from config.settings import Settings
from main import create_app


class StubLLMClient:
    def __init__(self, answer: str = "Тестовый ответ") -> None:
        self.answer = answer
        self.calls = 0
        self.messages = []

    async def generate(self, messages: Any, request_id: str) -> str:
        self.calls += 1
        self.messages.append(messages)
        return self.answer

    async def close(self) -> None:
        return None


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        api_key="test-key",
        base_url="https://llm.test/v1",
        retry_base_delay_seconds=0,
        log_file=str(tmp_path / "service.log"),
    )


@pytest.fixture
def stub_client() -> StubLLMClient:
    return StubLLMClient("  Полезный   ответ.  ")


@pytest_asyncio.fixture
async def api_client(settings: Settings, stub_client: StubLLMClient):
    app = create_app(settings=settings, llm_client=stub_client)  # type: ignore[arg-type]
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

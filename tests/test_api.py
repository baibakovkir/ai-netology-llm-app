import json
from pathlib import Path

import pytest

from tests.conftest import StubLLMClient


@pytest.mark.asyncio
async def test_successful_request_and_cache_hit(api_client, stub_client: StubLLMClient) -> None:
    first = await api_client.post("/chat", json={"message": "  Привет!  "})
    second = await api_client.post("/chat", json={"message": "Привет!"})

    assert first.status_code == 200
    assert first.json()["answer"] == "Полезный ответ."
    assert first.json()["cached"] is False
    assert second.status_code == 200
    assert second.json()["cached"] is True
    assert first.json()["request_id"] != second.json()["request_id"]
    assert stub_client.calls == 1
    assert stub_client.messages[0][0]["role"] == "system"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"message": ""},
        {"message": "   "},
        {"message": 42},
        {"message": "x" * 1001},
    ],
)
async def test_invalid_input_returns_400(api_client, payload) -> None:
    response = await api_client.post("/chat", json=payload)

    assert response.status_code == 400
    body = response.json()
    assert body["status"] == "error"
    assert body["error"]["code"] == "validation_error"
    assert body["request_id"]


@pytest.mark.asyncio
async def test_invalid_json_returns_400(api_client) -> None:
    response = await api_client.post(
        "/chat",
        content=b"not-json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "validation_error"


@pytest.mark.asyncio
async def test_health_does_not_need_llm(api_client, stub_client: StubLLMClient) -> None:
    response = await api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert stub_client.calls == 0


@pytest.mark.asyncio
async def test_logs_are_json_and_contain_cache_events(
    api_client, settings, stub_client: StubLLMClient
) -> None:
    await api_client.post("/chat", json={"message": "логирование"})
    await api_client.post("/chat", json={"message": "логирование"})

    lines = Path(settings.log_file).read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    events = {record["event"] for record in records}
    assert "cache_miss" in events
    assert "cache_hit" in events
    assert "prompt_built" in events
    assert all("timestamp" in record and "level" in record for record in records)


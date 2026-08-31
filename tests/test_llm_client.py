from typing import List

import httpx
import pytest

from config.settings import Settings
from llm.client import LLMClient
from llm.exceptions import LLMUnavailableError


MESSAGES = [
    {"role": "system", "content": "system"},
    {"role": "user", "content": "hello"},
]


def make_settings(**changes) -> Settings:
    values = {
        "api_key": "secret",
        "base_url": "https://llm.test/v1",
        "max_attempts": 3,
        "retry_base_delay_seconds": 0,
        "log_file": "/tmp/ai-llm-test.log",
    }
    values.update(changes)
    return Settings(**values)


@pytest.mark.asyncio
async def test_successful_provider_response() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://llm.test/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer secret"
        return httpx.Response(200, json={"choices": [{"message": {"content": "answer"}}]})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = LLMClient(make_settings(), http_client)
    assert await client.generate(MESSAGES, "req-1") == "answer"
    await http_client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("temporary_status", [429, 500, 503])
async def test_temporary_http_errors_are_retried(temporary_status: int) -> None:
    statuses: List[int] = [temporary_status, temporary_status, 200]

    async def handler(_: httpx.Request) -> httpx.Response:
        status = statuses.pop(0)
        if status == 200:
            return httpx.Response(200, json={"choices": [{"message": {"content": "recovered"}}]})
        return httpx.Response(status, json={"error": "temporary"})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = LLMClient(make_settings(), http_client)
    assert await client.generate(MESSAGES, "req-2") == "recovered"
    assert statuses == []
    await http_client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", ["timeout", "network"])
async def test_network_errors_exhaust_retries(error_type: str) -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if error_type == "timeout":
            raise httpx.ReadTimeout("timed out", request=request)
        raise httpx.ConnectError("offline", request=request)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = LLMClient(make_settings(), http_client)
    with pytest.raises(LLMUnavailableError):
        await client.generate(MESSAGES, "req-3")
    assert calls == 3
    await http_client.aclose()


@pytest.mark.asyncio
async def test_regular_4xx_is_not_retried() -> None:
    calls = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"error": "bad request"})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = LLMClient(make_settings(), http_client)
    with pytest.raises(LLMUnavailableError):
        await client.generate(MESSAGES, "req-4")
    assert calls == 1
    await http_client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"not-json"),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json={"choices": [{"message": {"content": 42}}]}),
    ],
)
async def test_invalid_provider_response_is_handled(response: httpx.Response) -> None:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response))
    client = LLMClient(make_settings(), http_client)
    with pytest.raises(LLMUnavailableError):
        await client.generate(MESSAGES, "req-5")
    await http_client.aclose()


@pytest.mark.asyncio
async def test_missing_api_key_returns_unavailable_without_http_call() -> None:
    calls = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = LLMClient(make_settings(api_key=""), http_client)
    with pytest.raises(LLMUnavailableError):
        await client.generate(MESSAGES, "req-6")
    assert calls == 0
    await http_client.aclose()


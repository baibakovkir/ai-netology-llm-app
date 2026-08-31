import httpx
import pytest

from config.settings import Settings
from llm.client import LLMClient
from main import FALLBACK_MESSAGE, create_app


@pytest.mark.asyncio
async def test_missing_key_returns_documented_fallback(tmp_path) -> None:
    settings = Settings(api_key="", log_file=str(tmp_path / "fallback.log"))
    provider_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(500)))
    llm_client = LLMClient(settings, provider_client)
    app = create_app(settings=settings, llm_client=llm_client)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/chat", json={"message": "Привет"})

    assert response.status_code == 503
    assert response.json()["status"] == "fallback"
    assert response.json()["answer"] == FALLBACK_MESSAGE
    await provider_client.aclose()


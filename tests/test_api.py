import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "embedding_model" in data


@pytest.mark.asyncio
async def test_list_models_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 1


@pytest.mark.asyncio
async def test_chat_completions_non_streaming():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        payload = {
            "model": "llama-3-8b-instruct",
            "messages": [
                {"role": "system", "content": "You are a concise tutor."},
                {"role": "user", "content": "What is Python?"}
            ],
            "temperature": 0.5,
            "stream": False
        }
        response = await client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "Python" in data["choices"][0]["message"]["content"] or "response" in data["choices"][0]["message"]["content"]

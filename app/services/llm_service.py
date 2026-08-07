import asyncio
import time
import json
import logging
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, List
import httpx
from app.core.config import settings
from app.schemas.openai import ChatCompletionRequest, ChatMessage

logger = logging.getLogger(__name__)

class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate(self, request: ChatCompletionRequest) -> str:
        """Non-streaming generation returning full response string."""
        pass

    @abstractmethod
    async def generate_stream(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        """Streaming generation yielding text tokens asynchronously."""
        pass


class MockLLMProvider(BaseLLMProvider):
    """
    Mock LLM provider simulating realistic LLM generation latency (~800ms)
    and token-by-token streaming response playback.
    """
    def __init__(self, latency_seconds: float = 0.8):
        self.latency_seconds = latency_seconds

    async def generate(self, request: ChatCompletionRequest) -> str:
        user_prompt = request.extract_user_prompt()
        await asyncio.sleep(self.latency_seconds)
        return f"This is an AI generated response for prompt: '{user_prompt}'. Served via high-throughput inference engine."

    async def generate_stream(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        user_prompt = request.extract_user_prompt()
        words = [
            "This", " is", " a", " simulated", " LLM", " response", " for",
            f" '{user_prompt[:25]}...'", " demonstrating", " high-throughput",
            " streaming", " with", " low", " latency."
        ]
        
        # Initial Time-To-First-Token (TTFT) simulation (~400ms)
        await asyncio.sleep(self.latency_seconds / 2)
        
        for word in words:
            await asyncio.sleep(0.04)  # 40ms token generation interval
            yield word


class OllamaProvider(BaseLLMProvider):
    def __init__(self, base_url: str = settings.OLLAMA_BASE_URL):
        self.base_url = base_url.rstrip("/")

    async def generate(self, request: ChatCompletionRequest) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "model": request.model,
                "messages": [msg.model_dump() for msg in request.messages],
                "stream": False,
                "options": {"temperature": request.temperature}
            }
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")

    async def generate_stream(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "model": request.model,
                "messages": [msg.model_dump() for msg in request.messages],
                "stream": True,
                "options": {"temperature": request.temperature}
            }
            async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line:
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content


class OpenAICompatibleProvider(BaseLLMProvider):
    def __init__(self, base_url: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def generate(self, request: ChatCompletionRequest) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = request.model_dump(exclude_none=True)
            payload["stream"] = False
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers()
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def generate_stream(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = request.model_dump(exclude_none=True)
            payload["stream"] = True
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers()
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except Exception:
                            continue


def get_llm_provider(provider_type: str = settings.DEFAULT_LLM_PROVIDER) -> BaseLLMProvider:
    if provider_type == "ollama":
        return OllamaProvider(base_url=settings.OLLAMA_BASE_URL)
    elif provider_type == "vllm":
        return OpenAICompatibleProvider(base_url=f"{settings.VLLM_BASE_URL}/v1")
    elif provider_type == "openai":
        return OpenAICompatibleProvider(base_url=settings.OPENAI_BASE_URL, api_key=settings.OPENAI_API_KEY)
    else:
        return MockLLMProvider()

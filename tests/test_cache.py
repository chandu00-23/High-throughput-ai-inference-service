import pytest
import asyncio
from app.services.cache_service import CacheService
from app.services.embedding_service import EmbeddingService

@pytest.mark.asyncio
async def test_composite_hash_generation():
    cache = CacheService()
    sys_hash1, comp_hash1 = cache.generate_composite_hash(
        model="llama-3-8b-instruct",
        system_prompt="You are a helpful assistant.",
        temperature=0.7,
        user_prompt="What is AI?"
    )
    
    sys_hash2, comp_hash2 = cache.generate_composite_hash(
        model="llama-3-8b-instruct",
        system_prompt="You are a helpful assistant.",
        temperature=0.7,
        user_prompt="What is AI?"
    )

    # Identical inputs must yield identical hashes
    assert sys_hash1 == sys_hash2
    assert comp_hash1 == comp_hash2

    # Varying temperature MUST produce a different composite hash (prevents temperature drift)
    _, comp_hash_different_temp = cache.generate_composite_hash(
        model="llama-3-8b-instruct",
        system_prompt="You are a helpful assistant.",
        temperature=0.0,
        user_prompt="What is AI?"
    )
    assert comp_hash1 != comp_hash_different_temp


@pytest.mark.asyncio
async def test_embedding_service_thread_offload():
    service = EmbeddingService(model_name="all-MiniLM-L6-v2")
    vector = await service.get_embedding("Test prompt for vector search")
    assert isinstance(vector, list)
    assert len(vector) == service.dimension

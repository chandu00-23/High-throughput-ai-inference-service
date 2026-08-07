import hashlib
import logging
import struct
import json
from typing import Optional, Tuple, Dict, Any, List
import redis.asyncio as redis
from redis.commands.search.field import VectorField, TagField, TextField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from app.core.config import settings

logger = logging.getLogger(__name__)

class CacheService:
    def __init__(self, redis_url: str = settings.REDIS_URL):
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None
        self.index_name = settings.REDIS_INDEX_NAME
        self.threshold = settings.SIMILARITY_THRESHOLD
        self.max_distance = 1.0 - self.threshold  # RediSearch Cosine Distance threshold (d <= 0.05)

    async def connect(self):
        """Connect to Redis and initialize RediSearch HNSW Index."""
        try:
            self.redis = redis.from_url(self.redis_url, decode_responses=False)
            await self.redis.ping()
            logger.info("Connected to Redis successfully.")
            await self._ensure_index_exists()
        except Exception as e:
            logger.error(f"Failed to connect to Redis at {self.redis_url}: {e}")
            self.redis = None

    async def close(self):
        if self.redis:
            await self.redis.aclose()
            logger.info("Redis connection closed.")

    def generate_composite_hash(self, model: str, system_prompt: str, temperature: float, user_prompt: str) -> Tuple[str, str]:
        """
        Generates:
        1. system_hash: SHA256 of system_prompt (for TAG filtering)
        2. composite_hash: SHA256 of model + system_prompt + temperature + user_prompt
        """
        sys_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16]
        raw_key = f"{model}:{sys_hash}:{temperature:.2f}:{user_prompt.strip()}"
        comp_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        return sys_hash, comp_hash

    async def _ensure_index_exists(self):
        """Creates RediSearch HNSW Vector Index if not already present."""
        if not self.redis:
            return
        
        try:
            # Check if index exists
            await self.redis.ft(self.index_name).info()
            logger.info(f"RediSearch index '{self.index_name}' already exists.")
        except Exception:
            logger.info(f"Creating RediSearch HNSW index '{self.index_name}'...")
            schema = (
                TagField("model"),
                TagField("system_hash"),
                TextField("prompt"),
                TextField("response"),
                VectorField(
                    "vector",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": settings.EMBEDDING_DIM,
                        "DISTANCE_METRIC": "COSINE",
                        "INITIAL_CAP": 1000,
                    }
                )
            )
            definition = IndexDefinition(prefix=["vec:"], index_type=IndexType.HASH)
            try:
                await self.redis.ft(self.index_name).create_index(schema, definition=definition)
                logger.info(f"RediSearch index '{self.index_name}' created successfully.")
            except Exception as create_err:
                logger.warning(f"Index creation notice: {create_err}")

    async def get_exact(self, composite_hash: str) -> Optional[str]:
        """Exact Hash Lookup (< 2ms response latency)."""
        if not self.redis:
            return None
        try:
            val = await self.redis.get(f"exact:{composite_hash}")
            if val:
                return val.decode("utf-8")
        except Exception as e:
            logger.error(f"Error in exact cache lookup: {e}")
        return None

    async def query_semantic(
        self, model: str, system_hash: str, vector: List[float]
    ) -> Optional[Tuple[str, float]]:
        """
        Semantic Vector Search using RediSearch HNSW with TAG Filtering.
        Returns: (cached_response, similarity_score) if distance <= max_distance (0.05)
        """
        if not self.redis:
            return None

        try:
            # Pack float vector into binary float32 bytes for Redis
            vec_bytes = struct.pack(f"{len(vector)}f", *vector)

            # Escape hyphens or special chars in tag strings for RediSearch query parser
            clean_model = model.replace("-", "\\-")
            clean_sys = system_hash.replace("-", "\\-")
            
            # RediSearch TAG Filtered KNN Query
            query_str = f"(@model:{{{clean_model}}} @system_hash:{{{clean_sys}}})=>[KNN 1 @vector $query_vec AS vector_distance]"
            
            q = Query(query_str)\
                .sort_by("vector_distance")\
                .return_fields("response", "vector_distance")\
                .dialect(2)

            res = await self.redis.ft(self.index_name).search(q, query_params={"query_vec": vec_bytes})

            if res and res.docs:
                doc = res.docs[0]
                distance = float(getattr(doc, "vector_distance", 1.0))
                similarity = 1.0 - distance

                logger.info(f"Vector search match found: distance={distance:.4f}, similarity={similarity:.4f}")

                if distance <= self.max_distance:
                    response_text = getattr(doc, "response", None)
                    if isinstance(response_text, bytes):
                        response_text = response_text.decode("utf-8")
                    return response_text, similarity
        except Exception as e:
            logger.error(f"Error in vector search cache query: {e}")
        return None

    async def set_async(
        self,
        composite_hash: str,
        system_hash: str,
        model: str,
        user_prompt: str,
        response_text: str,
        vector: List[float],
        ttl: int = settings.CACHE_TTL_SECONDS
    ):
        """Asynchronous non-blocking Redis cache population."""
        if not self.redis:
            return

        # 1. Memory Safety: Guard against bounded Redis payload explosion
        payload_bytes = len(response_text.encode("utf-8"))
        if payload_bytes > settings.MAX_PAYLOAD_BYTES:
            logger.warning(f"Payload size {payload_bytes} bytes exceeds limit of {settings.MAX_PAYLOAD_BYTES}. Skipping cache insert.")
            return

        try:
            vec_bytes = struct.pack(f"{len(vector)}f", *vector)
            
            pipe = self.redis.pipeline()
            
            # Save Exact Key
            pipe.set(f"exact:{composite_hash}", response_text, ex=ttl)
            
            # Save Vector HNSW Key
            vec_key = f"vec:{composite_hash}"
            mapping = {
                "model": model,
                "system_hash": system_hash,
                "prompt": user_prompt,
                "response": response_text,
                "vector": vec_bytes
            }
            pipe.hset(vec_key, mapping=mapping)
            pipe.expire(vec_key, ttl)
            
            await pipe.execute()
            logger.info(f"Cached composite key '{composite_hash[:10]}...' in Redis (TTL={ttl}s).")
        except Exception as e:
            logger.error(f"Failed to populate Redis cache: {e}")

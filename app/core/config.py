from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "High-Throughput AI Inference Microservice"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/v1"
    
    # Redis configuration
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_INDEX_NAME: str = "idx:semantic_cache"
    
    # Cache thresholds and safety guards
    SIMILARITY_THRESHOLD: float = 0.95  # Cosine similarity target (distance <= 0.05)
    MAX_PAYLOAD_BYTES: int = 102_400    # 100 KB max response length for Redis caching
    CACHE_TTL_SECONDS: int = 86_400     # 24 hour TTL
    
    # Embedding Configuration
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    
    # LLM Provider Configuration
    DEFAULT_LLM_PROVIDER: Literal["mock", "ollama", "vllm", "openai"] = "mock"
    DEFAULT_MODEL_NAME: str = "llama-3-8b-instruct"
    
    # External LLM endpoints
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    VLLM_BASE_URL: str = "http://localhost:8000"
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

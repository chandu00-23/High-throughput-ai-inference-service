import asyncio
import logging
import numpy as np
from typing import List
from app.core.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self._dim = settings.EMBEDDING_DIM
        self._initialize_model()

    def _initialize_model(self):
        """Lazy loader for SentenceTransformer model with CPU optimization."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model '{self.model_name}' on CPU...")
            self.model = SentenceTransformer(self.model_name, device="cpu")
            dim_func = getattr(self.model, "get_embedding_dimension", getattr(self.model, "get_sentence_embedding_dimension", None))
            if dim_func:
                self._dim = dim_func()
            logger.info(f"Embedding model loaded successfully. Vector dimension: {self._dim}")
        except Exception as e:
            logger.warning(f"SentenceTransformer load warning ({e}). Falling back to fast numpy embedding provider.")
            self.model = None

    def _sync_encode(self, text: str) -> List[float]:
        """Synchronous CPU computation of normalized embedding vector."""
        if self.model is not None:
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        
        # Fallback fast deterministic hashing embedding for zero-dependency test/offline mode
        # Generates a normalized pseudo-random vector based on text hash
        np.random.seed(abs(hash(text)) % (2**32))
        raw = np.random.randn(self._dim)
        norm = np.linalg.norm(raw)
        normalized = (raw / norm).tolist()
        return normalized

    async def get_embedding(self, text: str) -> List[float]:
        """Offload CPU-bound vector encoding to thread pool to prevent event loop blocking."""
        return await asyncio.to_thread(self._sync_encode, text)

    @property
    def dimension(self) -> int:
        return self._dim

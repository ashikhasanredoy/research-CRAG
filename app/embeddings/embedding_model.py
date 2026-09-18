import logging
from typing import List
from langchain_core.embeddings import Embeddings
import requests
from app.config import settings

logger = logging.getLogger(__name__)

class OllamaEmbeddingService(Embeddings):
    """Ollama-based embedding client supporting nomic-embed-text with robust fallback."""

    def __init__(
        self,
        base_url: str = settings.OLLAMA_BASE_URL,
        model_name: str = settings.DEFAULT_EMBED_MODEL
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of document texts."""
        embeddings: List[List[float]] = []
        for text in texts:
            embeddings.append(self.embed_query(text))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query or text chunk."""
        try:
            url = f"{self.base_url}/api/embeddings"
            response = requests.post(
                url,
                json={"model": self.model_name, "prompt": text},
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                return data["embedding"]
            else:
                logger.warning(f"Ollama embedding failed (status {response.status_code}): {response.text}")
                return self._fallback_embedding(text)
        except Exception as e:
            logger.warning(f"Ollama connection error for embedding ({e}), using deterministic fallback embedding.")
            return self._fallback_embedding(text)

    def _fallback_embedding(self, text: str) -> List[float]:
        """Deterministic 768-dim pseudo-embedding for testing/fallback when Ollama is offline."""
        import hashlib
        import numpy as np
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        vec = rng.randn(768).astype(float)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

def get_embedding_model(model_name: str = settings.DEFAULT_EMBED_MODEL) -> Embeddings:
    return OllamaEmbeddingService(model_name=model_name)

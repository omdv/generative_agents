"""
Embedding service for semantic search.

This module provides text embeddings for memory retrieval.
For simplicity, it uses a basic approach but can be extended
to use proper embedding models.
"""

import asyncio
import hashlib
import logging
from typing import Any

import httpx
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings.

    Currently uses OpenRouter's embedding endpoints or falls back
    to a simple TF-IDF-like approach.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "openai/text-embedding-3-small",
        use_simple: bool = True,
    ):
        """
        Initialize the embedding service.

        Args:
            api_key: OpenRouter API key (optional if using simple mode).
            base_url: API base URL.
            model: Embedding model to use.
            use_simple: Use simple hash-based embeddings instead of API.
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.use_simple = use_simple

        # Cache for embeddings
        self._cache: dict[str, list[float]] = {}

        # Semaphore for rate limiting API calls
        self._semaphore = asyncio.Semaphore(5)

    async def embed(self, text: str) -> list[float]:
        """
        Generate an embedding for the given text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        # Check cache first
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self.use_simple or not self.api_key:
            embedding = self._simple_embed(text)
        else:
            embedding = await self._api_embed(text)

        self._cache[cache_key] = embedding
        return embedding

    def _simple_embed(self, text: str, dim: int = 256) -> list[float]:
        """
        Generate a simple hash-based embedding.

        This is a lightweight approach that doesn't require an API call.
        It's not as semantically meaningful as proper embeddings but
        works for basic similarity calculations.

        Args:
            text: Text to embed.
            dim: Dimension of the embedding.

        Returns:
            Embedding vector.
        """
        # Normalize text
        text = text.lower().strip()

        # Create a pseudo-random embedding based on text hash
        # This ensures the same text always gets the same embedding
        hash_bytes = hashlib.sha256(text.encode()).digest()

        # Use the hash to seed a random number generator
        rng = np.random.default_rng(
            int.from_bytes(hash_bytes[:8], byteorder="big")
        )

        # Generate embedding
        embedding = rng.standard_normal(dim)

        # Add some word-level signal
        words = text.split()
        for i, word in enumerate(words):
            word_hash = hashlib.md5(word.encode()).digest()
            word_idx = int.from_bytes(word_hash[:4], byteorder="big") % dim
            embedding[word_idx] += 0.5 * (1.0 / (i + 1))  # Weight by position

        # Normalize to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()

    async def _api_embed(self, text: str) -> list[float]:
        """
        Generate an embedding using the API.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        async with self._semaphore:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            ) as client:
                try:
                    response = await client.post(
                        "/embeddings",
                        json={
                            "model": self.model,
                            "input": text,
                        },
                    )
                    response.raise_for_status()
                    data = response.json()

                    if "data" in data and len(data["data"]) > 0:
                        return data["data"][0]["embedding"]

                except Exception as e:
                    logger.warning(f"API embedding failed, using simple: {e}")
                    return self._simple_embed(text)

        return self._simple_embed(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        # Use asyncio.gather for concurrent embedding
        embeddings = await asyncio.gather(*[self.embed(text) for text in texts])
        return list(embeddings)

    def similarity(self, a: list[float], b: list[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            a: First embedding.
            b: Second embedding.

        Returns:
            Cosine similarity (0-1).
        """
        a_arr = np.array(a)
        b_arr = np.array(b)

        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


# Global singleton instance
_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """
    Get the global embedding service instance.
    """
    global _embedding_service

    if _embedding_service is None:
        try:
            from django.conf import settings
            _embedding_service = EmbeddingService(
                api_key=settings.OPENROUTER_API_KEY,
                base_url=settings.OPENROUTER_BASE_URL,
                use_simple=True,  # Default to simple for efficiency
            )
        except Exception:
            # Fall back to simple embeddings without Django
            _embedding_service = EmbeddingService(use_simple=True)

    return _embedding_service

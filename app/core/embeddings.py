"""
Embedding service with Matryoshka dimension support.

Matryoshka embeddings allow truncating full-dimension vectors to smaller
dimensions with minimal quality loss. OpenAI's text-embedding-3-small
supports: 256, 512, 1024, 1536 dimensions.
"""

import hashlib
import logging
from typing import List, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)

# Lazy import to avoid startup errors if openai not installed
_openai_client: Optional["AsyncOpenAI"] = None


def _get_openai_client():
    """Get or create the async OpenAI client."""
    global _openai_client
    if _openai_client is None:
        try:
            from openai import AsyncOpenAI

            settings = get_settings()
            _openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        except ImportError:
            raise ImportError(
                "OpenAI package is required for embeddings. "
                "Install with: pip install openai"
            )
    return _openai_client


class EmbeddingService:
    """
    Generate embeddings with configurable dimensions using Matryoshka.

    Matryoshka embeddings preserve quality when truncated to smaller
    dimensions, enabling cost/performance trade-offs.

    Usage:
        service = EmbeddingService(dimensions=512)
        embedding = await service.get_embedding("Hello world")
    """

    def __init__(self, dimensions: Optional[int] = None):
        """
        Initialize embedding service.

        Args:
            dimensions: Target embedding dimension (256, 512, 1024, 1536).
                       Defaults to settings.embedding_dimensions.
        """
        settings = get_settings()
        self.dimensions = dimensions or settings.embedding_dimensions
        self.model = settings.openai_embedding_model
        self._client = None

    @property
    def client(self):
        """Lazy-load the OpenAI client."""
        if self._client is None:
            self._client = _get_openai_client()
        return self._client

    async def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding with Matryoshka dimension reduction.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector of configured dimension
        """
        settings = get_settings()

        if settings.use_matryoshka and self.dimensions < 1536:
            # Use Matryoshka truncation via OpenAI API
            response = await self.client.embeddings.create(
                input=text,
                model=self.model,
                dimensions=self.dimensions,
            )
        else:
            # Full dimension embedding
            response = await self.client.embeddings.create(
                input=text,
                model=self.model,
            )

        return response.data[0].embedding

    async def get_embeddings_batch(
        self, texts: List[str], batch_size: int = 100
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed
            batch_size: Max texts per API call (OpenAI limit is 2048)

        Returns:
            List of embedding vectors
        """
        settings = get_settings()
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            if settings.use_matryoshka and self.dimensions < 1536:
                response = await self.client.embeddings.create(
                    input=batch,
                    model=self.model,
                    dimensions=self.dimensions,
                )
            else:
                response = await self.client.embeddings.create(
                    input=batch,
                    model=self.model,
                )

            # Preserve order from response
            batch_embeddings = sorted(response.data, key=lambda x: x.index)
            embeddings.extend([item.embedding for item in batch_embeddings])

        return embeddings

    @staticmethod
    def truncate_embedding(
        embedding: List[float], target_dim: int
    ) -> List[float]:
        """
        Truncate existing embedding to smaller dimension.

        Useful for migrating existing 1536d embeddings to 512d.
        Matryoshka embeddings preserve quality when truncated.

        Args:
            embedding: Full-dimension embedding
            target_dim: Target dimension (must be <= current)

        Returns:
            Truncated embedding
        """
        if len(embedding) < target_dim:
            raise ValueError(
                f"Cannot truncate {len(embedding)}d to {target_dim}d"
            )
        return embedding[:target_dim]

    @staticmethod
    def compute_text_hash(text: str) -> str:
        """
        Compute SHA256 hash for content deduplication.

        Normalizes text before hashing:
        - Strips whitespace
        - Converts to lowercase

        Args:
            text: Input text

        Returns:
            64-character hex hash string
        """
        normalized = text.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()


# Convenience function for one-off embeddings
async def get_embedding(text: str, dimensions: Optional[int] = None) -> List[float]:
    """
    Get embedding for text using default settings.

    Args:
        text: Input text
        dimensions: Optional dimension override

    Returns:
        Embedding vector
    """
    service = EmbeddingService(dimensions=dimensions)
    return await service.get_embedding(text)

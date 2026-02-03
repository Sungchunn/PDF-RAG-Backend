"""
Unit tests for embedding service module.
Tests Matryoshka embeddings, batch processing, and text hashing.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.embeddings import (
    EmbeddingService,
    get_embedding,
)


class TestEmbeddingService:
    """Tests for EmbeddingService class."""

    @pytest.fixture
    def mock_openai_response(self):
        """Create mock OpenAI embedding response."""
        mock_response = MagicMock()
        mock_data = MagicMock()
        mock_data.embedding = [0.1] * 512
        mock_data.index = 0
        mock_response.data = [mock_data]
        return mock_response

    @pytest.fixture
    def mock_openai_batch_response(self):
        """Create mock OpenAI batch embedding response."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 512, index=0),
            MagicMock(embedding=[0.2] * 512, index=1),
            MagicMock(embedding=[0.3] * 512, index=2),
        ]
        return mock_response

    # ============ Single Embedding Tests ============

    @patch("app.core.embeddings._get_openai_client")
    @patch("app.core.embeddings.get_settings")
    async def test_get_embedding_matryoshka(
        self, mock_get_settings, mock_get_client, mock_openai_response
    ):
        """Test embedding generation with Matryoshka dimensions."""
        mock_settings = MagicMock()
        mock_settings.embedding_dimensions = 512
        mock_settings.openai_embedding_model = "text-embedding-3-small"
        mock_settings.use_matryoshka = True
        mock_get_settings.return_value = mock_settings

        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_openai_response)
        mock_get_client.return_value = mock_client

        service = EmbeddingService(dimensions=512)
        result = await service.get_embedding("Hello world")

        assert len(result) == 512
        mock_client.embeddings.create.assert_called_once()
        call_kwargs = mock_client.embeddings.create.call_args[1]
        assert call_kwargs["dimensions"] == 512

    @patch("app.core.embeddings._get_openai_client")
    @patch("app.core.embeddings.get_settings")
    async def test_get_embedding_full_dimensions(
        self, mock_get_settings, mock_get_client
    ):
        """Test embedding generation with full dimensions (no Matryoshka)."""
        mock_settings = MagicMock()
        mock_settings.embedding_dimensions = 1536
        mock_settings.openai_embedding_model = "text-embedding-3-small"
        mock_settings.use_matryoshka = False
        mock_get_settings.return_value = mock_settings

        mock_response = MagicMock()
        mock_data = MagicMock()
        mock_data.embedding = [0.1] * 1536
        mock_response.data = [mock_data]

        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        service = EmbeddingService(dimensions=1536)
        result = await service.get_embedding("Hello world")

        assert len(result) == 1536
        # Should not pass dimensions param when use_matryoshka is False
        call_kwargs = mock_client.embeddings.create.call_args[1]
        assert "dimensions" not in call_kwargs

    # ============ Batch Embedding Tests ============

    @patch("app.core.embeddings._get_openai_client")
    @patch("app.core.embeddings.get_settings")
    async def test_get_embeddings_batch(
        self, mock_get_settings, mock_get_client, mock_openai_batch_response
    ):
        """Test batch embedding generation."""
        mock_settings = MagicMock()
        mock_settings.embedding_dimensions = 512
        mock_settings.openai_embedding_model = "text-embedding-3-small"
        mock_settings.use_matryoshka = True
        mock_get_settings.return_value = mock_settings

        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_openai_batch_response)
        mock_get_client.return_value = mock_client

        service = EmbeddingService(dimensions=512)
        texts = ["Text 1", "Text 2", "Text 3"]
        result = await service.get_embeddings_batch(texts)

        assert len(result) == 3
        assert all(len(emb) == 512 for emb in result)

    @patch("app.core.embeddings._get_openai_client")
    @patch("app.core.embeddings.get_settings")
    async def test_get_embeddings_batch_preserves_order(
        self, mock_get_settings, mock_get_client
    ):
        """Test that batch embeddings preserve input order."""
        mock_settings = MagicMock()
        mock_settings.embedding_dimensions = 512
        mock_settings.openai_embedding_model = "text-embedding-3-small"
        mock_settings.use_matryoshka = True
        mock_get_settings.return_value = mock_settings

        # Response with shuffled indices
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.3] * 512, index=2),
            MagicMock(embedding=[0.1] * 512, index=0),
            MagicMock(embedding=[0.2] * 512, index=1),
        ]

        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        service = EmbeddingService(dimensions=512)
        result = await service.get_embeddings_batch(["A", "B", "C"])

        # Should be sorted by original index
        assert result[0][0] == pytest.approx(0.1)
        assert result[1][0] == pytest.approx(0.2)
        assert result[2][0] == pytest.approx(0.3)

    @patch("app.core.embeddings._get_openai_client")
    @patch("app.core.embeddings.get_settings")
    async def test_get_embeddings_batch_chunking(
        self, mock_get_settings, mock_get_client
    ):
        """Test that large batches are split into chunks."""
        mock_settings = MagicMock()
        mock_settings.embedding_dimensions = 512
        mock_settings.openai_embedding_model = "text-embedding-3-small"
        mock_settings.use_matryoshka = True
        mock_get_settings.return_value = mock_settings

        # Create response for each batch call
        def create_response(count):
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(embedding=[0.1] * 512, index=i)
                for i in range(count)
            ]
            return mock_response

        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(
            side_effect=[create_response(2), create_response(1)]
        )
        mock_get_client.return_value = mock_client

        service = EmbeddingService(dimensions=512)
        texts = ["Text 1", "Text 2", "Text 3"]
        result = await service.get_embeddings_batch(texts, batch_size=2)

        # Should make 2 API calls
        assert mock_client.embeddings.create.call_count == 2
        assert len(result) == 3

    # ============ Truncation Tests ============

    def test_truncate_embedding_valid(self):
        """Test embedding truncation to smaller dimension."""
        full_embedding = list(range(1536))
        truncated = EmbeddingService.truncate_embedding(full_embedding, 512)

        assert len(truncated) == 512
        assert truncated == list(range(512))

    def test_truncate_embedding_same_size(self):
        """Test truncation when target equals current size."""
        embedding = [0.1] * 512
        truncated = EmbeddingService.truncate_embedding(embedding, 512)

        assert len(truncated) == 512
        assert truncated == embedding

    def test_truncate_embedding_invalid_target(self):
        """Test truncation fails when target exceeds current size."""
        embedding = [0.1] * 512

        with pytest.raises(ValueError, match="Cannot truncate"):
            EmbeddingService.truncate_embedding(embedding, 1536)

    # ============ Text Hash Tests ============

    def test_compute_text_hash_consistent(self):
        """Test that same text produces same hash."""
        text = "Hello world"
        hash1 = EmbeddingService.compute_text_hash(text)
        hash2 = EmbeddingService.compute_text_hash(text)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

    def test_compute_text_hash_normalized(self):
        """Test that hashing normalizes whitespace and case."""
        text1 = "  Hello World  "
        text2 = "hello world"

        hash1 = EmbeddingService.compute_text_hash(text1)
        hash2 = EmbeddingService.compute_text_hash(text2)

        assert hash1 == hash2

    def test_compute_text_hash_different_texts(self):
        """Test that different texts produce different hashes."""
        hash1 = EmbeddingService.compute_text_hash("Hello")
        hash2 = EmbeddingService.compute_text_hash("World")

        assert hash1 != hash2


class TestGetEmbeddingFunction:
    """Tests for the convenience get_embedding function."""

    @patch("app.core.embeddings._get_openai_client")
    @patch("app.core.embeddings.get_settings")
    async def test_get_embedding_convenience(self, mock_get_settings, mock_get_client):
        """Test the convenience function for getting embeddings."""
        mock_settings = MagicMock()
        mock_settings.embedding_dimensions = 512
        mock_settings.openai_embedding_model = "text-embedding-3-small"
        mock_settings.use_matryoshka = True
        mock_get_settings.return_value = mock_settings

        mock_response = MagicMock()
        mock_data = MagicMock()
        mock_data.embedding = [0.5] * 512
        mock_response.data = [mock_data]

        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        result = await get_embedding("Test text")

        assert len(result) == 512
        assert result[0] == pytest.approx(0.5)

    @patch("app.core.embeddings._get_openai_client")
    @patch("app.core.embeddings.get_settings")
    async def test_get_embedding_with_dimension_override(
        self, mock_get_settings, mock_get_client
    ):
        """Test convenience function with dimension override."""
        mock_settings = MagicMock()
        mock_settings.embedding_dimensions = 512
        mock_settings.openai_embedding_model = "text-embedding-3-small"
        mock_settings.use_matryoshka = True
        mock_get_settings.return_value = mock_settings

        mock_response = MagicMock()
        mock_data = MagicMock()
        mock_data.embedding = [0.5] * 256
        mock_response.data = [mock_data]

        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        result = await get_embedding("Test text", dimensions=256)

        assert len(result) == 256

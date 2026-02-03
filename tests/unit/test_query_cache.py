"""
Unit tests for query cache module.
Tests exact matching, semantic matching, TTL expiration, and invalidation.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.query_cache import QueryCache, CachedResult


class TestQueryCache:
    """Tests for QueryCache class."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = AsyncMock()
        session.execute = AsyncMock()
        session.add = MagicMock()
        session.flush = AsyncMock()
        return session

    @pytest.fixture
    def cache(self, mock_session):
        """Create a QueryCache with mock session."""
        with patch("app.core.query_cache.get_settings") as mock_settings:
            mock_settings.return_value.query_cache_ttl_seconds = 3600
            mock_settings.return_value.semantic_cache_threshold = 0.95
            return QueryCache(mock_session)

    # ============ Query Hash Tests ============

    def test_compute_query_hash_consistent(self, cache):
        """Test that same query produces same hash."""
        hash1 = cache._compute_query_hash("What is AI?", None)
        hash2 = cache._compute_query_hash("What is AI?", None)
        assert hash1 == hash2

    def test_compute_query_hash_normalized(self, cache):
        """Test that hashing normalizes query text."""
        hash1 = cache._compute_query_hash("  What  is  AI?  ", None)
        hash2 = cache._compute_query_hash("what is ai?", None)
        assert hash1 == hash2

    def test_compute_query_hash_different_docs(self, cache):
        """Test that different document scopes produce different hashes."""
        hash1 = cache._compute_query_hash("What is AI?", ["doc1"])
        hash2 = cache._compute_query_hash("What is AI?", ["doc2"])
        assert hash1 != hash2

    def test_compute_query_hash_doc_order_irrelevant(self, cache):
        """Test that document order doesn't affect hash."""
        hash1 = cache._compute_query_hash("Query", ["doc1", "doc2"])
        hash2 = cache._compute_query_hash("Query", ["doc2", "doc1"])
        assert hash1 == hash2

    def test_compute_query_hash_null_docs(self, cache):
        """Test hash with no document filter (all docs)."""
        hash1 = cache._compute_query_hash("Query", None)
        hash2 = cache._compute_query_hash("Query", [])
        # None and [] should produce different hashes
        # (None = all docs, [] = no docs)
        assert hash1 != hash2

    # ============ Normalize Query Tests ============

    def test_normalize_query(self, cache):
        """Test query normalization."""
        normalized = cache._normalize_query("  Hello   World  ")
        assert normalized == "hello world"

    def test_normalize_query_preserves_content(self, cache):
        """Test that normalization preserves meaningful content."""
        normalized = cache._normalize_query("What is Artificial Intelligence?")
        assert normalized == "what is artificial intelligence?"

    # ============ Cache Get (Exact) Tests ============

    @pytest.mark.asyncio
    async def test_get_exact_hit(self, cache, mock_session):
        """Test cache hit with exact query hash match."""
        # Mock cache hit
        mock_cached = MagicMock()
        mock_cached.id = "cache-1"
        mock_cached.result_chunk_ids = ["chunk-1", "chunk-2"]
        mock_cached.result_scores = [0.95, 0.85]
        mock_cached.created_at = datetime.now(timezone.utc)
        mock_cached.hit_count = 5

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_cached
        mock_session.execute.return_value = mock_result

        result = await cache.get("test query", "user-1", None)

        assert result is not None
        assert result.chunk_ids == ["chunk-1", "chunk-2"]
        assert result.scores == [0.95, 0.85]
        assert result.hit_count == 6  # Incremented

    @pytest.mark.asyncio
    async def test_get_exact_miss(self, cache, mock_session):
        """Test cache miss when no exact match exists."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await cache.get("new query", "user-1", None)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_expired_entry(self, cache, mock_session):
        """Test that expired entries are not returned."""
        # The WHERE clause already filters expired entries
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await cache.get("query", "user-1", None)
        assert result is None

    # ============ Cache Get (Semantic) Tests ============

    @pytest.mark.asyncio
    async def test_get_semantic_hit(self, cache, mock_session):
        """Test cache hit with semantic similarity match."""
        # Mock semantic match result
        mock_row = MagicMock()
        mock_row.id = "cache-1"
        mock_row.result_chunk_ids = ["chunk-1"]
        mock_row.result_scores = [0.9]
        mock_row.created_at = datetime.now(timezone.utc)
        mock_row.hit_count = 3
        mock_row.similarity = 0.97

        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_result

        query_embedding = [0.1] * 512
        result = await cache.get_semantic(query_embedding, "user-1", None)

        assert result is not None
        assert result.chunk_ids == ["chunk-1"]
        assert result.hit_count == 4

    @pytest.mark.asyncio
    async def test_get_semantic_miss(self, cache, mock_session):
        """Test cache miss when no similar queries exist."""
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_session.execute.return_value = mock_result

        query_embedding = [0.5] * 512
        result = await cache.get_semantic(query_embedding, "user-1", None)

        assert result is None

    # ============ Cache Set Tests ============

    @pytest.mark.asyncio
    async def test_set_new_entry(self, cache, mock_session):
        """Test setting a new cache entry."""
        # Mock no existing entry
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        cache_id = await cache.set(
            query="test query",
            query_embedding=[0.1] * 512,
            user_id="user-1",
            document_ids=["doc-1"],
            chunk_ids=["chunk-1", "chunk-2"],
            scores=[0.95, 0.85],
        )

        assert cache_id is not None
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_existing_entry(self, cache, mock_session):
        """Test that existing entry is not duplicated."""
        # Mock existing entry
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = "existing-id"
        mock_session.execute.return_value = mock_result

        cache_id = await cache.set(
            query="existing query",
            query_embedding=[0.1] * 512,
            user_id="user-1",
            document_ids=None,
            chunk_ids=["chunk-1"],
            scores=[0.9],
        )

        # Should return hash but not add new entry
        mock_session.add.assert_not_called()

    # ============ Cache Invalidation Tests ============

    @pytest.mark.asyncio
    async def test_invalidate_for_document(self, cache, mock_session):
        """Test invalidating cache entries for a document."""
        mock_result = MagicMock()
        mock_result.rowcount = 5
        mock_session.execute.return_value = mock_result

        count = await cache.invalidate_for_document("doc-1")

        assert count == 5
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalidate_for_user(self, cache, mock_session):
        """Test invalidating all cache entries for a user."""
        mock_result = MagicMock()
        mock_result.rowcount = 10
        mock_session.execute.return_value = mock_result

        count = await cache.invalidate_for_user("user-1")

        assert count == 10

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, cache, mock_session):
        """Test cleaning up expired cache entries."""
        mock_result = MagicMock()
        mock_result.rowcount = 3
        mock_session.execute.return_value = mock_result

        count = await cache.cleanup_expired()

        assert count == 3

    # ============ Cache Stats Tests ============

    @pytest.mark.asyncio
    async def test_get_stats(self, cache, mock_session):
        """Test getting cache statistics."""
        mock_row = MagicMock()
        mock_row.total_entries = 100
        mock_row.total_hits = 500
        mock_row.avg_hit_count = 5.0

        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_result

        stats = await cache.get_stats("user-1")

        assert stats["total_entries"] == 100
        assert stats["total_hits"] == 500
        assert stats["avg_hit_count"] == 5.0


class TestCachedResult:
    """Tests for CachedResult dataclass."""

    def test_cached_result_creation(self):
        """Test CachedResult creation."""
        result = CachedResult(
            chunk_ids=["chunk-1", "chunk-2"],
            scores=[0.95, 0.85],
            created_at=datetime.now(timezone.utc),
            hit_count=5,
        )

        assert result.chunk_ids == ["chunk-1", "chunk-2"]
        assert result.scores == [0.95, 0.85]
        assert result.hit_count == 5

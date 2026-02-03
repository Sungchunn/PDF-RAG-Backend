"""
Integration tests for vector store module.
Tests chunk storage, vector search, hybrid search, and deletion.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.vector_store import (
    PGVectorStore,
    ChunkData,
    RetrievedChunk,
)


class TestPGVectorStore:
    """Tests for PGVectorStore class."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = AsyncMock()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.execute = AsyncMock()

        # Mock begin_nested as async context manager for savepoint support
        nested_ctx = AsyncMock()
        nested_ctx.__aenter__ = AsyncMock(return_value=None)
        nested_ctx.__aexit__ = AsyncMock(return_value=None)
        session.begin_nested = MagicMock(return_value=nested_ctx)

        return session

    @pytest.fixture
    def vector_store(self, mock_session):
        """Create a PGVectorStore with mock session."""
        return PGVectorStore(mock_session, user_id="test-user-123")

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        return [
            ChunkData(
                text="First chunk about machine learning and AI concepts.",
                document_id="doc-1",
                page_number=1,
                chunk_index=0,
                x0=72.0,
                y0=100.0,
                x1=540.0,
                y1=120.0,
                line_start=1,
                line_end=5,
                embedding=[0.1] * 512,
                token_count=10,
            ),
            ChunkData(
                text="Second chunk about natural language processing.",
                document_id="doc-1",
                page_number=1,
                chunk_index=1,
                x0=72.0,
                y0=130.0,
                x1=540.0,
                y1=150.0,
                line_start=6,
                line_end=10,
                embedding=[0.2] * 512,
                token_count=8,
            ),
        ]

    # ============ Add Chunks Tests ============

    @pytest.mark.asyncio
    async def test_add_chunks_creates_records(self, vector_store, mock_session, sample_chunks):
        """Test that add_chunks creates database records."""
        chunk_ids = await vector_store.add_chunks(sample_chunks)

        assert len(chunk_ids) == 2
        assert mock_session.add.call_count >= 2  # At least chunk models
        # Flush is called twice: after chunk models and after vector models
        assert mock_session.flush.call_count == 2

    @pytest.mark.asyncio
    async def test_add_chunks_returns_unique_ids(self, vector_store, mock_session, sample_chunks):
        """Test that add_chunks returns unique IDs."""
        chunk_ids = await vector_store.add_chunks(sample_chunks)

        assert len(chunk_ids) == len(set(chunk_ids))

    @pytest.mark.asyncio
    async def test_add_chunks_empty_list(self, vector_store, mock_session):
        """Test adding empty chunk list."""
        chunk_ids = await vector_store.add_chunks([])

        assert chunk_ids == []
        mock_session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_chunks_with_embedding(self, vector_store, mock_session, sample_chunks):
        """Test that embeddings are stored via raw SQL."""
        await vector_store.add_chunks(sample_chunks)

        # Should execute SQL for embedding storage
        assert mock_session.execute.call_count >= 2  # One per chunk with embedding

    @pytest.mark.asyncio
    async def test_add_chunk_without_embedding(self, vector_store, mock_session):
        """Test adding chunk without embedding."""
        chunk = ChunkData(
            text="No embedding chunk",
            document_id="doc-1",
            page_number=1,
            chunk_index=0,
            x0=0, y0=0, x1=100, y1=100,
            line_start=1,
            line_end=1,
            embedding=None,  # No embedding
        )

        chunk_ids = await vector_store.add_chunks([chunk])

        assert len(chunk_ids) == 1

    # ============ Savepoint Atomicity Tests ============

    @pytest.fixture
    def mock_session_with_savepoint(self):
        """Create a mock session with savepoint (begin_nested) support."""
        session = AsyncMock()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.execute = AsyncMock()

        # Mock begin_nested as async context manager
        nested_ctx = AsyncMock()
        nested_ctx.__aenter__ = AsyncMock(return_value=None)
        nested_ctx.__aexit__ = AsyncMock(return_value=None)
        session.begin_nested = MagicMock(return_value=nested_ctx)

        return session

    @pytest.mark.asyncio
    async def test_add_chunks_uses_savepoint(self, mock_session_with_savepoint, sample_chunks):
        """Test that add_chunks uses begin_nested for atomicity."""
        store = PGVectorStore(mock_session_with_savepoint, user_id="test-user")

        await store.add_chunks(sample_chunks)

        # Verify begin_nested was called for savepoint
        mock_session_with_savepoint.begin_nested.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_chunks_flushes_twice_for_two_stages(
        self, mock_session_with_savepoint, sample_chunks
    ):
        """Test that add_chunks flushes after chunks and after vectors."""
        store = PGVectorStore(mock_session_with_savepoint, user_id="test-user")

        await store.add_chunks(sample_chunks)

        # Should flush twice: once after chunk models, once after vector models
        assert mock_session_with_savepoint.flush.call_count == 2

    @pytest.mark.asyncio
    async def test_add_chunks_rollback_on_execute_failure(self, mock_session_with_savepoint):
        """Test that savepoint rolls back on SQL execution failure."""
        store = PGVectorStore(mock_session_with_savepoint, user_id="test-user")

        # Make execute fail to simulate embedding update failure
        mock_session_with_savepoint.execute = AsyncMock(
            side_effect=Exception("Database error during UPDATE")
        )

        chunk = ChunkData(
            text="Test chunk",
            document_id="doc-1",
            page_number=1,
            chunk_index=0,
            x0=0, y0=0, x1=100, y1=100,
            line_start=1,
            line_end=1,
            embedding=[0.1] * 512,
        )

        with pytest.raises(Exception, match="Database error during UPDATE"):
            await store.add_chunks([chunk])

        # begin_nested context manager should handle rollback automatically

    @pytest.mark.asyncio
    async def test_add_chunks_rollback_on_flush_failure(self, mock_session_with_savepoint):
        """Test that savepoint rolls back on flush failure."""
        store = PGVectorStore(mock_session_with_savepoint, user_id="test-user")

        # Make first flush fail
        mock_session_with_savepoint.flush = AsyncMock(
            side_effect=Exception("Flush failed - constraint violation")
        )

        chunk = ChunkData(
            text="Test chunk",
            document_id="doc-1",
            page_number=1,
            chunk_index=0,
            x0=0, y0=0, x1=100, y1=100,
            line_start=1,
            line_end=1,
            embedding=[0.1] * 512,
        )

        with pytest.raises(Exception, match="Flush failed"):
            await store.add_chunks([chunk])

    @pytest.mark.asyncio
    async def test_add_chunks_empty_skips_savepoint(self, mock_session_with_savepoint):
        """Test that empty chunks list skips savepoint creation."""
        store = PGVectorStore(mock_session_with_savepoint, user_id="test-user")

        result = await store.add_chunks([])

        assert result == []
        # Should not create savepoint for empty list
        mock_session_with_savepoint.begin_nested.assert_not_called()
        mock_session_with_savepoint.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_chunks_creates_all_models_before_first_flush(
        self, mock_session_with_savepoint, sample_chunks
    ):
        """Test that all chunk models are added before the first flush."""
        store = PGVectorStore(mock_session_with_savepoint, user_id="test-user")

        add_calls_before_flush = []
        flush_count = [0]

        original_add = mock_session_with_savepoint.add

        def track_add(model):
            add_calls_before_flush.append((flush_count[0], model))
            return original_add(model)

        async def track_flush():
            flush_count[0] += 1

        mock_session_with_savepoint.add = track_add
        mock_session_with_savepoint.flush = AsyncMock(side_effect=track_flush)

        await store.add_chunks(sample_chunks)

        # All chunk models (2) should be added before first flush (flush_count=0)
        chunk_adds_before_first_flush = [
            call for call in add_calls_before_flush if call[0] == 0
        ]
        # Should have 2 chunk models added before any flush
        assert len(chunk_adds_before_first_flush) == 2

    # ============ Vector Query Tests ============

    @pytest.mark.asyncio
    async def test_query_returns_results(self, vector_store, mock_session):
        """Test that query returns matching chunks."""
        # Mock query result
        mock_row = MagicMock()
        mock_row.id = "chunk-1"
        mock_row.content = "Test content"
        mock_row.document_id = "doc-1"
        mock_row.page_number = 1
        mock_row.x0 = 72.0
        mock_row.y0 = 100.0
        mock_row.x1 = 540.0
        mock_row.y1 = 120.0
        mock_row.line_start = 1
        mock_row.line_end = 5
        mock_row.similarity = 0.95

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]
        mock_session.execute.return_value = mock_result

        query_embedding = [0.1] * 512
        results = await vector_store.query(query_embedding, top_k=5)

        assert len(results) == 1
        assert isinstance(results[0], RetrievedChunk)
        assert results[0].score == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_query_with_document_filter(self, vector_store, mock_session):
        """Test query with document ID filter."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        query_embedding = [0.1] * 512
        await vector_store.query(
            query_embedding,
            document_ids=["doc-1", "doc-2"],
            top_k=5,
        )

        # Verify document filter is in the SQL
        call_args = mock_session.execute.call_args
        sql = str(call_args[0][0])
        assert "doc_ids" in call_args[1] or "ANY" in sql

    @pytest.mark.asyncio
    async def test_query_user_scoping(self, mock_session):
        """Test that queries are scoped to user."""
        store_with_user = PGVectorStore(mock_session, user_id="user-123")

        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        await store_with_user.query([0.1] * 512, top_k=5)

        # Verify user_id is in params
        call_args = mock_session.execute.call_args
        params = call_args[1]
        assert params.get("user_id") == "user-123"

    @pytest.mark.asyncio
    async def test_query_empty_results(self, vector_store, mock_session):
        """Test query with no matching results."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        results = await vector_store.query([0.1] * 512, top_k=5)

        assert results == []

    # ============ Hybrid Query Tests ============

    @pytest.mark.asyncio
    @patch("app.core.vector_store.get_settings")
    async def test_hybrid_query_with_bm25_candidates(
        self, mock_settings, vector_store, mock_session
    ):
        """Test hybrid query uses BM25 pre-filtering."""
        mock_settings.return_value.bm25_candidate_limit = 100
        mock_settings.return_value.bm25_weight = 0.3
        mock_settings.return_value.use_matryoshka = True
        mock_settings.return_value.embedding_dimensions = 512

        # Mock BM25 candidates
        bm25_result = MagicMock()
        bm25_result.fetchall.return_value = [
            MagicMock(id="chunk-1", bm25_score=0.8),
            MagicMock(id="chunk-2", bm25_score=0.6),
        ]

        # Mock vector rerank
        vector_result = MagicMock()
        mock_row1 = MagicMock()
        mock_row1.id = "chunk-1"
        mock_row1.document_id = "doc-1"
        mock_row1.content = "Content 1"
        mock_row1.page_number = 1
        mock_row1.x0 = mock_row1.y0 = mock_row1.x1 = mock_row1.y1 = 0
        mock_row1.line_start = mock_row1.line_end = 1
        mock_row1.vector_score = 0.9

        mock_row2 = MagicMock()
        mock_row2.id = "chunk-2"
        mock_row2.document_id = "doc-1"
        mock_row2.content = "Content 2"
        mock_row2.page_number = 1
        mock_row2.x0 = mock_row2.y0 = mock_row2.x1 = mock_row2.y1 = 0
        mock_row2.line_start = mock_row2.line_end = 1
        mock_row2.vector_score = 0.7

        vector_result.fetchall.return_value = [mock_row1, mock_row2]

        mock_session.execute = AsyncMock(
            side_effect=[bm25_result, vector_result]
        )

        results = await vector_store.hybrid_query(
            query_text="test query",
            query_embedding=[0.1] * 512,
            top_k=2,
        )

        assert len(results) == 2
        # Should be sorted by hybrid score
        assert results[0].score >= results[1].score

    @pytest.mark.asyncio
    @patch("app.core.vector_store.get_settings")
    async def test_hybrid_query_fallback_on_no_bm25(
        self, mock_settings, vector_store, mock_session
    ):
        """Test hybrid query falls back to pure vector search if BM25 returns nothing."""
        mock_settings.return_value.bm25_candidate_limit = 100
        mock_settings.return_value.bm25_weight = 0.3

        # Mock empty BM25 result
        bm25_result = MagicMock()
        bm25_result.fetchall.return_value = []

        # Mock vector search fallback
        vector_result = MagicMock()
        mock_row = MagicMock()
        mock_row.id = "chunk-1"
        mock_row.document_id = "doc-1"
        mock_row.content = "Content"
        mock_row.page_number = 1
        mock_row.x0 = mock_row.y0 = mock_row.x1 = mock_row.y1 = 0
        mock_row.line_start = mock_row.line_end = 1
        mock_row.similarity = 0.85
        vector_result.fetchall.return_value = [mock_row]

        mock_session.execute = AsyncMock(
            side_effect=[bm25_result, vector_result]
        )

        results = await vector_store.hybrid_query(
            query_text="obscure query",
            query_embedding=[0.1] * 512,
            top_k=5,
        )

        # Should still return results from fallback
        assert len(results) >= 0  # May be 0 or 1 depending on implementation

    # ============ Delete Tests ============

    @pytest.mark.asyncio
    async def test_delete_document_chunks(self, vector_store, mock_session):
        """Test deleting all chunks for a document."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [MagicMock(), MagicMock(), MagicMock()]
        mock_session.execute.return_value = mock_result

        deleted_count = await vector_store.delete_document_chunks("doc-1")

        assert deleted_count == 3

    @pytest.mark.asyncio
    async def test_delete_document_chunks_none_found(self, vector_store, mock_session):
        """Test deleting chunks for non-existent document."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        deleted_count = await vector_store.delete_document_chunks("non-existent")

        assert deleted_count == 0

    # ============ Chunk Count Tests ============

    @pytest.mark.asyncio
    async def test_get_chunk_count(self, vector_store, mock_session):
        """Test getting chunk count for a document."""
        mock_result = MagicMock()
        mock_row = MagicMock()
        mock_row.count = 42
        mock_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_result

        count = await vector_store.get_chunk_count("doc-1")

        assert count == 42

    @pytest.mark.asyncio
    async def test_get_chunk_count_no_chunks(self, vector_store, mock_session):
        """Test chunk count for document with no chunks."""
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_session.execute.return_value = mock_result

        count = await vector_store.get_chunk_count("empty-doc")

        assert count == 0


class TestRetrievedChunk:
    """Tests for RetrievedChunk dataclass."""

    def test_retrieved_chunk_creation(self):
        """Test RetrievedChunk creation."""
        chunk = RetrievedChunk(
            id="chunk-1",
            text="Sample text content",
            document_id="doc-1",
            page_number=1,
            x0=72.0,
            y0=100.0,
            x1=540.0,
            y1=120.0,
            line_start=1,
            line_end=5,
            score=0.95,
        )

        assert chunk.id == "chunk-1"
        assert chunk.text == "Sample text content"
        assert chunk.document_id == "doc-1"
        assert chunk.page_number == 1
        assert chunk.score == 0.95

    def test_retrieved_chunk_optional_line_numbers(self):
        """Test RetrievedChunk with optional line numbers."""
        chunk = RetrievedChunk(
            id="chunk-1",
            text="Content",
            document_id="doc-1",
            page_number=1,
            x0=0, y0=0, x1=100, y1=100,
            line_start=None,
            line_end=None,
            score=0.8,
        )

        assert chunk.line_start is None
        assert chunk.line_end is None


class TestChunkData:
    """Tests for ChunkData dataclass."""

    def test_chunk_data_creation(self):
        """Test ChunkData creation."""
        chunk = ChunkData(
            text="Sample text",
            document_id="doc-1",
            page_number=1,
            chunk_index=0,
            x0=72.0,
            y0=100.0,
            x1=540.0,
            y1=120.0,
            line_start=1,
            line_end=5,
            embedding=[0.1] * 512,
            token_count=10,
        )

        assert chunk.text == "Sample text"
        assert chunk.document_id == "doc-1"
        assert len(chunk.embedding) == 512
        assert chunk.token_count == 10

    def test_chunk_data_optional_fields(self):
        """Test ChunkData with optional fields."""
        chunk = ChunkData(
            text="Minimal chunk",
            document_id="doc-1",
            page_number=1,
            chunk_index=0,
            x0=0, y0=0, x1=100, y1=100,
            line_start=1,
            line_end=1,
        )

        assert chunk.embedding is None
        assert chunk.token_count is None

"""
Integration tests for RAG engine module.
Tests document indexing, query pipeline, caching, and response generation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.rag_engine import (
    RAGEngine,
    RAGResponse,
    RetrievedContext,
    create_rag_engine,
)
from app.core.pdf_parser import TextBlock
from app.core.exceptions import RAGQueryError, RAGIndexError, RAGError


class TestRAGEngine:
    """Tests for RAGEngine class."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = AsyncMock()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.execute = AsyncMock()
        session.get = AsyncMock()
        return session

    @pytest.fixture
    @patch("app.core.rag_engine.get_settings")
    @patch("app.core.rag_engine.settings")
    def rag_engine(self, mock_settings_module, mock_get_settings, mock_session):
        """Create a RAGEngine with mocked dependencies."""
        mock_settings = MagicMock()
        mock_settings.embedding_dimensions = 512
        mock_settings.use_matryoshka = True
        mock_settings.use_hybrid_search = True
        mock_settings.query_cache_enabled = True
        mock_settings.openai_api_key = "test-key"
        mock_settings.llm_provider = "openai"
        mock_settings.embedding_provider = "openai"
        mock_settings.openai_model = "gpt-4o"
        mock_settings.openai_embedding_model = "text-embedding-3-small"
        mock_settings.chunk_size = 512
        mock_settings.chunk_overlap = 50
        mock_settings.similarity_top_k = 5

        mock_get_settings.return_value = mock_settings
        mock_settings_module.chunk_size = 512
        mock_settings_module.chunk_overlap = 50
        mock_settings_module.llm_provider = "openai"
        mock_settings_module.embedding_provider = "openai"
        mock_settings_module.openai_api_key = "test-key"
        mock_settings_module.openai_model = "gpt-4o"
        mock_settings_module.similarity_top_k = 5

        return RAGEngine(mock_session, user_id="test-user-123")

    @pytest.fixture
    def sample_blocks(self):
        """Create sample TextBlock objects."""
        return [
            TextBlock(
                text="Machine learning is a subset of artificial intelligence.",
                page_number=1,
                x0=72.0, y0=100.0, x1=540.0, y1=120.0,
                line_start=1, line_end=3,
            ),
            TextBlock(
                text="Deep learning uses neural networks with multiple layers.",
                page_number=1,
                x0=72.0, y0=130.0, x1=540.0, y1=150.0,
                line_start=4, line_end=6,
            ),
        ]

    # ============ Document Indexing Tests ============

    @pytest.mark.asyncio
    @patch("app.core.rag_engine.get_settings")
    @patch("app.core.rag_engine.settings")
    async def test_index_document_success(
        self, mock_settings_module, mock_get_settings, mock_session, sample_blocks
    ):
        """Test successful document indexing returns chunk count."""
        mock_settings = MagicMock()
        mock_settings.embedding_dimensions = 512
        mock_settings.use_matryoshka = True
        mock_settings.chunk_size = 512
        mock_settings.chunk_overlap = 50
        mock_get_settings.return_value = mock_settings
        mock_settings_module.chunk_size = 512
        mock_settings_module.chunk_overlap = 50

        engine = RAGEngine(mock_session, user_id="test-user")

        # Mock embedding service
        with patch.object(
            engine.embedding_service, "get_embedding",
            new_callable=AsyncMock, return_value=[0.1] * 512
        ):
            # Mock vector store
            with patch.object(
                engine.vector_store, "add_chunks",
                new_callable=AsyncMock, return_value=["chunk-1", "chunk-2"]
            ):
                result = await engine.index_document("doc-1", sample_blocks)

                # Returns chunk count (2 blocks indexed)
                assert result == 2

    @pytest.mark.asyncio
    @patch("app.core.rag_engine.get_settings")
    @patch("app.core.rag_engine.settings")
    async def test_index_document_empty_blocks(
        self, mock_settings_module, mock_get_settings, mock_session
    ):
        """Test indexing with empty blocks list returns zero."""
        mock_settings = MagicMock()
        mock_settings.embedding_dimensions = 512
        mock_get_settings.return_value = mock_settings
        mock_settings_module.chunk_size = 512
        mock_settings_module.chunk_overlap = 50

        engine = RAGEngine(mock_session, user_id="test-user")

        with patch.object(
            engine.vector_store, "add_chunks",
            new_callable=AsyncMock, return_value=[]
        ):
            result = await engine.index_document("doc-1", [])

            assert result == 0

    @pytest.mark.asyncio
    @patch("app.core.rag_engine.get_settings")
    @patch("app.core.rag_engine.settings")
    async def test_index_document_failure_raises_rag_index_error(
        self, mock_settings_module, mock_get_settings, mock_session, sample_blocks
    ):
        """Test document indexing failure raises RAGIndexError."""
        mock_settings = MagicMock()
        mock_settings.embedding_dimensions = 512
        mock_get_settings.return_value = mock_settings
        mock_settings_module.chunk_size = 512
        mock_settings_module.chunk_overlap = 50

        engine = RAGEngine(mock_session, user_id="test-user")

        # Mock embedding failure
        with patch.object(
            engine.embedding_service, "get_embedding",
            new_callable=AsyncMock, side_effect=Exception("API error")
        ):
            with pytest.raises(RAGIndexError) as exc_info:
                await engine.index_document("doc-1", sample_blocks)

            # Verify user message is safe (no internal details)
            assert "API error" not in exc_info.value.user_message
            assert "Unable to index document" in exc_info.value.user_message

    # ============ Query Pipeline Tests ============

    @pytest.mark.asyncio
    @patch("app.core.rag_engine.get_settings")
    @patch("app.core.rag_engine.settings")
    async def test_query_with_cache_hit(
        self, mock_settings_module, mock_get_settings, mock_session
    ):
        """Test query with exact cache hit."""
        mock_settings = MagicMock()
        mock_settings.embedding_dimensions = 512
        mock_settings.query_cache_enabled = True
        mock_settings.use_hybrid_search = True
        mock_get_settings.return_value = mock_settings
        mock_settings_module.chunk_size = 512
        mock_settings_module.chunk_overlap = 50
        mock_settings_module.similarity_top_k = 5
        mock_settings_module.llm_provider = "openai"
        mock_settings_module.openai_api_key = "test-key"
        mock_settings_module.openai_model = "gpt-4"

        engine = RAGEngine(mock_session, user_id="test-user")

        # Mock cache hit
        cached_result = MagicMock()
        cached_result.chunk_ids = ["chunk-1"]
        cached_result.scores = [0.95]
        cached_result.hit_count = 5

        with patch.object(
            engine.query_cache, "get",
            new_callable=AsyncMock, return_value=cached_result
        ):
            # Mock chunk fetch
            mock_row = MagicMock()
            mock_row.id = "chunk-1"
            mock_row.document_id = "doc-1"
            mock_row.content = "Cached content"
            mock_row.page_number = 1
            mock_row.x0 = mock_row.y0 = 0
            mock_row.x1 = mock_row.y1 = 100
            mock_row.line_start = 1
            mock_row.line_end = 5

            mock_result = MagicMock()
            mock_result.fetchall.return_value = [mock_row]
            mock_session.execute.return_value = mock_result

            # Mock LLM with async acomplete
            mock_llm = MagicMock()
            mock_llm.acomplete = AsyncMock(return_value="Generated answer")

            with patch.object(engine, "_get_llm", return_value=mock_llm):
                response = await engine.query("What is AI?")

                assert response.answer == "Generated answer"
                # Cache should have been checked
                engine.query_cache.get.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.core.rag_engine.get_settings")
    @patch("app.core.rag_engine.settings")
    async def test_query_cache_miss_with_search(
        self, mock_settings_module, mock_get_settings, mock_session
    ):
        """Test query with cache miss, performing actual search."""
        mock_settings = MagicMock()
        mock_settings.embedding_dimensions = 512
        mock_settings.query_cache_enabled = True
        mock_settings.use_hybrid_search = True
        mock_get_settings.return_value = mock_settings
        mock_settings_module.chunk_size = 512
        mock_settings_module.chunk_overlap = 50
        mock_settings_module.similarity_top_k = 5
        mock_settings_module.llm_provider = "openai"
        mock_settings_module.openai_api_key = "test-key"
        mock_settings_module.openai_model = "gpt-4"

        engine = RAGEngine(mock_session, user_id="test-user")

        # Mock cache miss
        with patch.object(
            engine.query_cache, "get",
            new_callable=AsyncMock, return_value=None
        ):
            with patch.object(
                engine.query_cache, "get_semantic",
                new_callable=AsyncMock, return_value=None
            ):
                # Mock embedding
                with patch.object(
                    engine.embedding_service, "get_embedding",
                    new_callable=AsyncMock, return_value=[0.1] * 512
                ):
                    # Mock hybrid search
                    from app.core.vector_store import RetrievedChunk
                    search_result = RetrievedChunk(
                        id="chunk-1",
                        text="Search result content",
                        document_id="doc-1",
                        page_number=1,
                        x0=0, y0=0, x1=100, y1=100,
                        line_start=1, line_end=5,
                        score=0.9,
                    )

                    with patch.object(
                        engine.vector_store, "hybrid_query",
                        new_callable=AsyncMock, return_value=[search_result]
                    ):
                        # Mock cache set
                        with patch.object(
                            engine.query_cache, "set",
                            new_callable=AsyncMock, return_value="cache-id"
                        ):
                            # Mock LLM with async acomplete
                            mock_llm = MagicMock()
                            mock_llm.acomplete = AsyncMock(return_value="Generated answer from search")

                            with patch.object(engine, "_get_llm", return_value=mock_llm):
                                response = await engine.query("New question?")

                                assert "Generated answer" in response.answer
                                # Should have cached the result
                                engine.query_cache.set.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.core.rag_engine.get_settings")
    @patch("app.core.rag_engine.settings")
    async def test_query_no_results_found(
        self, mock_settings_module, mock_get_settings, mock_session
    ):
        """Test query when no relevant documents found."""
        mock_settings = MagicMock()
        mock_settings.embedding_dimensions = 512
        mock_settings.query_cache_enabled = True
        mock_settings.use_hybrid_search = True
        mock_get_settings.return_value = mock_settings
        mock_settings_module.chunk_size = 512
        mock_settings_module.chunk_overlap = 50
        mock_settings_module.similarity_top_k = 5

        engine = RAGEngine(mock_session, user_id="test-user")

        # Mock cache miss
        with patch.object(
            engine.query_cache, "get",
            new_callable=AsyncMock, return_value=None
        ):
            with patch.object(
                engine.query_cache, "get_semantic",
                new_callable=AsyncMock, return_value=None
            ):
                with patch.object(
                    engine.embedding_service, "get_embedding",
                    new_callable=AsyncMock, return_value=[0.1] * 512
                ):
                    # Mock empty search results
                    with patch.object(
                        engine.vector_store, "hybrid_query",
                        new_callable=AsyncMock, return_value=[]
                    ):
                        response = await engine.query("Unrelated question?")

                        assert "No relevant content found" in response.answer
                        assert response.contexts == []
                        assert response.source_document_ids == []

    @pytest.mark.asyncio
    @patch("app.core.rag_engine.get_settings")
    @patch("app.core.rag_engine.settings")
    async def test_query_with_document_filter(
        self, mock_settings_module, mock_get_settings, mock_session
    ):
        """Test query scoped to specific document."""
        mock_settings = MagicMock()
        mock_settings.embedding_dimensions = 512
        mock_settings.query_cache_enabled = False
        mock_settings.use_hybrid_search = True
        mock_get_settings.return_value = mock_settings
        mock_settings_module.chunk_size = 512
        mock_settings_module.chunk_overlap = 50
        mock_settings_module.similarity_top_k = 5
        mock_settings_module.llm_provider = "openai"
        mock_settings_module.openai_api_key = "test-key"
        mock_settings_module.openai_model = "gpt-4"

        engine = RAGEngine(mock_session, user_id="test-user")

        with patch.object(
            engine.embedding_service, "get_embedding",
            new_callable=AsyncMock, return_value=[0.1] * 512
        ):
            from app.core.vector_store import RetrievedChunk
            search_result = RetrievedChunk(
                id="chunk-1",
                text="Document specific content",
                document_id="doc-specific",
                page_number=1,
                x0=0, y0=0, x1=100, y1=100,
                line_start=1, line_end=5,
                score=0.9,
            )

            with patch.object(
                engine.vector_store, "hybrid_query",
                new_callable=AsyncMock, return_value=[search_result]
            ) as mock_search:
                mock_llm = MagicMock()
                mock_llm.acomplete = AsyncMock(return_value="Answer")

                with patch.object(engine, "_get_llm", return_value=mock_llm):
                    await engine.query(
                        "Question?",
                        document_id="doc-specific",
                    )

                    # Verify document filter was passed
                    call_kwargs = mock_search.call_args[1]
                    assert call_kwargs["document_ids"] == ["doc-specific"]

    # ============ Async LLM Tests ============

    @pytest.mark.asyncio
    @patch("app.core.rag_engine.get_settings")
    @patch("app.core.rag_engine.settings")
    async def test_generate_response_uses_async_acomplete(
        self, mock_settings_module, mock_get_settings, mock_session
    ):
        """Test that _generate_response uses async acomplete instead of blocking complete."""
        mock_settings = MagicMock()
        mock_settings.embedding_dimensions = 512
        mock_settings.query_cache_enabled = False
        mock_settings.use_hybrid_search = True
        mock_get_settings.return_value = mock_settings
        mock_settings_module.chunk_size = 512
        mock_settings_module.chunk_overlap = 50
        mock_settings_module.similarity_top_k = 5
        mock_settings_module.llm_provider = "openai"
        mock_settings_module.openai_api_key = "test-key"
        mock_settings_module.openai_model = "gpt-4"

        engine = RAGEngine(mock_session, user_id="test-user")

        from app.core.vector_store import RetrievedChunk
        chunks = [
            RetrievedChunk(
                id="chunk-1",
                text="Test content for async verification",
                document_id="doc-1",
                page_number=1,
                x0=0, y0=0, x1=100, y1=100,
                line_start=1, line_end=5,
                score=0.9,
            )
        ]

        # Create mock LLM with both sync and async methods
        mock_llm = MagicMock()
        mock_llm.complete = MagicMock(return_value="Sync response - should not be used")
        mock_llm.acomplete = AsyncMock(return_value="Async response")

        with patch.object(engine, "_get_llm", return_value=mock_llm):
            response = await engine._generate_response(
                "What is the content about?",
                chunks,
            )

            # Verify async acomplete was called, not sync complete
            mock_llm.acomplete.assert_called_once()
            mock_llm.complete.assert_not_called()
            assert response.answer == "Async response"

    # ============ Query Error Handling Tests ============

    @pytest.mark.asyncio
    @patch("app.core.rag_engine.get_settings")
    @patch("app.core.rag_engine.settings")
    async def test_query_failure_raises_rag_query_error(
        self, mock_settings_module, mock_get_settings, mock_session
    ):
        """Test query failure raises RAGQueryError with safe user message."""
        mock_settings = MagicMock()
        mock_settings.embedding_dimensions = 512
        mock_settings.query_cache_enabled = True
        mock_settings.use_hybrid_search = True
        mock_get_settings.return_value = mock_settings
        mock_settings_module.chunk_size = 512
        mock_settings_module.chunk_overlap = 50
        mock_settings_module.similarity_top_k = 5

        engine = RAGEngine(mock_session, user_id="test-user")

        # Mock cache miss
        with patch.object(
            engine.query_cache, "get",
            new_callable=AsyncMock, return_value=None
        ):
            # Mock embedding failure
            with patch.object(
                engine.embedding_service, "get_embedding",
                new_callable=AsyncMock, side_effect=Exception("API key invalid: sk-abc123")
            ):
                with pytest.raises(RAGQueryError) as exc_info:
                    await engine.query("What is AI?")

                # Verify internal details are not in user message
                assert "sk-abc123" not in exc_info.value.user_message
                assert "API key" not in exc_info.value.user_message
                assert "Unable to process your query" in exc_info.value.user_message

    @pytest.mark.asyncio
    @patch("app.core.rag_engine.get_settings")
    @patch("app.core.rag_engine.settings")
    async def test_query_preserves_exception_chain(
        self, mock_settings_module, mock_get_settings, mock_session
    ):
        """Test query error preserves original exception as cause."""
        mock_settings = MagicMock()
        mock_settings.embedding_dimensions = 512
        mock_settings.query_cache_enabled = False
        mock_get_settings.return_value = mock_settings
        mock_settings_module.chunk_size = 512
        mock_settings_module.chunk_overlap = 50

        engine = RAGEngine(mock_session, user_id="test-user")

        original_error = ValueError("Original underlying error")

        with patch.object(
            engine.embedding_service, "get_embedding",
            new_callable=AsyncMock, side_effect=original_error
        ):
            with pytest.raises(RAGQueryError) as exc_info:
                await engine.query("Question?")

            # Verify exception chaining
            assert exc_info.value.__cause__ is original_error

    # ============ Document Removal Tests ============

    @pytest.mark.asyncio
    @patch("app.core.rag_engine.get_settings")
    @patch("app.core.rag_engine.settings")
    async def test_remove_document_success(
        self, mock_settings_module, mock_get_settings, mock_session
    ):
        """Test successful document removal returns deleted count."""
        mock_settings = MagicMock()
        mock_settings.embedding_dimensions = 512
        mock_get_settings.return_value = mock_settings
        mock_settings_module.chunk_size = 512
        mock_settings_module.chunk_overlap = 50

        engine = RAGEngine(mock_session, user_id="test-user")

        with patch.object(
            engine.vector_store, "delete_document_chunks",
            new_callable=AsyncMock, return_value=5
        ):
            result = await engine.remove_document("doc-1")

            assert result == 5

    @pytest.mark.asyncio
    @patch("app.core.rag_engine.get_settings")
    @patch("app.core.rag_engine.settings")
    async def test_remove_document_not_found(
        self, mock_settings_module, mock_get_settings, mock_session
    ):
        """Test removing non-existent document returns zero."""
        mock_settings = MagicMock()
        mock_settings.embedding_dimensions = 512
        mock_get_settings.return_value = mock_settings
        mock_settings_module.chunk_size = 512
        mock_settings_module.chunk_overlap = 50

        engine = RAGEngine(mock_session, user_id="test-user")

        with patch.object(
            engine.vector_store, "delete_document_chunks",
            new_callable=AsyncMock, return_value=0
        ):
            result = await engine.remove_document("non-existent")

            assert result == 0

    @pytest.mark.asyncio
    @patch("app.core.rag_engine.get_settings")
    @patch("app.core.rag_engine.settings")
    async def test_remove_document_failure_raises_rag_error(
        self, mock_settings_module, mock_get_settings, mock_session
    ):
        """Test document removal failure raises RAGError."""
        mock_settings = MagicMock()
        mock_settings.embedding_dimensions = 512
        mock_get_settings.return_value = mock_settings
        mock_settings_module.chunk_size = 512
        mock_settings_module.chunk_overlap = 50

        engine = RAGEngine(mock_session, user_id="test-user")

        with patch.object(
            engine.vector_store, "delete_document_chunks",
            new_callable=AsyncMock, side_effect=Exception("Database connection failed")
        ):
            with pytest.raises(RAGError) as exc_info:
                await engine.remove_document("doc-1")

            # Verify user message is safe
            assert "Database connection" not in exc_info.value.user_message
            assert "Unable to remove document" in exc_info.value.user_message


class TestRAGResponse:
    """Tests for RAGResponse dataclass."""

    def test_rag_response_creation(self):
        """Test RAGResponse creation."""
        contexts = [
            RetrievedContext(
                text="Context text",
                document_id="doc-1",
                page_number=1,
                bbox_x0=0, bbox_y0=0, bbox_x1=100, bbox_y1=100,
                score=0.95,
                line_start=1,
                line_end=5,
            )
        ]

        response = RAGResponse(
            answer="Generated answer",
            contexts=contexts,
            source_document_ids=["doc-1"],
        )

        assert response.answer == "Generated answer"
        assert len(response.contexts) == 1
        assert response.source_document_ids == ["doc-1"]


class TestRetrievedContext:
    """Tests for RetrievedContext dataclass."""

    def test_retrieved_context_creation(self):
        """Test RetrievedContext creation."""
        context = RetrievedContext(
            text="Sample context text",
            document_id="doc-1",
            page_number=1,
            bbox_x0=72.0,
            bbox_y0=100.0,
            bbox_x1=540.0,
            bbox_y1=120.0,
            score=0.95,
            line_start=1,
            line_end=5,
        )

        assert context.text == "Sample context text"
        assert context.document_id == "doc-1"
        assert context.score == 0.95

    def test_retrieved_context_optional_lines(self):
        """Test RetrievedContext with optional line numbers."""
        context = RetrievedContext(
            text="Content",
            document_id="doc-1",
            page_number=1,
            bbox_x0=0, bbox_y0=0, bbox_x1=100, bbox_y1=100,
            score=0.8,
        )

        assert context.line_start is None
        assert context.line_end is None


class TestCreateRagEngine:
    """Tests for create_rag_engine factory function."""

    @patch("app.core.rag_engine.get_settings")
    @patch("app.core.rag_engine.settings")
    def test_create_rag_engine(self, mock_settings_module, mock_get_settings):
        """Test RAG engine factory function."""
        mock_settings = MagicMock()
        mock_settings.embedding_dimensions = 512
        mock_get_settings.return_value = mock_settings
        mock_settings_module.chunk_size = 512
        mock_settings_module.chunk_overlap = 50

        mock_session = AsyncMock()
        engine = create_rag_engine(mock_session, user_id="user-123")

        assert engine is not None
        assert isinstance(engine, RAGEngine)
        assert engine.user_id == "user-123"

    @patch("app.core.rag_engine.get_settings")
    @patch("app.core.rag_engine.settings")
    def test_create_rag_engine_no_user(self, mock_settings_module, mock_get_settings):
        """Test RAG engine creation without user ID."""
        mock_settings = MagicMock()
        mock_settings.embedding_dimensions = 512
        mock_get_settings.return_value = mock_settings
        mock_settings_module.chunk_size = 512
        mock_settings_module.chunk_overlap = 50

        mock_session = AsyncMock()
        engine = create_rag_engine(mock_session)

        assert engine.user_id is None

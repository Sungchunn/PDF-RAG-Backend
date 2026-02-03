"""
API tests for chat endpoints.
Tests RAG Q&A, citations, and document-scoped queries.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import status
from httpx import AsyncClient

from app.db.models import DocumentModel


class TestChatEndpoint:
    """Tests for POST /api/chat/."""

    @pytest.fixture
    def mock_rag_response(self):
        """Create a mock RAG response."""
        from app.core.rag_engine import RAGResponse, RetrievedContext

        return RAGResponse(
            answer="This is the generated answer based on the documents.",
            contexts=[
                RetrievedContext(
                    text="Source text from the document.",
                    document_id="doc-1",
                    page_number=1,
                    bbox_x0=72.0,
                    bbox_y0=100.0,
                    bbox_x1=540.0,
                    bbox_y1=120.0,
                    score=0.95,
                    line_start=1,
                    line_end=5,
                ),
            ],
            source_document_ids=["doc-1"],
        )

    @pytest.mark.asyncio
    @patch("app.api.routes.chat.settings")
    @patch("app.api.routes.chat.RAGEngine")
    async def test_chat_success(
        self, mock_rag_engine_class, mock_settings,
        test_client, test_user, auth_headers, mock_rag_response
    ):
        """Test successful chat message."""
        mock_settings.llm_provider = "openai"
        mock_settings.embedding_provider = "openai"
        mock_settings.openai_api_key = "test-key"

        mock_engine = AsyncMock()
        mock_engine.query = AsyncMock(return_value=mock_rag_response)
        mock_rag_engine_class.return_value = mock_engine

        response = await test_client.post(
            "/api/chat/",
            headers=auth_headers,
            json={"message": "What is machine learning?"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "answer" in data
        assert "citations" in data
        assert "sourceDocuments" in data
        assert data["answer"] == mock_rag_response.answer

    @pytest.mark.asyncio
    @patch("app.api.routes.chat.settings")
    @patch("app.api.routes.chat.RAGEngine")
    async def test_chat_with_document_id(
        self, mock_rag_engine_class, mock_settings,
        test_client, test_user, test_document, auth_headers, mock_rag_response
    ):
        """Test chat scoped to specific document."""
        mock_settings.llm_provider = "openai"
        mock_settings.embedding_provider = "openai"
        mock_settings.openai_api_key = "test-key"

        mock_engine = AsyncMock()
        mock_engine.query = AsyncMock(return_value=mock_rag_response)
        mock_rag_engine_class.return_value = mock_engine

        response = await test_client.post(
            "/api/chat/",
            headers=auth_headers,
            json={
                "message": "Summarize this document",
                "documentId": test_document.id,
            },
        )

        assert response.status_code == status.HTTP_200_OK
        mock_engine.query.assert_called_once()
        call_args = mock_engine.query.call_args
        assert call_args[1]["document_id"] == test_document.id

    @pytest.mark.asyncio
    @patch("app.api.routes.chat.settings")
    async def test_chat_document_not_found(
        self, mock_settings, test_client, auth_headers
    ):
        """Test chat with non-existent document."""
        mock_settings.llm_provider = "openai"
        mock_settings.embedding_provider = "openai"
        mock_settings.openai_api_key = "test-key"

        response = await test_client.post(
            "/api/chat/",
            headers=auth_headers,
            json={
                "message": "Question?",
                "documentId": "non-existent-doc",
            },
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    @patch("app.api.routes.chat.settings")
    async def test_chat_document_not_ready(
        self, mock_settings, test_client, test_user, db_session, auth_headers
    ):
        """Test chat with document still processing."""
        mock_settings.llm_provider = "openai"
        mock_settings.embedding_provider = "openai"
        mock_settings.openai_api_key = "test-key"

        # Create processing document
        processing_doc = DocumentModel(
            id="processing-doc",
            user_id=test_user.id,
            name="processing.pdf",
            size_bytes=1024,
            mime_type="application/pdf",
            status="processing",  # Not ready
            storage_key="/uploads/processing.pdf",
            checksum_sha256="abc123",
            uploaded_at=datetime.now(timezone.utc),
        )
        db_session.add(processing_doc)
        await db_session.flush()

        response = await test_client.post(
            "/api/chat/",
            headers=auth_headers,
            json={
                "message": "Question?",
                "documentId": "processing-doc",
            },
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "not ready" in response.json()["detail"]

    @pytest.mark.asyncio
    @patch("app.api.routes.chat.settings")
    async def test_chat_document_other_user(
        self, mock_settings, test_client, db_session, auth_headers
    ):
        """Test chat with document owned by different user."""
        mock_settings.llm_provider = "openai"
        mock_settings.embedding_provider = "openai"
        mock_settings.openai_api_key = "test-key"

        # Create document owned by different user
        other_doc = DocumentModel(
            id="other-user-doc-chat",
            user_id="other-user-id",
            name="other.pdf",
            size_bytes=1024,
            mime_type="application/pdf",
            status="ready",
            storage_key="/uploads/other.pdf",
            checksum_sha256="xyz789",
            uploaded_at=datetime.now(timezone.utc),
        )
        db_session.add(other_doc)
        await db_session.flush()

        response = await test_client.post(
            "/api/chat/",
            headers=auth_headers,
            json={
                "message": "Question?",
                "documentId": "other-user-doc-chat",
            },
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    @patch("app.api.routes.chat.settings")
    async def test_chat_without_api_key(
        self, mock_settings, test_client, auth_headers
    ):
        """Test chat when API key is not configured."""
        mock_settings.llm_provider = "openai"
        mock_settings.embedding_provider = "openai"
        mock_settings.openai_api_key = ""  # No API key

        response = await test_client.post(
            "/api/chat/",
            headers=auth_headers,
            json={"message": "Question?"},
        )

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "API key" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_chat_without_auth(self, test_client):
        """Test chat without authentication."""
        response = await test_client.post(
            "/api/chat/",
            json={"message": "Question?"},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_chat_missing_message(self, test_client, auth_headers):
        """Test chat without message."""
        response = await test_client.post(
            "/api/chat/",
            headers=auth_headers,
            json={},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    @patch("app.api.routes.chat.settings")
    @patch("app.api.routes.chat.RAGEngine")
    async def test_chat_returns_citations(
        self, mock_rag_engine_class, mock_settings,
        test_client, auth_headers, mock_rag_response
    ):
        """Test that chat returns proper citation format."""
        mock_settings.llm_provider = "openai"
        mock_settings.embedding_provider = "openai"
        mock_settings.openai_api_key = "test-key"

        mock_engine = AsyncMock()
        mock_engine.query = AsyncMock(return_value=mock_rag_response)
        mock_rag_engine_class.return_value = mock_engine

        response = await test_client.post(
            "/api/chat/",
            headers=auth_headers,
            json={"message": "Question?"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify citation structure
        assert len(data["citations"]) == 1
        citation = data["citations"][0]
        assert "documentId" in citation
        assert "pageNumber" in citation
        assert "boundingBox" in citation
        assert "lineStart" in citation
        assert "lineEnd" in citation
        assert "text" in citation
        assert "confidence" in citation

        # Verify bounding box structure
        bbox = citation["boundingBox"]
        assert "x0" in bbox
        assert "y0" in bbox
        assert "x1" in bbox
        assert "y1" in bbox

    @pytest.mark.asyncio
    @patch("app.api.routes.chat.settings")
    @patch("app.api.routes.chat.RAGEngine")
    async def test_chat_with_provider_override(
        self, mock_rag_engine_class, mock_settings,
        test_client, auth_headers, mock_rag_response
    ):
        """Test chat with LLM provider override."""
        mock_settings.llm_provider = "openai"
        mock_settings.embedding_provider = "openai"
        mock_settings.openai_api_key = "test-key"
        mock_settings.gemini_api_key = "test-gemini-key"

        mock_engine = AsyncMock()
        mock_engine.query = AsyncMock(return_value=mock_rag_response)
        mock_rag_engine_class.return_value = mock_engine

        response = await test_client.post(
            "/api/chat/",
            headers=auth_headers,
            json={
                "message": "Question?",
                "provider": "gemini",
                "model": "gemini-1.5-pro",
            },
        )

        # Should pass provider to RAG engine
        mock_engine.query.assert_called_once()
        call_kwargs = mock_engine.query.call_args[1]
        assert call_kwargs.get("llm_provider") == "gemini"
        assert call_kwargs.get("llm_model") == "gemini-1.5-pro"

    @pytest.mark.asyncio
    @patch("app.api.routes.chat.settings")
    @patch("app.api.routes.chat.RAGEngine")
    async def test_chat_rag_engine_error(
        self, mock_rag_engine_class, mock_settings,
        test_client, auth_headers
    ):
        """Test chat when RAG engine raises an error."""
        mock_settings.llm_provider = "openai"
        mock_settings.embedding_provider = "openai"
        mock_settings.openai_api_key = "test-key"

        mock_engine = AsyncMock()
        mock_engine.query = AsyncMock(side_effect=Exception("RAG error"))
        mock_rag_engine_class.return_value = mock_engine

        response = await test_client.post(
            "/api/chat/",
            headers=auth_headers,
            json={"message": "Question?"},
        )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Error processing query" in response.json()["detail"]


class TestChatStreamEndpoint:
    """Tests for POST /api/chat/stream."""

    @pytest.mark.asyncio
    async def test_stream_not_implemented(self, test_client, auth_headers):
        """Test that streaming endpoint returns not implemented."""
        response = await test_client.post(
            "/api/chat/stream",
            headers=auth_headers,
            json={"message": "Question?"},
        )

        assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED

    @pytest.mark.asyncio
    async def test_stream_without_auth(self, test_client):
        """Test streaming without authentication."""
        response = await test_client.post(
            "/api/chat/stream",
            json={"message": "Question?"},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

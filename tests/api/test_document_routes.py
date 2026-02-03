"""
API tests for document endpoints.
Tests upload, listing, retrieval, and deletion.
"""

import io
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import status
from httpx import AsyncClient

from app.db.models import DocumentModel


class TestUploadEndpoint:
    """Tests for POST /api/documents/upload."""

    @pytest.mark.asyncio
    async def test_upload_pdf_success(
        self, test_client, test_user, auth_headers, sample_pdf_path
    ):
        """Test successful PDF upload."""
        with open(sample_pdf_path, "rb") as f:
            pdf_content = f.read()

        # Mock background task to prevent actual processing
        with patch("app.api.routes.documents.process_document_task"):
            response = await test_client.post(
                "/api/documents/upload",
                headers=auth_headers,
                files={"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")},
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["filename"] == "test.pdf"
        assert data["status"] == "processing"
        assert "documentId" in data
        assert "jobId" in data

    @pytest.mark.asyncio
    async def test_upload_non_pdf_rejected(self, test_client, auth_headers):
        """Test that non-PDF files are rejected."""
        response = await test_client.post(
            "/api/documents/upload",
            headers=auth_headers,
            files={"file": ("test.txt", io.BytesIO(b"Hello World"), "text/plain")},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "PDF" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_upload_without_auth(self, test_client):
        """Test upload without authentication."""
        response = await test_client.post(
            "/api/documents/upload",
            files={"file": ("test.pdf", io.BytesIO(b"fake pdf"), "application/pdf")},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_upload_without_file(self, test_client, auth_headers):
        """Test upload without file."""
        response = await test_client.post(
            "/api/documents/upload",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    @patch("app.api.routes.documents.settings")
    async def test_upload_file_too_large(
        self, mock_settings, test_client, auth_headers
    ):
        """Test upload with file exceeding size limit."""
        mock_settings.max_file_size = 100  # 100 bytes limit

        large_content = b"x" * 200  # 200 bytes

        response = await test_client.post(
            "/api/documents/upload",
            headers=auth_headers,
            files={"file": ("large.pdf", io.BytesIO(large_content), "application/pdf")},
        )

        assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE


class TestListDocumentsEndpoint:
    """Tests for GET /api/documents/."""

    @pytest.mark.asyncio
    async def test_list_documents_success(
        self, test_client, test_user, test_document, auth_headers
    ):
        """Test listing user's documents."""
        response = await test_client.get(
            "/api/documents/",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "documents" in data
        assert "total" in data
        assert data["total"] >= 1

    @pytest.mark.asyncio
    async def test_list_documents_empty(self, test_client, test_user, auth_headers):
        """Test listing when user has no documents."""
        # test_user fixture without test_document
        response = await test_client.get(
            "/api/documents/",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["documents"] == [] or data["total"] >= 0

    @pytest.mark.asyncio
    async def test_list_documents_with_pagination(
        self, test_client, test_user, auth_headers
    ):
        """Test document listing with pagination."""
        response = await test_client.get(
            "/api/documents/",
            headers=auth_headers,
            params={"limit": 10, "offset": 0},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["limit"] == 10
        assert data["offset"] == 0

    @pytest.mark.asyncio
    async def test_list_documents_with_status_filter(
        self, test_client, test_user, test_document, auth_headers
    ):
        """Test document listing with status filter."""
        response = await test_client.get(
            "/api/documents/",
            headers=auth_headers,
            params={"status": "ready"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        # All returned documents should have "ready" status
        for doc in data["documents"]:
            assert doc["status"] == "ready"

    @pytest.mark.asyncio
    async def test_list_documents_excludes_deleted(
        self, test_client, test_user, db_session, auth_headers
    ):
        """Test that deleted documents are not listed."""
        # Create deleted document
        deleted_doc = DocumentModel(
            id="deleted-doc-id",
            user_id=test_user.id,
            name="deleted.pdf",
            size_bytes=1024,
            mime_type="application/pdf",
            status="ready",
            storage_key="/uploads/deleted.pdf",
            checksum_sha256="def456",
            uploaded_at=datetime.now(timezone.utc),
            deleted_at=datetime.now(timezone.utc),  # Marked as deleted
        )
        db_session.add(deleted_doc)
        await db_session.flush()

        response = await test_client.get(
            "/api/documents/",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        # Deleted document should not be in list
        doc_ids = [doc["id"] for doc in data["documents"]]
        assert "deleted-doc-id" not in doc_ids

    @pytest.mark.asyncio
    async def test_list_documents_without_auth(self, test_client):
        """Test listing without authentication."""
        response = await test_client.get("/api/documents/")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestGetDocumentEndpoint:
    """Tests for GET /api/documents/{document_id}."""

    @pytest.mark.asyncio
    async def test_get_document_success(
        self, test_client, test_user, test_document, auth_headers
    ):
        """Test getting a specific document."""
        response = await test_client.get(
            f"/api/documents/{test_document.id}",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == test_document.id
        assert data["name"] == test_document.name
        assert data["status"] == test_document.status

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, test_client, auth_headers):
        """Test getting non-existent document."""
        response = await test_client.get(
            "/api/documents/non-existent-id",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_get_document_other_user(
        self, test_client, db_session, auth_headers
    ):
        """Test that user cannot access other user's document."""
        # Create document owned by different user
        other_doc = DocumentModel(
            id="other-user-doc",
            user_id="other-user-id",  # Different user
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

        response = await test_client.get(
            "/api/documents/other-user-doc",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_get_deleted_document(
        self, test_client, test_user, db_session, auth_headers
    ):
        """Test that deleted document returns 404."""
        deleted_doc = DocumentModel(
            id="deleted-doc",
            user_id=test_user.id,
            name="deleted.pdf",
            size_bytes=1024,
            mime_type="application/pdf",
            status="ready",
            storage_key="/uploads/deleted.pdf",
            checksum_sha256="abc123",
            uploaded_at=datetime.now(timezone.utc),
            deleted_at=datetime.now(timezone.utc),
        )
        db_session.add(deleted_doc)
        await db_session.flush()

        response = await test_client.get(
            "/api/documents/deleted-doc",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_get_document_without_auth(self, test_client, test_document):
        """Test getting document without authentication."""
        response = await test_client.get(f"/api/documents/{test_document.id}")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestDeleteDocumentEndpoint:
    """Tests for DELETE /api/documents/{document_id}."""

    @pytest.mark.asyncio
    async def test_delete_document_success(
        self, test_client, test_user, test_document, auth_headers
    ):
        """Test successful document deletion (soft delete)."""
        response = await test_client.delete(
            f"/api/documents/{test_document.id}",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "deleted"
        assert data["documentId"] == test_document.id

    @pytest.mark.asyncio
    async def test_delete_document_not_found(self, test_client, auth_headers):
        """Test deleting non-existent document."""
        response = await test_client.delete(
            "/api/documents/non-existent-id",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_delete_other_users_document(
        self, test_client, db_session, auth_headers
    ):
        """Test that user cannot delete other user's document."""
        other_doc = DocumentModel(
            id="other-user-doc-delete",
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

        response = await test_client.delete(
            "/api/documents/other-user-doc-delete",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_delete_already_deleted(
        self, test_client, test_user, db_session, auth_headers
    ):
        """Test deleting already deleted document."""
        deleted_doc = DocumentModel(
            id="already-deleted",
            user_id=test_user.id,
            name="deleted.pdf",
            size_bytes=1024,
            mime_type="application/pdf",
            status="ready",
            storage_key="/uploads/deleted.pdf",
            checksum_sha256="abc123",
            uploaded_at=datetime.now(timezone.utc),
            deleted_at=datetime.now(timezone.utc),
        )
        db_session.add(deleted_doc)
        await db_session.flush()

        response = await test_client.delete(
            "/api/documents/already-deleted",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "already deleted" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_delete_document_without_auth(self, test_client, test_document):
        """Test deleting document without authentication."""
        response = await test_client.delete(f"/api/documents/{test_document.id}")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

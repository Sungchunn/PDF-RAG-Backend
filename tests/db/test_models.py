"""
Database model tests.
Tests model creation, relationships, and constraints.
"""

import pytest
from datetime import datetime, timezone

from app.db.models import (
    UserModel,
    DocumentModel,
    DocumentChunkModel,
    QueryCacheModel,
    ProcessingJobModel,
)


class TestUserModel:
    """Tests for UserModel."""

    def test_user_creation(self):
        """Test UserModel creation with required fields."""
        user = UserModel(
            id="user-123",
            email="test@example.com",
            password_hash="hashed_password",
            created_at=datetime.now(timezone.utc),
        )

        assert user.id == "user-123"
        assert user.email == "test@example.com"
        assert user.password_hash == "hashed_password"

    def test_user_default_values(self):
        """Test UserModel default values."""
        user = UserModel(
            id="user-123",
            email="test@example.com",
            password_hash="hashed",
            created_at=datetime.now(timezone.utc),
        )

        assert user.is_active is True
        assert user.is_verified is False
        assert user.display_name is None

    def test_user_with_display_name(self):
        """Test UserModel with display name."""
        user = UserModel(
            id="user-123",
            email="test@example.com",
            password_hash="hashed",
            display_name="Test User",
            created_at=datetime.now(timezone.utc),
        )

        assert user.display_name == "Test User"


class TestDocumentModel:
    """Tests for DocumentModel."""

    def test_document_creation(self):
        """Test DocumentModel creation."""
        doc = DocumentModel(
            id="doc-123",
            user_id="user-123",
            name="test.pdf",
            size_bytes=1024,
            mime_type="application/pdf",
            status="processing",
            storage_key="/uploads/test.pdf",
            checksum_sha256="abc123",
            uploaded_at=datetime.now(timezone.utc),
        )

        assert doc.id == "doc-123"
        assert doc.name == "test.pdf"
        assert doc.status == "processing"

    def test_document_optional_fields(self):
        """Test DocumentModel optional fields."""
        doc = DocumentModel(
            id="doc-123",
            user_id="user-123",
            name="test.pdf",
            size_bytes=1024,
            mime_type="application/pdf",
            status="processing",
            storage_key="/uploads/test.pdf",
            checksum_sha256="abc123",
            uploaded_at=datetime.now(timezone.utc),
        )

        assert doc.page_count is None
        assert doc.deleted_at is None

    def test_document_with_page_count(self):
        """Test DocumentModel with page count."""
        doc = DocumentModel(
            id="doc-123",
            user_id="user-123",
            name="test.pdf",
            size_bytes=1024,
            mime_type="application/pdf",
            status="ready",
            storage_key="/uploads/test.pdf",
            checksum_sha256="abc123",
            page_count=10,
            uploaded_at=datetime.now(timezone.utc),
        )

        assert doc.page_count == 10

    def test_document_soft_delete(self):
        """Test DocumentModel soft delete field."""
        doc = DocumentModel(
            id="doc-123",
            user_id="user-123",
            name="deleted.pdf",
            size_bytes=1024,
            mime_type="application/pdf",
            status="ready",
            storage_key="/uploads/deleted.pdf",
            checksum_sha256="abc123",
            uploaded_at=datetime.now(timezone.utc),
            deleted_at=datetime.now(timezone.utc),
        )

        assert doc.deleted_at is not None


class TestDocumentChunkModel:
    """Tests for DocumentChunkModel."""

    def test_chunk_creation(self):
        """Test DocumentChunkModel creation."""
        chunk = DocumentChunkModel(
            id="chunk-123",
            document_id="doc-123",
            page_number=1,
            chunk_index=0,
            content="This is the chunk content.",
            created_at=datetime.now(timezone.utc),
        )

        assert chunk.id == "chunk-123"
        assert chunk.document_id == "doc-123"
        assert chunk.content == "This is the chunk content."

    def test_chunk_with_coordinates(self):
        """Test DocumentChunkModel with bounding box."""
        chunk = DocumentChunkModel(
            id="chunk-123",
            document_id="doc-123",
            page_number=1,
            chunk_index=0,
            content="Content",
            x0=72.0,
            y0=100.0,
            x1=540.0,
            y1=120.0,
            created_at=datetime.now(timezone.utc),
        )

        assert chunk.x0 == 72.0
        assert chunk.y0 == 100.0
        assert chunk.x1 == 540.0
        assert chunk.y1 == 120.0

    def test_chunk_with_line_numbers(self):
        """Test DocumentChunkModel with line numbers."""
        chunk = DocumentChunkModel(
            id="chunk-123",
            document_id="doc-123",
            page_number=1,
            chunk_index=0,
            content="Content",
            line_start=1,
            line_end=5,
            created_at=datetime.now(timezone.utc),
        )

        assert chunk.line_start == 1
        assert chunk.line_end == 5

    def test_chunk_token_count(self):
        """Test DocumentChunkModel with token count."""
        chunk = DocumentChunkModel(
            id="chunk-123",
            document_id="doc-123",
            page_number=1,
            chunk_index=0,
            content="Content with tokens",
            token_count=100,
            created_at=datetime.now(timezone.utc),
        )

        assert chunk.token_count == 100


class TestQueryCacheModel:
    """Tests for QueryCacheModel."""

    def test_cache_creation(self):
        """Test QueryCacheModel creation."""
        cache = QueryCacheModel(
            id="cache-123",
            query_hash="abc123",
            user_id="user-123",
            result_chunk_ids=["chunk-1", "chunk-2"],
            result_scores=[0.95, 0.85],
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc),
        )

        assert cache.id == "cache-123"
        assert cache.query_hash == "abc123"
        assert cache.result_chunk_ids == ["chunk-1", "chunk-2"]

    def test_cache_with_embedding(self):
        """Test QueryCacheModel with query embedding."""
        cache = QueryCacheModel(
            id="cache-123",
            query_hash="abc123",
            query_embedding=[0.1] * 512,
            user_id="user-123",
            result_chunk_ids=["chunk-1"],
            result_scores=[0.9],
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc),
        )

        assert len(cache.query_embedding) == 512

    def test_cache_hit_count(self):
        """Test QueryCacheModel hit count tracking."""
        cache = QueryCacheModel(
            id="cache-123",
            query_hash="abc123",
            user_id="user-123",
            result_chunk_ids=[],
            result_scores=[],
            hit_count=5,
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc),
        )

        assert cache.hit_count == 5

    def test_cache_document_scope(self):
        """Test QueryCacheModel with document scope."""
        cache = QueryCacheModel(
            id="cache-123",
            query_hash="abc123",
            user_id="user-123",
            document_ids=["doc-1", "doc-2"],
            result_chunk_ids=[],
            result_scores=[],
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc),
        )

        assert cache.document_ids == ["doc-1", "doc-2"]


class TestProcessingJobModel:
    """Tests for ProcessingJobModel."""

    def test_job_creation(self):
        """Test ProcessingJobModel creation."""
        job = ProcessingJobModel(
            id="job-123",
            document_id="doc-123",
            job_type="document_upload",
            status="pending",
            created_at=datetime.now(timezone.utc),
        )

        assert job.id == "job-123"
        assert job.job_type == "document_upload"
        assert job.status == "pending"

    def test_job_with_progress(self):
        """Test ProcessingJobModel with progress."""
        job = ProcessingJobModel(
            id="job-123",
            document_id="doc-123",
            job_type="document_upload",
            status="in_progress",
            progress_percent=50,
            created_at=datetime.now(timezone.utc),
        )

        assert job.progress_percent == 50

    def test_job_with_error(self):
        """Test ProcessingJobModel with error message."""
        job = ProcessingJobModel(
            id="job-123",
            document_id="doc-123",
            job_type="document_upload",
            status="failed",
            error_message="Processing failed: timeout",
            created_at=datetime.now(timezone.utc),
        )

        assert job.status == "failed"
        assert "timeout" in job.error_message

    def test_job_completion(self):
        """Test ProcessingJobModel completed status."""
        now = datetime.now(timezone.utc)
        job = ProcessingJobModel(
            id="job-123",
            document_id="doc-123",
            job_type="document_upload",
            status="completed",
            progress_percent=100,
            created_at=now,
            completed_at=now,
        )

        assert job.status == "completed"
        assert job.progress_percent == 100
        assert job.completed_at is not None

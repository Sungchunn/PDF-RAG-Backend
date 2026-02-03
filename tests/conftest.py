"""
Shared test fixtures and configuration for pytest.
"""

import os
import pytest
from datetime import datetime, timezone
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

# Set test environment before importing app modules
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "postgresql+asyncpg://test:test@localhost:5432/test_pdfrag"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-testing-only"
os.environ["OPENAI_API_KEY"] = "test-openai-key"

from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.database import Base, get_session
from app.db.models import UserModel, DocumentModel
from app.core.auth import hash_password, create_token_pair


# Test database URL (in-memory SQLite for unit tests)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test session."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def async_engine():
    """Create async test database engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest.fixture
async def db_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    async_session_maker = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_maker() as session:
        yield session
        await session.rollback()


@pytest.fixture
def mock_db_session():
    """Create a mock database session for unit tests."""
    session = AsyncMock(spec=AsyncSession)
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.execute = AsyncMock()
    session.get = AsyncMock()
    return session


@pytest.fixture
async def test_client(db_session) -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client with database override."""

    async def override_get_session():
        yield db_session

    app.dependency_overrides[get_session] = override_get_session

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        yield client

    app.dependency_overrides.clear()


@pytest.fixture
def sync_test_client() -> Generator[TestClient, None, None]:
    """Create a synchronous test client for simple endpoint tests."""
    with TestClient(app) as client:
        yield client


# ============ User Fixtures ============

@pytest.fixture
def test_user_data():
    """Test user data."""
    return {
        "email": "test@example.com",
        "password": "testpassword123",
        "display_name": "Test User",
    }


@pytest.fixture
async def test_user(db_session, test_user_data) -> UserModel:
    """Create a test user in the database."""
    user = UserModel(
        id="test-user-id-123",
        email=test_user_data["email"],
        password_hash=hash_password(test_user_data["password"]),
        display_name=test_user_data["display_name"],
        is_active=True,
        is_verified=True,
        created_at=datetime.now(timezone.utc),
    )
    db_session.add(user)
    await db_session.flush()
    return user


@pytest.fixture
def test_user_tokens(test_user) -> dict:
    """Generate tokens for test user."""
    tokens = create_token_pair(test_user.id, test_user.email)
    return {
        "access_token": tokens.access_token,
        "refresh_token": tokens.refresh_token,
    }


@pytest.fixture
def auth_headers(test_user_tokens) -> dict:
    """Authorization headers with test user token."""
    return {"Authorization": f"Bearer {test_user_tokens['access_token']}"}


# ============ Document Fixtures ============

@pytest.fixture
async def test_document(db_session, test_user) -> DocumentModel:
    """Create a test document in the database."""
    doc = DocumentModel(
        id="test-doc-id-123",
        user_id=test_user.id,
        name="test-document.pdf",
        size_bytes=1024,
        mime_type="application/pdf",
        status="ready",
        storage_key="/uploads/test-doc-id-123.pdf",
        checksum_sha256="abc123",
        page_count=5,
        uploaded_at=datetime.now(timezone.utc),
    )
    db_session.add(doc)
    await db_session.flush()
    return doc


@pytest.fixture
def sample_pdf_path(tmp_path):
    """Create a sample PDF file for testing."""
    pdf_path = tmp_path / "sample.pdf"
    # Minimal PDF structure
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Hello World) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000206 00000 n
trailer
<< /Size 5 /Root 1 0 R >>
startxref
300
%%EOF"""
    pdf_path.write_bytes(pdf_content)
    return pdf_path


# ============ Mock Fixtures ============

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for embedding and LLM tests."""
    mock_client = AsyncMock()

    # Mock embedding response
    mock_embedding_response = MagicMock()
    mock_embedding_data = MagicMock()
    mock_embedding_data.embedding = [0.1] * 512
    mock_embedding_data.index = 0
    mock_embedding_response.data = [mock_embedding_data]
    mock_client.embeddings.create = AsyncMock(return_value=mock_embedding_response)

    return mock_client


@pytest.fixture
def mock_settings():
    """Mock application settings."""
    settings = MagicMock()
    settings.openai_api_key = "test-key"
    settings.openai_embedding_model = "text-embedding-3-small"
    settings.embedding_dimensions = 512
    settings.use_matryoshka = True
    settings.use_hybrid_search = True
    settings.bm25_candidate_limit = 100
    settings.bm25_weight = 0.3
    settings.query_cache_enabled = True
    settings.query_cache_ttl_seconds = 3600
    settings.semantic_cache_threshold = 0.95
    settings.min_chunk_size = 100
    settings.max_chunk_size = 1000
    settings.skip_headers_footers = True
    settings.deduplicate_chunks = True
    settings.chunk_size = 512
    settings.chunk_overlap = 50
    settings.similarity_top_k = 5
    settings.llm_provider = "openai"
    settings.embedding_provider = "openai"
    settings.openai_model = "gpt-4o"
    return settings


# ============ Text Block Fixtures ============

@pytest.fixture
def sample_text_blocks():
    """Sample TextBlock objects for chunking tests."""
    from app.core.pdf_parser import TextBlock

    return [
        TextBlock(
            text="This is the main content of the document.",
            page_number=1,
            x0=72.0,
            y0=100.0,
            x1=540.0,
            y1=120.0,
            line_start=1,
            line_end=3,
        ),
        TextBlock(
            text="More content here with important information.",
            page_number=1,
            x0=72.0,
            y0=130.0,
            x1=540.0,
            y1=150.0,
            line_start=4,
            line_end=6,
        ),
        TextBlock(
            text="Page 1",  # Page number - should be filtered
            page_number=1,
            x0=300.0,
            y0=750.0,
            x1=320.0,
            y1=760.0,
            line_start=7,
            line_end=7,
        ),
        TextBlock(
            text="Confidential Document",  # Header - should be filtered
            page_number=1,
            x0=72.0,
            y0=50.0,
            x1=200.0,
            y1=70.0,
            line_start=0,
            line_end=0,
        ),
    ]


@pytest.fixture
def sample_chunks():
    """Sample ChunkData objects for vector store tests."""
    from app.core.vector_store import ChunkData

    return [
        ChunkData(
            text="First chunk of content for testing.",
            document_id="doc-1",
            page_number=1,
            chunk_index=0,
            x0=72.0,
            y0=100.0,
            x1=540.0,
            y1=120.0,
            line_start=1,
            line_end=3,
            embedding=[0.1] * 512,
        ),
        ChunkData(
            text="Second chunk with different content.",
            document_id="doc-1",
            page_number=1,
            chunk_index=1,
            x0=72.0,
            y0=130.0,
            x1=540.0,
            y1=150.0,
            line_start=4,
            line_end=6,
            embedding=[0.2] * 512,
        ),
    ]

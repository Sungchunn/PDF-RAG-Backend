"""
Query cache model for RAG cost optimization.
"""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from app.db.database import Base


class QueryCacheModel(Base):
    """
    Cache for query results with semantic deduplication.

    Enables:
    - Exact hash match for identical queries
    - Semantic similarity match for similar queries
    - TTL-based expiration
    - Per-user and per-document scoping
    """

    __tablename__ = "query_cache"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    # Query identification
    query_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )
    """SHA256 hash of normalized query + document scope"""
    query_embedding: Mapped[Optional[list]] = mapped_column(
        Vector(512),
        nullable=True,
    )
    """Query embedding for semantic deduplication"""
    # Scope
    user_id: Mapped[str] = mapped_column(
        String,
        nullable=False,
        index=True,
    )
    document_ids: Mapped[Optional[list]] = mapped_column(
        ARRAY(String),
        nullable=True,
    )
    """Document IDs this query was scoped to (NULL = all user docs)"""
    # Cached results
    result_chunk_ids: Mapped[list] = mapped_column(
        ARRAY(String),
        nullable=False,
    )
    """Chunk IDs in result order"""
    result_scores: Mapped[list] = mapped_column(
        ARRAY(Float),
        nullable=False,
    )
    """Similarity scores for each chunk"""
    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    hit_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    """Number of cache hits for analytics"""

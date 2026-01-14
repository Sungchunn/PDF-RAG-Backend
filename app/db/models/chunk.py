"""
Document chunk and vector models for RAG.
"""

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

from sqlalchemy import DateTime, Integer, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from app.db.database import Base

if TYPE_CHECKING:
    from app.db.models.document import DocumentModel
    from app.db.models.citation import CitationModel


class DocumentChunkModel(Base):
    """Text chunks extracted from documents with embeddings."""

    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    document_id: Mapped[str] = mapped_column(
        String,
        nullable=False,
        index=True,
    )
    page_number: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    token_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    # Bounding box coordinates
    x0: Mapped[Optional[Decimal]] = mapped_column(
        Numeric,
        nullable=True,
    )
    y0: Mapped[Optional[Decimal]] = mapped_column(
        Numeric,
        nullable=True,
    )
    x1: Mapped[Optional[Decimal]] = mapped_column(
        Numeric,
        nullable=True,
    )
    y1: Mapped[Optional[Decimal]] = mapped_column(
        Numeric,
        nullable=True,
    )
    # Line number references
    line_start: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    line_end: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    # Vector embedding (1536 dims for text-embedding-3-small)
    embedding: Mapped[Optional[list]] = mapped_column(
        Vector(1536),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    # Relationships
    document: Mapped["DocumentModel"] = relationship(
        "DocumentModel",
        back_populates="chunks",
    )
    vectors: Mapped[list["ChunkVectorModel"]] = relationship(
        "ChunkVectorModel",
        back_populates="chunk",
        cascade="all, delete-orphan",
    )
    citations: Mapped[list["CitationModel"]] = relationship(
        "CitationModel",
        back_populates="chunk",
    )


class ChunkVectorModel(Base):
    """Vector store references for chunks.

    Enables tracking of embeddings across multiple vector stores.
    """

    __tablename__ = "chunk_vectors"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    chunk_id: Mapped[str] = mapped_column(
        String,
        nullable=False,
        index=True,
    )
    vector_store: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    vector_id: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    indexed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    # Relationships
    chunk: Mapped["DocumentChunkModel"] = relationship(
        "DocumentChunkModel",
        back_populates="vectors",
    )

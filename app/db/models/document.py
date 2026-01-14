"""
Document-related ORM models.
"""

from datetime import datetime, date
from decimal import Decimal
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

from sqlalchemy import BigInteger, DateTime, Integer, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base

if TYPE_CHECKING:
    from app.db.models.user import UserModel
    from app.db.models.chunk import DocumentChunkModel
    from app.db.models.conversation import ConversationModel


class DocumentModel(Base):
    """Core document metadata table."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    user_id: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        index=True,
    )
    name: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    size_bytes: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )
    mime_type: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String,
        nullable=False,
        index=True,
    )
    page_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    storage_key: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    checksum_sha256: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
    )
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationships
    user: Mapped[Optional["UserModel"]] = relationship(
        "UserModel",
        back_populates="documents",
    )
    metadata_: Mapped[Optional["DocumentMetadataModel"]] = relationship(
        "DocumentMetadataModel",
        back_populates="document",
        uselist=False,
    )
    summary: Mapped[Optional["DocumentSummaryModel"]] = relationship(
        "DocumentSummaryModel",
        back_populates="document",
        uselist=False,
    )
    stats: Mapped[Optional["DocumentStatsModel"]] = relationship(
        "DocumentStatsModel",
        back_populates="document",
        uselist=False,
    )
    pages: Mapped[list["DocumentPageModel"]] = relationship(
        "DocumentPageModel",
        back_populates="document",
        cascade="all, delete-orphan",
    )
    chunks: Mapped[list["DocumentChunkModel"]] = relationship(
        "DocumentChunkModel",
        back_populates="document",
        cascade="all, delete-orphan",
    )
    conversations: Mapped[list["ConversationModel"]] = relationship(
        "ConversationModel",
        back_populates="document",
    )
    tag_links: Mapped[list["DocumentTagLinkModel"]] = relationship(
        "DocumentTagLinkModel",
        back_populates="document",
        cascade="all, delete-orphan",
    )


class DocumentMetadataModel(Base):
    """Extended document metadata."""

    __tablename__ = "document_metadata"

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
    title: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
    )
    author: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
    )
    source_created_at: Mapped[Optional[date]] = mapped_column(
        nullable=True,
    )
    extra_json: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    # Relationships
    document: Mapped["DocumentModel"] = relationship(
        "DocumentModel",
        back_populates="metadata_",
    )


class DocumentSummaryModel(Base):
    """AI-generated document summaries."""

    __tablename__ = "document_summaries"

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
    summary_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    model: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationships
    document: Mapped["DocumentModel"] = relationship(
        "DocumentModel",
        back_populates="summary",
    )


class DocumentStatsModel(Base):
    """Document statistics."""

    __tablename__ = "document_stats"

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
    chunk_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    word_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    message_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    # Relationships
    document: Mapped["DocumentModel"] = relationship(
        "DocumentModel",
        back_populates="stats",
    )


class DocumentPageModel(Base):
    """Page-level information for documents."""

    __tablename__ = "document_pages"

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
    page_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    width_points: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    height_points: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    # Relationships
    document: Mapped["DocumentModel"] = relationship(
        "DocumentModel",
        back_populates="pages",
    )
    blocks: Mapped[list["DocumentPageBlockModel"]] = relationship(
        "DocumentPageBlockModel",
        back_populates="page",
        cascade="all, delete-orphan",
    )


class DocumentPageBlockModel(Base):
    """Text blocks within document pages."""

    __tablename__ = "document_page_blocks"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    page_id: Mapped[str] = mapped_column(
        String,
        nullable=False,
        index=True,
    )
    block_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    block_type: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    x0: Mapped[Decimal] = mapped_column(
        Numeric,
        nullable=False,
    )
    y0: Mapped[Decimal] = mapped_column(
        Numeric,
        nullable=False,
    )
    x1: Mapped[Decimal] = mapped_column(
        Numeric,
        nullable=False,
    )
    y1: Mapped[Decimal] = mapped_column(
        Numeric,
        nullable=False,
    )
    confidence: Mapped[Optional[Decimal]] = mapped_column(
        Numeric,
        nullable=True,
    )
    line_start: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    line_end: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    # Relationships
    page: Mapped["DocumentPageModel"] = relationship(
        "DocumentPageModel",
        back_populates="blocks",
    )


class TagModel(Base):
    """Document tags for categorization."""

    __tablename__ = "tags"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    name: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    # Relationships
    document_links: Mapped[list["DocumentTagLinkModel"]] = relationship(
        "DocumentTagLinkModel",
        back_populates="tag",
    )


class DocumentTagLinkModel(Base):
    """Many-to-many link between documents and tags."""

    __tablename__ = "document_tag_links"

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
    tag_id: Mapped[str] = mapped_column(
        String,
        nullable=False,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    # Relationships
    document: Mapped["DocumentModel"] = relationship(
        "DocumentModel",
        back_populates="tag_links",
    )
    tag: Mapped["TagModel"] = relationship(
        "TagModel",
        back_populates="document_links",
    )

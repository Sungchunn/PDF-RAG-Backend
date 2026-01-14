"""
Citation models for RAG source references.
"""

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

from sqlalchemy import DateTime, Integer, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base

if TYPE_CHECKING:
    from app.db.models.chunk import DocumentChunkModel
    from app.db.models.conversation import MessageModel


class CitationModel(Base):
    """Source citations linking answers to document locations."""

    __tablename__ = "citations"

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
    chunk_id: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
    )
    page_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    context_before: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    context_after: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    # Bounding box coordinates
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
    # Line number references
    line_start: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    line_end: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    confidence: Mapped[Optional[Decimal]] = mapped_column(
        Numeric,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    # Relationships
    chunk: Mapped[Optional["DocumentChunkModel"]] = relationship(
        "DocumentChunkModel",
        back_populates="citations",
    )
    message_links: Mapped[list["MessageCitationModel"]] = relationship(
        "MessageCitationModel",
        back_populates="citation",
    )


class MessageCitationModel(Base):
    """Links messages to citations with display ordering."""

    __tablename__ = "message_citations"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    message_id: Mapped[str] = mapped_column(
        String,
        nullable=False,
        index=True,
    )
    citation_id: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    display_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    # Relationships
    message: Mapped["MessageModel"] = relationship(
        "MessageModel",
        back_populates="citation_links",
    )
    citation: Mapped["CitationModel"] = relationship(
        "CitationModel",
        back_populates="message_links",
    )

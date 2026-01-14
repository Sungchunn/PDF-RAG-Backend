"""
Conversation and message models for chat history.
"""

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

from sqlalchemy import DateTime, Integer, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base

if TYPE_CHECKING:
    from app.db.models.user import UserModel
    from app.db.models.document import DocumentModel
    from app.db.models.citation import MessageCitationModel


class ConversationModel(Base):
    """Chat conversation sessions."""

    __tablename__ = "conversations"

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
    document_id: Mapped[str] = mapped_column(
        String,
        nullable=False,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    last_message_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationships
    user: Mapped[Optional["UserModel"]] = relationship(
        "UserModel",
        back_populates="conversations",
    )
    document: Mapped["DocumentModel"] = relationship(
        "DocumentModel",
        back_populates="conversations",
    )
    messages: Mapped[list["MessageModel"]] = relationship(
        "MessageModel",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="MessageModel.created_at",
    )


class MessageModel(Base):
    """Individual chat messages with metadata."""

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    conversation_id: Mapped[str] = mapped_column(
        String,
        nullable=False,
        index=True,
    )
    document_id: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    role: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    # LLM configuration
    model: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
    )
    temperature: Mapped[Optional[Decimal]] = mapped_column(
        Numeric,
        nullable=True,
    )
    max_tokens: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    # Token usage
    tokens_prompt: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    tokens_completion: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    tokens_total: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    # Latency metrics
    retrieval_latency_ms: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    generation_latency_ms: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    # Relationships
    conversation: Mapped["ConversationModel"] = relationship(
        "ConversationModel",
        back_populates="messages",
    )
    citation_links: Mapped[list["MessageCitationModel"]] = relationship(
        "MessageCitationModel",
        back_populates="message",
        cascade="all, delete-orphan",
    )

"""
User model for authentication and ownership.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base

if TYPE_CHECKING:
    from app.db.models.document import DocumentModel
    from app.db.models.conversation import ConversationModel


class UserModel(Base):
    """User account for authentication and document ownership."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    email: Mapped[str] = mapped_column(
        String,
        unique=True,
        nullable=False,
        index=True,
    )
    password_hash: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    display_name: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )
    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    # Relationships
    documents: Mapped[list["DocumentModel"]] = relationship(
        "DocumentModel",
        back_populates="user",
        lazy="selectin",
    )
    conversations: Mapped[list["ConversationModel"]] = relationship(
        "ConversationModel",
        back_populates="user",
        lazy="selectin",
    )

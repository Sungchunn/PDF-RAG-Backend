"""
Processing job models for async document handling.
"""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class ProcessingJobModel(Base):
    """Async processing job tracking."""

    __tablename__ = "processing_jobs"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    document_id: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        index=True,
    )
    job_type: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String,
        nullable=False,
        index=True,
    )
    status_url: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    error_code: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
    )
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    error_retryable: Mapped[Optional[bool]] = mapped_column(
        Boolean,
        nullable=True,
    )
    progress_percent: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    idempotency_key: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
    )

    # Relationships
    stages: Mapped[list["ProcessingJobStageModel"]] = relationship(
        "ProcessingJobStageModel",
        back_populates="job",
        cascade="all, delete-orphan",
    )
    items: Mapped[list["ProcessingJobItemModel"]] = relationship(
        "ProcessingJobItemModel",
        back_populates="job",
        cascade="all, delete-orphan",
    )


class ProcessingJobStageModel(Base):
    """Individual stages within a processing job."""

    __tablename__ = "processing_job_stages"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    job_id: Mapped[str] = mapped_column(
        String,
        nullable=False,
        index=True,
    )
    stage_name: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    percent_complete: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    # Relationships
    job: Mapped["ProcessingJobModel"] = relationship(
        "ProcessingJobModel",
        back_populates="stages",
    )


class ProcessingJobItemModel(Base):
    """Items processed within a job."""

    __tablename__ = "processing_job_items"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    job_id: Mapped[str] = mapped_column(
        String,
        nullable=False,
        index=True,
    )
    item_type: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    # Relationships
    job: Mapped["ProcessingJobModel"] = relationship(
        "ProcessingJobModel",
        back_populates="items",
    )


class IdempotencyKeyModel(Base):
    """Idempotency keys for API request deduplication."""

    __tablename__ = "idempotency_keys"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    idempotency_key: Mapped[str] = mapped_column(
        String,
        nullable=False,
        index=True,
    )
    scope: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    request_hash: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    response_code: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    response_body: Mapped[Optional[dict]] = mapped_column(
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

"""
Processing service for async document handling with job tracking.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ProcessingJobModel, ProcessingJobStageModel


class JobType(str, Enum):
    """Types of processing jobs."""

    DOCUMENT_UPLOAD = "document_upload"
    REINDEX = "reindex"
    SUMMARY_GENERATION = "summary_generation"


class JobStatus(str, Enum):
    """Job status values."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class StageStatus(str, Enum):
    """Stage status values."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingService:
    """Service for managing async processing jobs."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_job(
        self,
        document_id: str,
        job_type: JobType,
        idempotency_key: Optional[str] = None,
    ) -> ProcessingJobModel:
        """
        Create a new processing job.

        Args:
            document_id: Document being processed
            job_type: Type of job
            idempotency_key: Optional key for deduplication

        Returns:
            Created ProcessingJobModel
        """
        now = datetime.now(timezone.utc)
        job = ProcessingJobModel(
            id=str(uuid4()),
            document_id=document_id,
            job_type=job_type.value,
            status=JobStatus.PENDING.value,
            created_at=now,
            idempotency_key=idempotency_key,
        )
        self.session.add(job)
        await self.session.flush()
        return job

    async def get_job(self, job_id: str) -> Optional[ProcessingJobModel]:
        """Get a job by ID."""
        return await self.session.get(ProcessingJobModel, job_id)

    async def get_jobs_by_document(
        self,
        document_id: str,
        limit: int = 10,
    ) -> list[ProcessingJobModel]:
        """Get jobs for a document."""
        result = await self.session.execute(
            select(ProcessingJobModel)
            .where(ProcessingJobModel.document_id == document_id)
            .order_by(ProcessingJobModel.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        error_message: Optional[str] = None,
        error_code: Optional[str] = None,
        progress_percent: Optional[int] = None,
    ) -> bool:
        """
        Update job status.

        Args:
            job_id: Job ID
            status: New status
            error_message: Error message if failed
            error_code: Error code if failed
            progress_percent: Progress percentage (0-100)

        Returns:
            True if updated successfully
        """
        job = await self.get_job(job_id)
        if not job:
            return False

        now = datetime.now(timezone.utc)
        job.status = status.value

        if status == JobStatus.RUNNING and not job.started_at:
            job.started_at = now

        if status in (JobStatus.COMPLETED, JobStatus.FAILED):
            job.completed_at = now

        if error_message:
            job.error_message = error_message
            job.error_retryable = False  # Could be more sophisticated

        if error_code:
            job.error_code = error_code

        if progress_percent is not None:
            job.progress_percent = progress_percent

        await self.session.flush()
        return True

    async def create_stage(
        self,
        job_id: str,
        stage_name: str,
    ) -> ProcessingJobStageModel:
        """
        Create a processing stage for a job.

        Args:
            job_id: Parent job ID
            stage_name: Name of the stage

        Returns:
            Created ProcessingJobStageModel
        """
        stage = ProcessingJobStageModel(
            id=str(uuid4()),
            job_id=job_id,
            stage_name=stage_name,
            status=StageStatus.PENDING.value,
        )
        self.session.add(stage)
        await self.session.flush()
        return stage

    async def update_stage_status(
        self,
        stage_id: str,
        status: StageStatus,
        percent_complete: Optional[int] = None,
    ) -> bool:
        """
        Update stage status.

        Args:
            stage_id: Stage ID
            status: New status
            percent_complete: Progress percentage

        Returns:
            True if updated successfully
        """
        stage = await self.session.get(ProcessingJobStageModel, stage_id)
        if not stage:
            return False

        now = datetime.now(timezone.utc)
        stage.status = status.value

        if status == StageStatus.RUNNING and not stage.started_at:
            stage.started_at = now

        if status in (StageStatus.COMPLETED, StageStatus.FAILED):
            stage.completed_at = now

        if percent_complete is not None:
            stage.percent_complete = percent_complete

        await self.session.flush()
        return True

    async def get_pending_jobs(
        self,
        job_type: Optional[JobType] = None,
        limit: int = 10,
    ) -> list[ProcessingJobModel]:
        """Get pending jobs for processing."""
        query = select(ProcessingJobModel).where(
            ProcessingJobModel.status == JobStatus.PENDING.value
        )

        if job_type:
            query = query.where(ProcessingJobModel.job_type == job_type.value)

        query = query.order_by(ProcessingJobModel.created_at).limit(limit)

        result = await self.session.execute(query)
        return list(result.scalars().all())

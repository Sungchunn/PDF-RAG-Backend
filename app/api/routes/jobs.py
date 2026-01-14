"""
Job status endpoints for tracking async processing.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.api.deps import SessionDep, CurrentUserDep
from app.db.models import DocumentModel
from app.services.processing_service import ProcessingService


router = APIRouter()


# Response Models
class JobStageResponse(BaseModel):
    """Job stage information."""

    id: str
    stageName: str
    status: str
    percentComplete: Optional[int]


class JobResponse(BaseModel):
    """Job status response."""

    id: str
    documentId: Optional[str]
    jobType: str
    status: str
    progressPercent: Optional[int]
    errorCode: Optional[str]
    errorMessage: Optional[str]
    createdAt: str
    startedAt: Optional[str]
    completedAt: Optional[str]
    stages: list[JobStageResponse]


def format_datetime(dt) -> Optional[str]:
    """Format datetime to ISO string."""
    return dt.isoformat() if dt else None


@router.get("/{job_id}", response_model=JobResponse)
async def get_job_status(
    job_id: str,
    session: SessionDep,
    current_user: CurrentUserDep,
):
    """
    Get status of a processing job.

    Only returns jobs for documents owned by the current user.
    """
    processing = ProcessingService(session)
    job = await processing.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )

    # Verify ownership via document
    if job.document_id:
        doc = await session.get(DocumentModel, job.document_id)
        if not doc or doc.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found",
            )

    # Get stages
    stages = [
        JobStageResponse(
            id=stage.id,
            stageName=stage.stage_name,
            status=stage.status,
            percentComplete=stage.percent_complete,
        )
        for stage in job.stages
    ]

    return JobResponse(
        id=job.id,
        documentId=job.document_id,
        jobType=job.job_type,
        status=job.status,
        progressPercent=job.progress_percent,
        errorCode=job.error_code,
        errorMessage=job.error_message,
        createdAt=format_datetime(job.created_at),
        startedAt=format_datetime(job.started_at),
        completedAt=format_datetime(job.completed_at),
        stages=stages,
    )


@router.get("/document/{document_id}", response_model=list[JobResponse])
async def get_document_jobs(
    document_id: str,
    session: SessionDep,
    current_user: CurrentUserDep,
    limit: int = 10,
):
    """
    Get all jobs for a document.

    Only returns jobs for documents owned by the current user.
    """
    # Verify document ownership
    doc = await session.get(DocumentModel, document_id)
    if not doc or doc.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    processing = ProcessingService(session)
    jobs = await processing.get_jobs_by_document(document_id, limit=limit)

    return [
        JobResponse(
            id=job.id,
            documentId=job.document_id,
            jobType=job.job_type,
            status=job.status,
            progressPercent=job.progress_percent,
            errorCode=job.error_code,
            errorMessage=job.error_message,
            createdAt=format_datetime(job.created_at),
            startedAt=format_datetime(job.started_at),
            completedAt=format_datetime(job.completed_at),
            stages=[
                JobStageResponse(
                    id=stage.id,
                    stageName=stage.stage_name,
                    status=stage.status,
                    percentComplete=stage.percent_complete,
                )
                for stage in job.stages
            ],
        )
        for job in jobs
    ]

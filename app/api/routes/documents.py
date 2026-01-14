"""
Document upload and management endpoints.
"""

import hashlib
import os
import shutil
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import select

from app.api.deps import SessionDep, CurrentUserDep
from app.config import settings
from app.db.models import DocumentModel
from app.services.processing_service import ProcessingService, JobType
from app.workers.document_worker import process_document_task

router = APIRouter()


# Response Models
class DocumentResponse(BaseModel):
    """Document information response."""

    id: str
    name: str
    sizeBytes: int
    mimeType: str
    status: str
    pageCount: Optional[int]
    uploadedAt: str
    deletedAt: Optional[str]


class DocumentUploadResponse(BaseModel):
    """Document upload response."""

    documentId: str
    filename: str
    status: str
    jobId: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Paginated document list response."""

    documents: list[DocumentResponse]
    total: int
    limit: int
    offset: int


def format_datetime(dt) -> Optional[str]:
    """Format datetime to ISO string."""
    return dt.isoformat() if dt else None


def format_document(doc: DocumentModel) -> DocumentResponse:
    """Format document model to response."""
    return DocumentResponse(
        id=doc.id,
        name=doc.name,
        sizeBytes=doc.size_bytes,
        mimeType=doc.mime_type,
        status=doc.status,
        pageCount=doc.page_count,
        uploadedAt=format_datetime(doc.uploaded_at),
        deletedAt=format_datetime(doc.deleted_at),
    )


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    session: SessionDep,
    current_user: CurrentUserDep,
    file: UploadFile = File(...),
):
    """
    Upload a PDF document for processing.

    The document will be:
    1. Saved to the upload directory
    2. Parsed to extract text with coordinates
    3. Chunked and indexed in the vector store

    Processing happens asynchronously. Use the returned jobId to track progress.
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed",
        )

    # Check file size
    file.file.seek(0, 2)  # Seek to end
    size = file.file.tell()
    file.file.seek(0)  # Reset to beginning

    if size > settings.max_file_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum limit of {settings.max_file_size // (1024 * 1024)}MB",
        )

    # Generate unique ID and save file
    document_id = str(uuid4())
    file_path = os.path.join(settings.upload_dir, f"{document_id}.pdf")

    try:
        # Read file content for checksum
        content = file.file.read()
        checksum = hashlib.sha256(content).hexdigest()

        # Save file
        with open(file_path, "wb") as buffer:
            buffer.write(content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}",
        )

    # Create document record
    now = datetime.now(timezone.utc)
    document = DocumentModel(
        id=document_id,
        user_id=current_user.id,
        name=file.filename,
        size_bytes=size,
        mime_type="application/pdf",
        status="processing",
        storage_key=file_path,
        checksum_sha256=checksum,
        uploaded_at=now,
    )
    session.add(document)

    # Create processing job
    processing = ProcessingService(session)
    job = await processing.create_job(document_id, JobType.DOCUMENT_UPLOAD)
    await session.flush()

    # Queue background processing
    background_tasks.add_task(
        process_document_task,
        session,
        job.id,
        document_id,
        file_path,
        current_user.id,
    )

    return DocumentUploadResponse(
        documentId=document_id,
        filename=file.filename,
        status="processing",
        jobId=job.id,
    )


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    session: SessionDep,
    current_user: CurrentUserDep,
    status_filter: Optional[str] = Query(None, alias="status"),
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0),
):
    """
    List user's documents with pagination.

    Filters:
    - status: Filter by document status (processing, ready, error)
    """
    # Build query
    query = (
        select(DocumentModel)
        .where(DocumentModel.user_id == current_user.id)
        .where(DocumentModel.deleted_at.is_(None))
    )

    if status_filter:
        query = query.where(DocumentModel.status == status_filter)

    # Get total count
    count_query = select(DocumentModel.id).where(
        DocumentModel.user_id == current_user.id,
        DocumentModel.deleted_at.is_(None),
    )
    if status_filter:
        count_query = count_query.where(DocumentModel.status == status_filter)

    count_result = await session.execute(count_query)
    total = len(count_result.all())

    # Get paginated results
    query = query.order_by(DocumentModel.uploaded_at.desc())
    query = query.offset(offset).limit(limit)

    result = await session.execute(query)
    documents = result.scalars().all()

    return DocumentListResponse(
        documents=[format_document(doc) for doc in documents],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    session: SessionDep,
    current_user: CurrentUserDep,
):
    """Get a specific document by ID."""
    doc = await session.get(DocumentModel, document_id)

    if not doc or doc.user_id != current_user.id or doc.deleted_at:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return format_document(doc)


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    session: SessionDep,
    current_user: CurrentUserDep,
):
    """
    Soft-delete a document.

    The document record is kept but marked as deleted.
    The physical file remains for potential recovery.
    """
    doc = await session.get(DocumentModel, document_id)

    if not doc or doc.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    if doc.deleted_at:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document already deleted",
        )

    doc.deleted_at = datetime.now(timezone.utc)
    await session.flush()

    return {"status": "deleted", "documentId": document_id}

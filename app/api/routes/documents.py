"""
Document upload and management endpoints.
"""

import os
import shutil
from datetime import datetime
from typing import List
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.config import settings
from app.models.schemas import Document, DocumentUploadResponse

router = APIRouter()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document for processing.
    
    The document will be:
    1. Saved to the upload directory
    2. Parsed to extract text with coordinates
    3. Chunked and indexed in the vector store
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
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}",
        )

    # TODO: Trigger async processing (parsing, embedding, indexing)

    return DocumentUploadResponse(
        documentId=document_id,
        filename=file.filename,
        status="processing",
    )


@router.get("/", response_model=List[Document])
async def list_documents():
    """List all uploaded documents."""
    # TODO: Implement database query
    return []


@router.get("/{document_id}", response_model=Document)
async def get_document(document_id: str):
    """Get a specific document by ID."""
    # TODO: Implement database query
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Document {document_id} not found",
    )


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its associated data."""
    file_path = os.path.join(settings.upload_dir, f"{document_id}.pdf")

    if os.path.exists(file_path):
        os.remove(file_path)

    # TODO: Remove from database and vector store

    return {"status": "deleted", "documentId": document_id}

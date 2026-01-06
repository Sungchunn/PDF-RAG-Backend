"""
Pydantic schemas for API request/response validation.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


# ============ Document Schemas ============

class BoundingBox(BaseModel):
    """Coordinates for text location in PDF."""

    x0: float = Field(..., description="Left x coordinate")
    y0: float = Field(..., description="Top y coordinate")
    x1: float = Field(..., description="Right x coordinate")
    y1: float = Field(..., description="Bottom y coordinate")


class Citation(BaseModel):
    """Citation pointing to exact location in source document."""

    documentId: str = Field(..., description="Source document ID")
    pageNumber: int = Field(..., ge=1, description="Page number (1-indexed)")
    boundingBox: BoundingBox = Field(..., description="Text location coordinates")
    text: str = Field(..., description="Cited text content")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score"
    )


class DocumentStatus(BaseModel):
    """Possible document processing statuses."""

    status: str = Field(..., pattern="^(uploading|processing|ready|error)$")


class Document(BaseModel):
    """Document metadata."""

    id: str = Field(..., description="Unique document ID")
    name: str = Field(..., description="Original filename")
    size: int = Field(..., ge=0, description="File size in bytes")
    uploadedAt: datetime = Field(..., description="Upload timestamp")
    status: str = Field(
        "processing", pattern="^(uploading|processing|ready|error)$"
    )
    pageCount: Optional[int] = Field(None, ge=1, description="Number of pages")
    errorMessage: Optional[str] = Field(None, description="Error message if status is error")


class DocumentUploadResponse(BaseModel):
    """Response after document upload."""

    documentId: str = Field(..., description="Assigned document ID")
    filename: str = Field(..., description="Original filename")
    status: str = Field("processing", description="Initial processing status")


# ============ Chat Schemas ============

class ChatRequest(BaseModel):
    """Chat message request."""

    message: str = Field(..., min_length=1, max_length=5000, description="User message")
    documentId: Optional[str] = Field(None, description="Optional specific document to query")


class ChatResponse(BaseModel):
    """Chat response with answer and citations."""

    answer: str = Field(..., description="Generated answer")
    citations: List[Citation] = Field(default_factory=list, description="Source citations")
    sourceDocuments: List[str] = Field(
        default_factory=list, description="IDs of documents used"
    )


# ============ Chunk Schema for Vector Store ============

class DocumentChunk(BaseModel):
    """A chunk of text extracted from a document with metadata."""

    id: str = Field(..., description="Unique chunk ID")
    documentId: str = Field(..., description="Parent document ID")
    pageNumber: int = Field(..., ge=1, description="Source page number")
    boundingBox: BoundingBox = Field(..., description="Text location")
    text: str = Field(..., description="Chunk text content")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")

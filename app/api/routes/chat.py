"""
Chat and Q&A endpoints for RAG.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.api.deps import SessionDep, CurrentUserDep
from app.config import settings
from app.core.rag_engine import RAGEngine
from app.core.exceptions import RAGQueryError
from app.db.models import DocumentModel


router = APIRouter()


# Request/Response Models
class ChatRequest(BaseModel):
    """Chat request."""

    message: str
    documentId: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None


class BoundingBox(BaseModel):
    """Bounding box coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float


class Citation(BaseModel):
    """Citation with source location."""

    documentId: str
    pageNumber: int
    boundingBox: BoundingBox
    lineStart: Optional[int] = None
    lineEnd: Optional[int] = None
    text: str
    confidence: float


class ChatResponse(BaseModel):
    """Chat response with citations."""

    answer: str
    citations: list[Citation]
    sourceDocuments: list[str]


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    session: SessionDep,
    current_user: CurrentUserDep,
):
    """
    Send a message and get a RAG-powered response with citations.

    The response includes:
    - Generated answer from the LLM
    - Citations with page numbers, bounding boxes, and line numbers
    - List of source document IDs used

    Only queries documents owned by the current user.
    """
    # Validate API key configuration
    llm_provider = (request.provider or settings.llm_provider).strip().lower()
    if llm_provider == "openai" and not settings.openai_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY.",
        )
    if llm_provider == "gemini" and not settings.gemini_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini API key not configured. Please set GEMINI_API_KEY.",
        )

    embed_provider = settings.embedding_provider.strip().lower()
    if embed_provider == "openai" and not settings.openai_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenAI API key not configured for embeddings.",
        )
    if embed_provider == "gemini" and not settings.gemini_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini API key not configured for embeddings.",
        )

    # Validate document ownership if specified
    if request.documentId:
        doc = await session.get(DocumentModel, request.documentId)
        if not doc or doc.user_id != current_user.id or doc.deleted_at:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found",
            )
        if doc.status != "ready":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Document is not ready (status: {doc.status})",
            )

    # Create RAG engine with user isolation
    rag_engine = RAGEngine(session, user_id=current_user.id)

    try:
        response = await rag_engine.query(
            request.message,
            document_id=request.documentId,
            llm_provider=request.provider,
            llm_model=request.model,
        )

        # Convert to API response
        citations = [
            Citation(
                documentId=ctx.document_id,
                pageNumber=ctx.page_number,
                boundingBox=BoundingBox(
                    x0=ctx.bbox_x0,
                    y0=ctx.bbox_y0,
                    x1=ctx.bbox_x1,
                    y1=ctx.bbox_y1,
                ),
                lineStart=ctx.line_start,
                lineEnd=ctx.line_end,
                text=ctx.text[:200] + "..." if len(ctx.text) > 200 else ctx.text,
                confidence=ctx.score,
            )
            for ctx in response.contexts
        ]

        return ChatResponse(
            answer=response.answer,
            citations=citations,
            sourceDocuments=response.source_document_ids,
        )

    except RAGQueryError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.user_message,
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again.",
        )


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    session: SessionDep,
    current_user: CurrentUserDep,
):
    """
    Stream a chat response for real-time updates.

    Uses Server-Sent Events (SSE) for streaming.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Streaming not yet implemented",
    )

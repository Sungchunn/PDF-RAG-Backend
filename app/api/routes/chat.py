"""
Chat and Q&A endpoints for RAG.
"""

from fastapi import APIRouter, HTTPException, status

from app.api.deps import ApiKeyDep
from app.models.schemas import ChatRequest, ChatResponse, Citation

router = APIRouter()


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest, _api_key: ApiKeyDep):
    """
    Send a message and get a RAG-powered response with citations.
    
    The response includes:
    - Generated answer from the LLM
    - Citations with exact page numbers and bounding box coordinates
    - List of source document IDs used
    """
    # TODO: Implement RAG pipeline
    # 1. Retrieve relevant chunks from vector store
    # 2. Construct prompt with context
    # 3. Generate response with LLM
    # 4. Extract and format citations

    # Placeholder response
    return ChatResponse(
        answer="This is a placeholder response. The RAG pipeline is not yet implemented.",
        citations=[],
        sourceDocuments=[],
    )


@router.post("/stream")
async def chat_stream(request: ChatRequest, _api_key: ApiKeyDep):
    """
    Stream a chat response for real-time updates.
    
    Uses Server-Sent Events (SSE) for streaming.
    """
    # TODO: Implement streaming response
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Streaming not yet implemented",
    )

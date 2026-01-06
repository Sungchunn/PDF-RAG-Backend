"""
Chat service for RAG-powered Q&A.
"""

from typing import Optional, List

from app.core.rag_engine import get_rag_engine, RAGResponse
from app.models.schemas import ChatResponse, Citation, BoundingBox


class ChatService:
    """Service for chat and Q&A operations."""

    async def process_message(
        self,
        message: str,
        document_id: Optional[str] = None,
    ) -> ChatResponse:
        """
        Process a chat message and return RAG-powered response.
        
        Args:
            message: User's question
            document_id: Optional specific document to query
            
        Returns:
            ChatResponse with answer and citations
        """
        try:
            rag_engine = get_rag_engine()
            rag_response: RAGResponse = rag_engine.query(message, document_id)

            # Convert contexts to citations
            citations: List[Citation] = []
            for ctx in rag_response.contexts:
                citation = Citation(
                    documentId=ctx.document_id,
                    pageNumber=ctx.page_number,
                    boundingBox=BoundingBox(
                        x0=ctx.bbox_x0,
                        y0=ctx.bbox_y0,
                        x1=ctx.bbox_x1,
                        y1=ctx.bbox_y1,
                    ),
                    text=ctx.text[:200] + "..." if len(ctx.text) > 200 else ctx.text,
                    confidence=ctx.score,
                )
                citations.append(citation)

            return ChatResponse(
                answer=rag_response.answer,
                citations=citations,
                sourceDocuments=rag_response.source_document_ids,
            )

        except Exception as e:
            return ChatResponse(
                answer=f"Error processing your question: {str(e)}",
                citations=[],
                sourceDocuments=[],
            )


# Singleton instance
chat_service = ChatService()

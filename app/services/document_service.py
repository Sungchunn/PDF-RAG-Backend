"""
Document service for handling document operations.
"""

import os
from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from app.config import settings
from app.core.pdf_parser import get_pdf_parser, ParsedDocument
from app.core.rag_engine import RAGEngine
from app.db.database import async_session_maker
from app.models.schemas import Document, DocumentUploadResponse


class DocumentService:
    """Service for document CRUD and processing operations."""

    def __init__(self):
        self._documents: dict[str, Document] = {}  # In-memory store (replace with DB)

    async def upload_document(
        self,
        file_path: str,
        filename: str,
        size: int,
    ) -> DocumentUploadResponse:
        """
        Process an uploaded document.
        
        Steps:
        1. Parse PDF to extract text with coordinates
        2. Index text blocks in vector store
        3. Store document metadata
        """
        document_id = str(uuid4())

        # Store initial document record
        doc = Document(
            id=document_id,
            name=filename,
            size=size,
            uploadedAt=datetime.utcnow(),
            status="processing",
            pageCount=None,
            errorMessage=None,
        )
        self._documents[document_id] = doc

        try:
            # Parse PDF
            parser = get_pdf_parser()
            parsed: ParsedDocument = parser.parse(file_path)

            if parsed.error:
                doc.status = "error"
                doc.errorMessage = parsed.error
                return DocumentUploadResponse(
                    documentId=document_id,
                    filename=filename,
                    status="error",
                )

            doc.pageCount = parsed.page_count

            # Index in RAG engine
            success = False
            async with async_session_maker() as session:
                rag_engine = RAGEngine(session)
                success = await rag_engine.index_document(document_id, parsed.blocks)
                if success:
                    await session.commit()
                else:
                    await session.rollback()

            if success:
                doc.status = "ready"
            else:
                doc.status = "error"
                doc.errorMessage = "Failed to index document"

        except Exception as e:
            doc.status = "error"
            doc.errorMessage = str(e)

        return DocumentUploadResponse(
            documentId=document_id,
            filename=filename,
            status=doc.status,
        )

    async def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self._documents.get(document_id)

    async def list_documents(self) -> List[Document]:
        """List all documents."""
        return list(self._documents.values())

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and its associated data."""
        if document_id not in self._documents:
            return False

        # Remove from RAG engine
        try:
            async with async_session_maker() as session:
                rag_engine = RAGEngine(session)
                removed = await rag_engine.remove_document(document_id)
                if removed:
                    await session.commit()
                else:
                    await session.rollback()
        except Exception:
            pass

        # Remove file
        file_path = os.path.join(settings.upload_dir, f"{document_id}.pdf")
        if os.path.exists(file_path):
            os.remove(file_path)

        # Remove from store
        del self._documents[document_id]
        return True


# Singleton instance
document_service = DocumentService()

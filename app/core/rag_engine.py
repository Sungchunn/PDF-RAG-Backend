"""
RAG engine using LlamaIndex for document indexing and retrieval.
"""

from typing import List, Optional
from dataclasses import dataclass

try:
    from llama_index.core import (
        VectorStoreIndex,
        Document as LlamaDocument,
        Settings,
        StorageContext,
    )
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False

from app.config import settings
from app.core.pdf_parser import TextBlock


@dataclass
class RetrievedContext:
    """Context retrieved from the vector store."""

    text: str
    document_id: str
    page_number: int
    bbox_x0: float
    bbox_y0: float
    bbox_x1: float
    bbox_y1: float
    score: float


@dataclass
class RAGResponse:
    """Response from the RAG pipeline."""

    answer: str
    contexts: List[RetrievedContext]
    source_document_ids: List[str]


class RAGEngine:
    """
    RAG engine for PDF document Q&A using LlamaIndex.
    
    Features:
    - Text chunking with coordinate preservation
    - Vector embedding and storage
    - Semantic retrieval with citations
    - LLM-powered answer generation
    """

    def __init__(self):
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError(
                "LlamaIndex is required for RAG. "
                "Install with: pip install llama-index"
            )

        # Configure LlamaIndex settings
        Settings.chunk_size = settings.chunk_size
        Settings.chunk_overlap = settings.chunk_overlap

        if settings.openai_api_key:
            Settings.llm = OpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
            )
            Settings.embed_model = OpenAIEmbedding(
                model=settings.openai_embedding_model,
                api_key=settings.openai_api_key,
            )

        self._indices: dict[str, VectorStoreIndex] = {}

    def index_document(
        self,
        document_id: str,
        blocks: List[TextBlock],
    ) -> bool:
        """
        Index a document's text blocks for retrieval.
        
        Stores coordinate metadata for precision citations.
        """
        try:
            # Convert blocks to LlamaIndex documents with metadata
            documents = []
            for i, block in enumerate(blocks):
                doc = LlamaDocument(
                    text=block.text,
                    metadata={
                        "document_id": document_id,
                        "page_number": block.page_number,
                        "bbox_x0": block.x0,
                        "bbox_y0": block.y0,
                        "bbox_x1": block.x1,
                        "bbox_y1": block.y1,
                        "block_index": i,
                    },
                )
                documents.append(doc)

            # Create or update index
            if document_id in self._indices:
                # Add to existing index
                for doc in documents:
                    self._indices[document_id].insert(doc)
            else:
                # Create new index
                self._indices[document_id] = VectorStoreIndex.from_documents(
                    documents,
                    show_progress=True,
                )

            return True

        except Exception as e:
            print(f"Error indexing document {document_id}: {e}")
            return False

    def query(
        self,
        question: str,
        document_id: Optional[str] = None,
    ) -> RAGResponse:
        """
        Query the RAG pipeline with a question.
        
        Args:
            question: User's question
            document_id: Optional specific document to query
            
        Returns:
            RAGResponse with answer and citations
        """
        if not self._indices:
            return RAGResponse(
                answer="No documents have been indexed yet.",
                contexts=[],
                source_document_ids=[],
            )

        try:
            # Select index(es) to query
            if document_id and document_id in self._indices:
                indices_to_query = {document_id: self._indices[document_id]}
            else:
                indices_to_query = self._indices

            all_contexts: List[RetrievedContext] = []
            source_doc_ids: set = set()

            # Query each index
            for doc_id, index in indices_to_query.items():
                query_engine = index.as_query_engine(
                    similarity_top_k=settings.similarity_top_k,
                )
                response = query_engine.query(question)

                # Extract retrieved contexts
                for node in response.source_nodes:
                    meta = node.node.metadata
                    context = RetrievedContext(
                        text=node.node.text,
                        document_id=meta.get("document_id", doc_id),
                        page_number=meta.get("page_number", 1),
                        bbox_x0=meta.get("bbox_x0", 0),
                        bbox_y0=meta.get("bbox_y0", 0),
                        bbox_x1=meta.get("bbox_x1", 0),
                        bbox_y1=meta.get("bbox_y1", 0),
                        score=node.score or 0.0,
                    )
                    all_contexts.append(context)
                    source_doc_ids.add(context.document_id)

            # Generate answer using the first index's query engine
            # (In production, you'd want a more sophisticated approach)
            first_index = list(indices_to_query.values())[0]
            query_engine = first_index.as_query_engine()
            response = query_engine.query(question)

            return RAGResponse(
                answer=str(response),
                contexts=all_contexts,
                source_document_ids=list(source_doc_ids),
            )

        except Exception as e:
            return RAGResponse(
                answer=f"Error processing query: {str(e)}",
                contexts=[],
                source_document_ids=[],
            )

    def remove_document(self, document_id: str) -> bool:
        """Remove a document from the index."""
        if document_id in self._indices:
            del self._indices[document_id]
            return True
        return False


# Create engine instance (lazy initialization)
_rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """Get or create the RAG engine instance."""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine

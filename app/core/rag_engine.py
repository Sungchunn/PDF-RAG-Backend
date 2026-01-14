"""
RAG engine using LlamaIndex for document indexing and pgvector for storage.
"""

from dataclasses import dataclass
from typing import List, Optional
import logging

from sqlalchemy.ext.asyncio import AsyncSession

try:
    from llama_index.core import Settings
    from llama_index.core.node_parser import SentenceSplitter

    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False

try:
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from llama_index.embeddings.gemini import GeminiEmbedding
    from llama_index.llms.gemini import Gemini

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from app.config import settings
from app.core.pdf_parser import TextBlock
from app.core.vector_store import PGVectorStore, ChunkData

logger = logging.getLogger(__name__)


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
    line_start: Optional[int] = None
    line_end: Optional[int] = None


@dataclass
class RAGResponse:
    """Response from the RAG pipeline."""

    answer: str
    contexts: List[RetrievedContext]
    source_document_ids: List[str]


class RAGEngine:
    """
    RAG engine for PDF document Q&A using LlamaIndex and pgvector.

    Features:
    - Text chunking with coordinate preservation
    - Persistent vector storage in PostgreSQL
    - User-scoped document retrieval
    - Line number tracking for citations
    - LLM-powered answer generation
    """

    def __init__(
        self,
        session: AsyncSession,
        user_id: Optional[str] = None,
    ):
        """
        Initialize RAG engine.

        Args:
            session: Database session for vector operations
            user_id: Optional user ID for document isolation
        """
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError(
                "LlamaIndex is required for RAG. "
                "Install with: poetry add llama-index"
            )

        self.session = session
        self.user_id = user_id
        self.vector_store = PGVectorStore(session, user_id)

        # Configure LlamaIndex settings
        Settings.chunk_size = settings.chunk_size
        Settings.chunk_overlap = settings.chunk_overlap

        self._embed_model = None
        self._llm = None

    def _get_embed_model(self):
        """Get or configure the embedding model."""
        if self._embed_model is not None:
            return self._embed_model

        provider = settings.embedding_provider.strip().lower()

        if provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError(
                    "OpenAI LlamaIndex packages not installed. "
                    "Install with: poetry add llama-index-embeddings-openai"
                )
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key not configured for embeddings.")
            self._embed_model = OpenAIEmbedding(
                model=settings.openai_embedding_model,
                api_key=settings.openai_api_key,
            )
            return self._embed_model

        if provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError(
                    "Gemini LlamaIndex packages not installed. "
                    "Install with: poetry add llama-index-embeddings-gemini"
                )
            if not settings.gemini_api_key:
                raise ValueError("Gemini API key not configured for embeddings.")
            self._embed_model = GeminiEmbedding(
                model=settings.gemini_embedding_model,
                api_key=settings.gemini_api_key,
            )
            return self._embed_model

        raise ValueError("Unsupported embedding provider. Use 'openai' or 'gemini'.")

    def _get_llm(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Get or configure the LLM."""
        selected_provider = (provider or settings.llm_provider).strip().lower()

        if selected_provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError(
                    "OpenAI LlamaIndex packages not installed. "
                    "Install with: poetry add llama-index-llms-openai"
                )
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key not configured.")
            return OpenAI(
                model=model or settings.openai_model,
                api_key=settings.openai_api_key,
            )

        if selected_provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError(
                    "Gemini LlamaIndex packages not installed. "
                    "Install with: poetry add llama-index-llms-gemini"
                )
            if not settings.gemini_api_key:
                raise ValueError("Gemini API key not configured.")
            return Gemini(
                model=model or settings.gemini_model,
                api_key=settings.gemini_api_key,
            )

        raise ValueError("Unsupported LLM provider. Use 'openai' or 'gemini'.")

    async def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        embed_model = self._get_embed_model()
        embedding = embed_model.get_text_embedding(text)
        return embedding

    async def index_document(
        self,
        document_id: str,
        blocks: List[TextBlock],
    ) -> bool:
        """
        Index a document's text blocks to pgvector.

        Args:
            document_id: Document ID
            blocks: List of parsed text blocks

        Returns:
            True if successful
        """
        try:
            chunks: List[ChunkData] = []

            for i, block in enumerate(blocks):
                # Generate embedding
                embedding = await self._get_embedding(block.text)

                # Create chunk data
                chunk = ChunkData(
                    text=block.text,
                    document_id=document_id,
                    page_number=block.page_number,
                    chunk_index=i,
                    x0=block.x0,
                    y0=block.y0,
                    x1=block.x1,
                    y1=block.y1,
                    line_start=block.line_start,
                    line_end=block.line_end,
                    embedding=embedding,
                    token_count=len(block.text.split()),  # Simple word count
                )
                chunks.append(chunk)

            # Store in pgvector
            await self.vector_store.add_chunks(chunks)
            logger.info(f"Indexed {len(chunks)} chunks for document {document_id}")
            return True

        except Exception as e:
            logger.error(f"Error indexing document {document_id}: {e}")
            return False

    async def query(
        self,
        question: str,
        document_id: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> RAGResponse:
        """
        Query the RAG pipeline with a question.

        Args:
            question: User's question
            document_id: Optional specific document to query
            llm_provider: Optional LLM provider override
            llm_model: Optional LLM model override

        Returns:
            RAGResponse with answer and citations
        """
        try:
            # Generate embedding for question
            query_embedding = await self._get_embedding(question)

            # Retrieve relevant chunks from pgvector
            document_ids = [document_id] if document_id else None
            retrieved = await self.vector_store.query(
                query_embedding,
                document_ids=document_ids,
                top_k=settings.similarity_top_k,
            )

            if not retrieved:
                return RAGResponse(
                    answer="No relevant content found in the documents.",
                    contexts=[],
                    source_document_ids=[],
                )

            # Convert to RetrievedContext
            contexts = [
                RetrievedContext(
                    text=chunk.text,
                    document_id=chunk.document_id,
                    page_number=chunk.page_number,
                    bbox_x0=chunk.x0,
                    bbox_y0=chunk.y0,
                    bbox_x1=chunk.x1,
                    bbox_y1=chunk.y1,
                    score=chunk.score,
                    line_start=chunk.line_start,
                    line_end=chunk.line_end,
                )
                for chunk in retrieved
            ]

            # Build context for LLM
            context_text = "\n\n".join(
                f"[Source: Page {ctx.page_number}, Lines {ctx.line_start}-{ctx.line_end}]\n{ctx.text}"
                for ctx in contexts
            )

            # Generate answer using LLM
            llm = self._get_llm(provider=llm_provider, model=llm_model)
            prompt = f"""Based on the following context from documents, answer the question.

Context:
{context_text}

Question: {question}

Answer:"""

            response = llm.complete(prompt)
            answer = str(response)

            source_doc_ids = list(set(ctx.document_id for ctx in contexts))

            return RAGResponse(
                answer=answer,
                contexts=contexts,
                source_document_ids=source_doc_ids,
            )

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return RAGResponse(
                answer=f"Error processing query: {str(e)}",
                contexts=[],
                source_document_ids=[],
            )

    async def remove_document(self, document_id: str) -> bool:
        """Remove a document's chunks from the vector store."""
        try:
            deleted_count = await self.vector_store.delete_document_chunks(document_id)
            logger.info(f"Deleted {deleted_count} chunks for document {document_id}")
            return deleted_count > 0
        except Exception as e:
            logger.error(f"Error removing document {document_id}: {e}")
            return False


def create_rag_engine(
    session: AsyncSession,
    user_id: Optional[str] = None,
) -> RAGEngine:
    """
    Create a RAG engine instance.

    Args:
        session: Database session
        user_id: Optional user ID for document isolation

    Returns:
        RAGEngine instance
    """
    return RAGEngine(session, user_id)

"""
RAG engine using LlamaIndex for document indexing and pgvector for storage.

Optimizations:
- Hybrid search (BM25 + vector reranking) for reduced cost
- Query result caching with semantic deduplication
- Matryoshka embeddings (512 dimensions) for reduced storage
"""

from dataclasses import dataclass
from typing import List, Optional
import logging

from sqlalchemy import text
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
    from llama_index.llms.gemini import Gemini

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from app.config import settings, get_settings
from app.core.pdf_parser import TextBlock
from app.core.vector_store import PGVectorStore, ChunkData, RetrievedChunk
from app.core.embeddings import EmbeddingService
from app.core.query_cache import QueryCache

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

        # Initialize optimized embedding service
        app_settings = get_settings()
        self.embedding_service = EmbeddingService(
            dimensions=app_settings.embedding_dimensions
        )

        # Initialize query cache
        self.query_cache = QueryCache(session)

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
        """Generate embedding for text using optimized embedding service."""
        # Use the new Matryoshka-aware embedding service
        return await self.embedding_service.get_embedding(text)

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
        use_cache: bool = True,
    ) -> RAGResponse:
        """
        Query the RAG pipeline with a question.

        Optimized pipeline:
        1. Check query cache (exact hash + semantic similarity)
        2. If miss: Run hybrid search (BM25 + vector rerank)
        3. Cache results
        4. Generate LLM response with retrieved context

        Args:
            question: User's question
            document_id: Optional specific document to query
            llm_provider: Optional LLM provider override
            llm_model: Optional LLM model override
            use_cache: Whether to use query caching (default True)

        Returns:
            RAGResponse with answer and citations
        """
        app_settings = get_settings()
        document_ids = [document_id] if document_id else None

        try:
            # Step 1: Check cache (exact match first)
            if use_cache and app_settings.query_cache_enabled:
                cached = await self.query_cache.get(question, self.user_id, document_ids)
                if cached:
                    logger.info(f"Cache hit (exact) for query, hit_count={cached.hit_count}")
                    chunks = await self._fetch_chunks_by_ids(cached.chunk_ids)
                    return await self._generate_response(
                        question, chunks, llm_provider, llm_model
                    )

            # Step 2: Generate query embedding
            query_embedding = await self._get_embedding(question)

            # Step 2b: Check semantic cache (if enabled)
            if use_cache and app_settings.query_cache_enabled:
                cached = await self.query_cache.get_semantic(
                    query_embedding, self.user_id, document_ids
                )
                if cached:
                    logger.info(f"Cache hit (semantic) for query, hit_count={cached.hit_count}")
                    chunks = await self._fetch_chunks_by_ids(cached.chunk_ids)
                    return await self._generate_response(
                        question, chunks, llm_provider, llm_model
                    )

            # Step 3: Hybrid search or pure vector search
            if app_settings.use_hybrid_search:
                retrieved = await self.vector_store.hybrid_query(
                    query_text=question,
                    query_embedding=query_embedding,
                    document_ids=document_ids,
                    top_k=settings.similarity_top_k,
                )
            else:
                # Fall back to pure vector search
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

            # Step 4: Cache results
            if use_cache and app_settings.query_cache_enabled and retrieved:
                await self.query_cache.set(
                    query=question,
                    query_embedding=query_embedding,
                    user_id=self.user_id,
                    document_ids=document_ids,
                    chunk_ids=[c.id for c in retrieved],
                    scores=[c.score for c in retrieved],
                )
                logger.debug(f"Cached query result with {len(retrieved)} chunks")

            # Step 5: Generate response
            return await self._generate_response(
                question, retrieved, llm_provider, llm_model
            )

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return RAGResponse(
                answer=f"Error processing query: {str(e)}",
                contexts=[],
                source_document_ids=[],
            )

    async def _fetch_chunks_by_ids(self, chunk_ids: List[str]) -> List[RetrievedChunk]:
        """Fetch chunks by ID list (for cache hits)."""
        if not chunk_ids:
            return []

        result = await self.session.execute(
            text("""
                SELECT id, document_id, content, page_number,
                       x0, y0, x1, y1, line_start, line_end
                FROM document_chunks
                WHERE id = ANY(:ids)
            """),
            {"ids": chunk_ids},
        )

        # Build map and preserve original order from cache
        chunk_map = {row.id: row for row in result}
        return [
            RetrievedChunk(
                id=chunk_map[cid].id,
                text=chunk_map[cid].content,
                document_id=chunk_map[cid].document_id,
                page_number=chunk_map[cid].page_number,
                x0=float(chunk_map[cid].x0) if chunk_map[cid].x0 else 0.0,
                y0=float(chunk_map[cid].y0) if chunk_map[cid].y0 else 0.0,
                x1=float(chunk_map[cid].x1) if chunk_map[cid].x1 else 0.0,
                y1=float(chunk_map[cid].y1) if chunk_map[cid].y1 else 0.0,
                line_start=chunk_map[cid].line_start,
                line_end=chunk_map[cid].line_end,
                score=1.0,  # Score not stored in cache
            )
            for cid in chunk_ids
            if cid in chunk_map
        ]

    async def _generate_response(
        self,
        question: str,
        retrieved: List[RetrievedChunk],
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> RAGResponse:
        """Generate LLM response from retrieved chunks."""
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

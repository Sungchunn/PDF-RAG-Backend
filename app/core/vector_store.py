"""
PostgreSQL pgvector store integration for document chunk storage and retrieval.

Supports:
- Pure vector search (cosine similarity)
- Hybrid search (BM25 + vector reranking) for cost optimization
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import uuid4
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DocumentChunkModel, ChunkVectorModel
from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ChunkData:
    """Data for a chunk to be stored."""

    text: str
    document_id: str
    page_number: int
    chunk_index: int
    x0: float
    y0: float
    x1: float
    y1: float
    line_start: int
    line_end: int
    embedding: Optional[List[float]] = None
    token_count: Optional[int] = None


@dataclass
class RetrievedChunk:
    """A chunk retrieved from vector search."""

    id: str
    text: str
    document_id: str
    page_number: int
    x0: float
    y0: float
    x1: float
    y1: float
    line_start: Optional[int]
    line_end: Optional[int]
    score: float  # Similarity score


class PGVectorStore:
    """
    Custom vector store using PostgreSQL pgvector.

    Benefits over external vector stores:
    - Uses existing PostgreSQL database
    - Supports user-scoped queries
    - Integrates with SQLAlchemy models
    - Transactional consistency with other data
    """

    def __init__(
        self,
        session: AsyncSession,
        user_id: Optional[str] = None,
    ):
        self.session = session
        self.user_id = user_id

    async def add_chunks(
        self,
        chunks: List[ChunkData],
    ) -> List[str]:
        """
        Store chunks with embeddings in pgvector.

        Args:
            chunks: List of chunk data with embeddings

        Returns:
            List of created chunk IDs
        """
        chunk_ids = []

        for chunk in chunks:
            chunk_id = str(uuid4())
            now = datetime.now(timezone.utc)

            # Create chunk record
            chunk_model = DocumentChunkModel(
                id=chunk_id,
                document_id=chunk.document_id,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                content=chunk.text,
                token_count=chunk.token_count,
                x0=chunk.x0,
                y0=chunk.y0,
                x1=chunk.x1,
                y1=chunk.y1,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                created_at=now,
            )
            self.session.add(chunk_model)

            # Store embedding using raw SQL (pgvector requires special handling)
            if chunk.embedding:
                # Update embedding via raw SQL
                embedding_str = "[" + ",".join(str(x) for x in chunk.embedding) + "]"
                await self.session.execute(
                    text(
                        """
                        UPDATE document_chunks
                        SET embedding = :embedding::vector
                        WHERE id = :chunk_id
                        """
                    ),
                    {"embedding": embedding_str, "chunk_id": chunk_id},
                )

                # Create vector reference record
                vector_model = ChunkVectorModel(
                    id=str(uuid4()),
                    chunk_id=chunk_id,
                    vector_store="pgvector",
                    vector_id=chunk_id,
                    indexed_at=now,
                )
                self.session.add(vector_model)

            chunk_ids.append(chunk_id)

        await self.session.flush()
        return chunk_ids

    async def query(
        self,
        query_embedding: List[float],
        document_ids: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        """
        Query similar chunks using pgvector cosine similarity.

        Args:
            query_embedding: Embedding vector for the query
            document_ids: Optional list of document IDs to filter by
            top_k: Number of results to return

        Returns:
            List of retrieved chunks with similarity scores
        """
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        # Build query with optional filters
        query_parts = [
            """
            SELECT
                dc.id,
                dc.content,
                dc.document_id,
                dc.page_number,
                dc.x0,
                dc.y0,
                dc.x1,
                dc.y1,
                dc.line_start,
                dc.line_end,
                1 - (dc.embedding <=> :query_embedding::vector) as similarity
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE dc.embedding IS NOT NULL
            """
        ]

        params = {"query_embedding": embedding_str, "top_k": top_k}

        # Filter by user if set
        if self.user_id:
            query_parts.append("AND d.user_id = :user_id")
            params["user_id"] = self.user_id

        # Filter by document IDs if specified
        if document_ids:
            query_parts.append("AND dc.document_id = ANY(:doc_ids)")
            params["doc_ids"] = document_ids

        # Exclude deleted documents
        query_parts.append("AND d.deleted_at IS NULL")

        # Order by similarity and limit
        query_parts.append("ORDER BY similarity DESC LIMIT :top_k")

        query_sql = "\n".join(query_parts)

        result = await self.session.execute(text(query_sql), params)
        rows = result.fetchall()

        return [
            RetrievedChunk(
                id=row.id,
                text=row.content,
                document_id=row.document_id,
                page_number=row.page_number,
                x0=float(row.x0) if row.x0 else 0.0,
                y0=float(row.y0) if row.y0 else 0.0,
                x1=float(row.x1) if row.x1 else 0.0,
                y1=float(row.y1) if row.y1 else 0.0,
                line_start=row.line_start,
                line_end=row.line_end,
                score=float(row.similarity),
            )
            for row in rows
        ]

    async def hybrid_query(
        self,
        query_text: str,
        query_embedding: List[float],
        document_ids: Optional[List[str]] = None,
        top_k: int = 5,
        bm25_candidates: Optional[int] = None,
        bm25_weight: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """
        Two-stage hybrid retrieval: BM25 pre-filter → vector rerank.

        This dramatically reduces the number of vectors scanned while
        maintaining high retrieval quality.

        Pipeline:
        1. Use PostgreSQL full-text search (BM25) to get candidate chunks
        2. Compute vector similarity only on candidates
        3. Combine scores using weighted fusion

        Args:
            query_text: Raw query text for BM25
            query_embedding: Query embedding for vector similarity (512d)
            document_ids: Optional document filter
            top_k: Final number of results to return
            bm25_candidates: Number of BM25 candidates (default from settings)
            bm25_weight: Weight for BM25 in hybrid score (default from settings)

        Returns:
            Top-K chunks ranked by hybrid score
        """
        settings = get_settings()
        bm25_candidates = bm25_candidates or settings.bm25_candidate_limit
        bm25_weight = bm25_weight if bm25_weight is not None else settings.bm25_weight

        # Stage 1: BM25 pre-filter
        candidates = await self._bm25_prefilter(
            query_text=query_text,
            document_ids=document_ids,
            limit=bm25_candidates,
        )

        if not candidates:
            logger.debug("BM25 returned no candidates, falling back to pure vector search")
            # Fall back to pure vector search if BM25 returns nothing
            return await self.query(
                query_embedding=query_embedding,
                document_ids=document_ids,
                top_k=top_k,
            )

        logger.debug(f"BM25 pre-filter returned {len(candidates)} candidates")

        # Stage 2: Vector rerank on candidates
        return await self._vector_rerank(
            query_embedding=query_embedding,
            candidate_ids=[c["id"] for c in candidates],
            bm25_scores={c["id"]: c["bm25_score"] for c in candidates},
            top_k=top_k,
            bm25_weight=bm25_weight,
        )

    async def _bm25_prefilter(
        self,
        query_text: str,
        document_ids: Optional[List[str]],
        limit: int,
    ) -> List[Dict]:
        """
        Use PostgreSQL full-text search to get candidate chunks.

        This is much faster than vector similarity for initial filtering.

        Args:
            query_text: Raw query text
            document_ids: Optional document filter
            limit: Maximum candidates to return

        Returns:
            List of {id, bm25_score} dicts
        """
        # Build filters
        filters = ["content_tsv @@ plainto_tsquery('english', :query)"]
        params = {"query": query_text, "limit": limit}

        if self.user_id:
            filters.append("d.user_id = :user_id")
            params["user_id"] = self.user_id

        if document_ids:
            filters.append("dc.document_id = ANY(:doc_ids)")
            params["doc_ids"] = document_ids

        filters.append("d.deleted_at IS NULL")

        where_clause = " AND ".join(filters)

        result = await self.session.execute(
            text(f"""
                SELECT
                    dc.id,
                    ts_rank(dc.content_tsv, plainto_tsquery('english', :query)) as bm25_score
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE {where_clause}
                ORDER BY bm25_score DESC
                LIMIT :limit
            """),
            params,
        )

        return [{"id": row.id, "bm25_score": float(row.bm25_score)} for row in result]

    async def _vector_rerank(
        self,
        query_embedding: List[float],
        candidate_ids: List[str],
        bm25_scores: Dict[str, float],
        top_k: int,
        bm25_weight: float,
    ) -> List[RetrievedChunk]:
        """
        Rerank candidates using vector similarity + BM25 hybrid score.

        Only computes vector similarity on pre-filtered candidates,
        not the entire database.

        Args:
            query_embedding: Query embedding
            candidate_ids: Chunk IDs from BM25
            bm25_scores: BM25 scores per chunk ID
            top_k: Number of results to return
            bm25_weight: Weight for BM25 in final score (0-1)

        Returns:
            Top-K chunks by hybrid score
        """
        if not candidate_ids:
            return []

        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        # Try 512d embedding first, fall back to 1536d
        settings = get_settings()
        if settings.use_matryoshka and settings.embedding_dimensions == 512:
            embedding_column = "embedding_512"
        else:
            embedding_column = "embedding"

        # Compute vector similarity only on candidates
        result = await self.session.execute(
            text(f"""
                SELECT
                    id,
                    document_id,
                    content,
                    page_number,
                    x0, y0, x1, y1,
                    line_start,
                    line_end,
                    1 - ({embedding_column} <=> :embedding::vector) as vector_score
                FROM document_chunks
                WHERE id = ANY(:ids)
                  AND {embedding_column} IS NOT NULL
            """),
            {
                "embedding": embedding_str,
                "ids": candidate_ids,
            },
        )

        # Normalize BM25 scores to 0-1 range for fair combination
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
        if max_bm25 == 0:
            max_bm25 = 1.0

        # Compute hybrid scores and build chunks
        chunks = []
        for row in result:
            bm25_score = bm25_scores.get(row.id, 0) / max_bm25  # Normalize
            vector_score = float(row.vector_score)

            # Weighted combination
            hybrid_score = (1 - bm25_weight) * vector_score + bm25_weight * bm25_score

            chunks.append(
                RetrievedChunk(
                    id=row.id,
                    text=row.content,
                    document_id=row.document_id,
                    page_number=row.page_number,
                    x0=float(row.x0) if row.x0 else 0.0,
                    y0=float(row.y0) if row.y0 else 0.0,
                    x1=float(row.x1) if row.x1 else 0.0,
                    y1=float(row.y1) if row.y1 else 0.0,
                    line_start=row.line_start,
                    line_end=row.line_end,
                    score=hybrid_score,
                )
            )

        # Sort by hybrid score and return top-K
        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks[:top_k]

    async def delete_document_chunks(self, document_id: str) -> int:
        """
        Delete all chunks for a document.

        Args:
            document_id: Document ID to delete chunks for

        Returns:
            Number of chunks deleted
        """
        result = await self.session.execute(
            text(
                """
                DELETE FROM document_chunks
                WHERE document_id = :document_id
                RETURNING id
                """
            ),
            {"document_id": document_id},
        )
        deleted = result.fetchall()
        return len(deleted)

    async def get_chunk_count(self, document_id: str) -> int:
        """Get the number of chunks for a document."""
        result = await self.session.execute(
            text(
                """
                SELECT COUNT(*) as count
                FROM document_chunks
                WHERE document_id = :document_id
                """
            ),
            {"document_id": document_id},
        )
        row = result.fetchone()
        return row.count if row else 0

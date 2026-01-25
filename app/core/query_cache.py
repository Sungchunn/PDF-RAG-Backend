"""
Query result caching with semantic deduplication.

Features:
1. Exact match caching via query hash
2. Semantic similarity caching (similar questions = same answer)
3. TTL-based expiration
4. Cache invalidation on document updates
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy import delete, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models.cache import QueryCacheModel

logger = logging.getLogger(__name__)


@dataclass
class CachedResult:
    """Cached query result."""

    chunk_ids: List[str]
    scores: List[float]
    created_at: datetime
    hit_count: int


class QueryCache:
    """
    Query result cache with exact and semantic matching.

    Reduces costs by caching retrieval results:
    - Exact hash match: Same query text = cached result
    - Semantic match: Similar queries (via embedding) = cached result

    Usage:
        cache = QueryCache(session)

        # Check cache
        result = await cache.get(query, user_id, document_ids)
        if result:
            return result.chunk_ids  # Cache hit!

        # ... perform actual search ...

        # Store result
        await cache.set(query, embedding, user_id, document_ids, chunk_ids, scores)
    """

    def __init__(
        self,
        session: AsyncSession,
        ttl_seconds: Optional[int] = None,
        semantic_threshold: Optional[float] = None,
    ):
        """
        Initialize query cache.

        Args:
            session: Database session
            ttl_seconds: Cache TTL (default from settings)
            semantic_threshold: Similarity threshold for semantic matching
        """
        settings = get_settings()
        self.session = session
        self.ttl_seconds = ttl_seconds or settings.query_cache_ttl_seconds
        self.semantic_threshold = (
            semantic_threshold or settings.semantic_cache_threshold
        )

    async def get(
        self,
        query: str,
        user_id: str,
        document_ids: Optional[List[str]] = None,
    ) -> Optional[CachedResult]:
        """
        Check cache for query result (exact hash match).

        Args:
            query: User's query text
            user_id: User ID for scoping
            document_ids: Optional document filter

        Returns:
            CachedResult if found, None otherwise
        """
        query_hash = self._compute_query_hash(query, document_ids)
        now = datetime.now(timezone.utc)

        # Exact hash match
        result = await self.session.execute(
            select(QueryCacheModel).where(
                QueryCacheModel.query_hash == query_hash,
                QueryCacheModel.user_id == user_id,
                QueryCacheModel.expires_at > now,
            )
        )
        cached = result.scalar_one_or_none()

        if cached:
            # Update hit count
            await self.session.execute(
                update(QueryCacheModel)
                .where(QueryCacheModel.id == cached.id)
                .values(hit_count=QueryCacheModel.hit_count + 1)
            )
            logger.debug(f"Cache hit (exact) for query hash {query_hash[:8]}...")

            return CachedResult(
                chunk_ids=cached.result_chunk_ids,
                scores=cached.result_scores,
                created_at=cached.created_at,
                hit_count=cached.hit_count + 1,
            )

        return None

    async def get_semantic(
        self,
        query_embedding: List[float],
        user_id: str,
        document_ids: Optional[List[str]] = None,
    ) -> Optional[CachedResult]:
        """
        Check cache using semantic similarity.

        More expensive than hash lookup, but catches similar queries.
        Only called after hash lookup misses.

        Args:
            query_embedding: Embedding of user's query (512d)
            user_id: User ID for scoping
            document_ids: Optional document filter

        Returns:
            CachedResult if similar query found, None otherwise
        """
        now = datetime.now(timezone.utc)
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        # Build document filter
        doc_filter = ""
        params = {
            "embedding": embedding_str,
            "user_id": user_id,
            "now": now,
            "threshold": self.semantic_threshold,
        }

        if document_ids:
            # Filter by document scope (array contains)
            doc_filter = "AND document_ids @> :doc_ids"
            params["doc_ids"] = document_ids

        # Use pgvector to find similar cached queries
        result = await self.session.execute(
            text(f"""
                SELECT
                    id,
                    result_chunk_ids,
                    result_scores,
                    created_at,
                    hit_count,
                    1 - (query_embedding <=> :embedding::vector) as similarity
                FROM query_cache
                WHERE user_id = :user_id
                  AND expires_at > :now
                  AND query_embedding IS NOT NULL
                  AND 1 - (query_embedding <=> :embedding::vector) >= :threshold
                  {doc_filter}
                ORDER BY similarity DESC
                LIMIT 1
            """),
            params,
        )
        row = result.fetchone()

        if row:
            # Update hit count
            await self.session.execute(
                text("""
                    UPDATE query_cache
                    SET hit_count = hit_count + 1
                    WHERE id = :id
                """),
                {"id": row.id},
            )
            logger.debug(
                f"Cache hit (semantic) with similarity {row.similarity:.3f}"
            )

            return CachedResult(
                chunk_ids=row.result_chunk_ids,
                scores=row.result_scores,
                created_at=row.created_at,
                hit_count=row.hit_count + 1,
            )

        return None

    async def set(
        self,
        query: str,
        query_embedding: List[float],
        user_id: str,
        document_ids: Optional[List[str]],
        chunk_ids: List[str],
        scores: List[float],
    ) -> str:
        """
        Store query result in cache.

        Args:
            query: User's query text
            query_embedding: Query embedding for semantic matching
            user_id: User ID
            document_ids: Document scope (None = all user docs)
            chunk_ids: Retrieved chunk IDs (in order)
            scores: Similarity scores

        Returns:
            Cache entry ID
        """
        query_hash = self._compute_query_hash(query, document_ids)
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=self.ttl_seconds)

        # Check if entry already exists (race condition handling)
        existing = await self.session.execute(
            select(QueryCacheModel.id).where(
                QueryCacheModel.query_hash == query_hash,
                QueryCacheModel.user_id == user_id,
            )
        )
        if existing.scalar_one_or_none():
            logger.debug(f"Cache entry already exists for hash {query_hash[:8]}...")
            return query_hash

        cache_entry = QueryCacheModel(
            query_hash=query_hash,
            query_embedding=query_embedding,
            user_id=user_id,
            document_ids=document_ids,
            result_chunk_ids=chunk_ids,
            result_scores=scores,
            created_at=now,
            expires_at=expires_at,
        )

        self.session.add(cache_entry)
        await self.session.flush()

        logger.debug(
            f"Cached query result: {len(chunk_ids)} chunks, "
            f"expires in {self.ttl_seconds}s"
        )

        return cache_entry.id

    async def invalidate_for_document(self, document_id: str) -> int:
        """
        Invalidate cache entries containing a document.

        Call this when a document is updated or deleted.

        Args:
            document_id: Document ID to invalidate

        Returns:
            Number of entries invalidated
        """
        # Delete cache entries where document_ids contains this document
        # or where document_ids is NULL (all docs) for the user
        result = await self.session.execute(
            delete(QueryCacheModel).where(
                QueryCacheModel.document_ids.contains([document_id])
            )
        )

        if result.rowcount > 0:
            logger.info(
                f"Invalidated {result.rowcount} cache entries for document {document_id}"
            )

        return result.rowcount

    async def invalidate_for_user(self, user_id: str) -> int:
        """
        Invalidate all cache entries for a user.

        Args:
            user_id: User ID to invalidate

        Returns:
            Number of entries invalidated
        """
        result = await self.session.execute(
            delete(QueryCacheModel).where(QueryCacheModel.user_id == user_id)
        )

        if result.rowcount > 0:
            logger.info(f"Invalidated {result.rowcount} cache entries for user {user_id}")

        return result.rowcount

    async def cleanup_expired(self) -> int:
        """
        Remove expired cache entries.

        Should be called periodically (e.g., hourly cron job).

        Returns:
            Number of entries removed
        """
        now = datetime.now(timezone.utc)
        result = await self.session.execute(
            delete(QueryCacheModel).where(QueryCacheModel.expires_at <= now)
        )

        if result.rowcount > 0:
            logger.info(f"Cleaned up {result.rowcount} expired cache entries")

        return result.rowcount

    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent hashing."""
        # Lowercase, strip, normalize whitespace
        return " ".join(query.lower().split())

    def _compute_query_hash(
        self,
        query: str,
        document_ids: Optional[List[str]],
    ) -> str:
        """
        Compute hash for query + document scope.

        Same query on different documents = different hash.
        """
        normalized = self._normalize_query(query)
        scope = ",".join(sorted(document_ids)) if document_ids else "__all__"
        combined = f"{normalized}|{scope}"
        return hashlib.sha256(combined.encode()).hexdigest()

    async def get_stats(self, user_id: str) -> dict:
        """
        Get cache statistics for a user.

        Returns:
            Dict with total_entries, total_hits, avg_hit_count
        """
        result = await self.session.execute(
            text("""
                SELECT
                    COUNT(*) as total_entries,
                    COALESCE(SUM(hit_count), 0) as total_hits,
                    COALESCE(AVG(hit_count), 0) as avg_hit_count
                FROM query_cache
                WHERE user_id = :user_id
                  AND expires_at > :now
            """),
            {"user_id": user_id, "now": datetime.now(timezone.utc)},
        )
        row = result.fetchone()

        return {
            "total_entries": row.total_entries,
            "total_hits": row.total_hits,
            "avg_hit_count": float(row.avg_hit_count),
        }

# RAG Cost Optimization Implementation

**Date:** 25/01/26
**Status:** Proposed
**Author:** Claude

---

## 1. Overview

### 1.1 Problem Statement

The current PDF-RAG system faces three cost scalability challenges at high user volumes:

| Cost Driver | Current State | Impact |
|-------------|---------------|--------|
| **Too many vectors** | Every text block = 1 vector (~500 per 100-page PDF) | High embedding API costs |
| **Vectors too large** | 1536 dimensions (OpenAI default) | High storage costs |
| **Pure vector search** | Full scan on every request, no caching | High latency and compute |

### 1.2 Solution Summary

Implement a **hybrid retrieval pipeline** combining:
- BM25 keyword pre-filtering before vector search
- Intelligent chunking with deduplication
- Matryoshka embeddings at 512 dimensions
- Query result caching with semantic deduplication

### 1.3 Expected Outcomes

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Vectors per 100-page doc | ~500 | ~200 | 60% reduction |
| Embedding dimensions | 1536 | 512 | 66% reduction |
| Vectors scanned per query | All | ~100 | 95%+ reduction |
| **Combined embedding API cost** | 100% | ~13% | **87% savings** |

---

## 2. Architecture

### 2.1 Current Retrieval Flow

```
User Query
    │
    ▼
Embed Query (1536 dimensions)
    │
    ▼
Full pgvector Scan (cosine similarity on ALL chunks)
    │
    ▼
Top-K Results
    │
    ▼
LLM Response Generation
```

### 2.2 Proposed Retrieval Flow

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│         Query Cache Check           │
│  (exact hash + semantic similarity) │
└─────────────────────────────────────┘
    │
    ├── Cache HIT ──────────────────────────────┐
    │                                           │
    ▼ Cache MISS                                │
┌─────────────────────────────────────┐         │
│       BM25 Pre-filter (FTS)         │         │
│  PostgreSQL full-text search        │         │
│  Returns top 100 candidates         │         │
└─────────────────────────────────────┘         │
    │                                           │
    ▼                                           │
┌─────────────────────────────────────┐         │
│    Embed Query (512 dimensions)     │         │
│    Matryoshka embedding             │         │
└─────────────────────────────────────┘         │
    │                                           │
    ▼                                           │
┌─────────────────────────────────────┐         │
│      Vector Rerank on Candidates    │         │
│  Cosine similarity on 100 chunks    │         │
│  (not full database scan)           │         │
└─────────────────────────────────────┘         │
    │                                           │
    ▼                                           │
┌─────────────────────────────────────┐         │
│         Cache Result                │         │
│    Store with TTL (1 hour)          │         │
└─────────────────────────────────────┘         │
    │                                           │
    ◄───────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│     LLM Response Generation         │
│  Context from retrieved chunks      │
└─────────────────────────────────────┘
```

### 2.3 Document Processing Flow (Updated)

```
PDF Upload
    │
    ▼
┌─────────────────────────────────────┐
│       Parse PDF (PyMuPDF)           │
│  Extract text blocks with bbox      │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│      Block Classification           │  ◄── NEW
│  header | footer | body | toc       │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│      Smart Chunking                 │  ◄── NEW
│  - Skip headers/footers             │
│  - Merge small adjacent blocks      │
│  - Deduplicate by content hash      │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│   Generate 512d Embeddings          │  ◄── NEW
│   Matryoshka truncation             │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│      Store in pgvector              │
│  - embedding_512 column             │
│  - content_tsv for FTS (auto)       │
└─────────────────────────────────────┘
```

---

## 3. Database Schema Changes

### 3.1 Migration: Add FTS and Reduced Embedding Columns

**File:** `app/db/migrations/xxx_add_hybrid_search_columns.py`

```sql
-- Step 1: Add full-text search column (auto-populated)
ALTER TABLE document_chunks
ADD COLUMN content_tsv tsvector
GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

-- Step 2: Create GIN index for fast text search
-- Use CONCURRENTLY to avoid locking production table
CREATE INDEX CONCURRENTLY idx_chunks_content_fts
ON document_chunks USING GIN(content_tsv);

-- Step 3: Add reduced-dimension embedding column
ALTER TABLE document_chunks
ADD COLUMN embedding_512 vector(512);

-- Step 4: Create IVFFlat index for 512d vectors
CREATE INDEX idx_chunks_embedding_512
ON document_chunks
USING ivfflat (embedding_512 vector_cosine_ops)
WITH (lists = 100);
```

### 3.2 Migration: Add Chunk Metadata Columns

**File:** `app/db/migrations/xxx_add_chunk_metadata.py`

```sql
-- Chunk type classification
ALTER TABLE document_chunks
ADD COLUMN chunk_type VARCHAR(20) DEFAULT 'text';

-- Hierarchical chunking support
ALTER TABLE document_chunks
ADD COLUMN parent_chunk_id VARCHAR;

-- Merge tracking
ALTER TABLE document_chunks
ADD COLUMN is_merged BOOLEAN DEFAULT FALSE;

-- Content deduplication
ALTER TABLE document_chunks
ADD COLUMN content_hash VARCHAR(64);

-- Index for deduplication lookups
CREATE INDEX idx_chunks_content_hash ON document_chunks(content_hash);
```

### 3.3 Migration: Create Query Cache Table

**File:** `app/db/migrations/xxx_create_query_cache.py`

```sql
CREATE TABLE query_cache (
    id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::varchar,

    -- Query identification
    query_hash VARCHAR(64) NOT NULL,           -- SHA256 of normalized query
    query_embedding vector(512),               -- For semantic deduplication

    -- Scope
    user_id VARCHAR NOT NULL,
    document_ids VARCHAR[],                    -- NULL = all user docs

    -- Cached results
    result_chunk_ids VARCHAR[] NOT NULL,
    result_scores FLOAT[] NOT NULL,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    hit_count INTEGER DEFAULT 0,

    -- Constraints
    CONSTRAINT query_cache_expires_future CHECK (expires_at > created_at)
);

-- Fast lookup by query hash
CREATE INDEX idx_query_cache_hash ON query_cache(query_hash, user_id);

-- Cleanup expired entries
CREATE INDEX idx_query_cache_expires ON query_cache(expires_at);

-- Semantic similarity search for cache deduplication
CREATE INDEX idx_query_cache_embedding
ON query_cache
USING ivfflat (query_embedding vector_cosine_ops)
WITH (lists = 50);
```

### 3.4 Entity Relationship Diagram (Updated)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        document_chunks                               │
├─────────────────────────────────────────────────────────────────────┤
│ id                 VARCHAR PRIMARY KEY                              │
│ document_id        VARCHAR NOT NULL                                 │
│ user_id            VARCHAR NOT NULL                                 │
│ content            TEXT NOT NULL                                    │
│ embedding          vector(1536)        -- Original (backward compat) │
│ embedding_512      vector(512)         -- NEW: Reduced dimension     │
│ content_tsv        tsvector GENERATED  -- NEW: Full-text search      │
│ chunk_type         VARCHAR(20)         -- NEW: header/footer/body    │
│ parent_chunk_id    VARCHAR             -- NEW: Hierarchical ref      │
│ is_merged          BOOLEAN             -- NEW: Merge flag            │
│ content_hash       VARCHAR(64)         -- NEW: Deduplication         │
│ page_number        INTEGER                                          │
│ chunk_index        INTEGER                                          │
│ x0, y0, x1, y1     FLOAT               -- Bounding box              │
│ line_start         INTEGER                                          │
│ line_end           INTEGER                                          │
│ token_count        INTEGER                                          │
│ created_at         TIMESTAMP                                        │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ user_id, document_ids[]
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         query_cache                                  │
├─────────────────────────────────────────────────────────────────────┤
│ id                 VARCHAR PRIMARY KEY                              │
│ query_hash         VARCHAR(64)         -- Normalized query hash      │
│ query_embedding    vector(512)         -- Semantic deduplication     │
│ user_id            VARCHAR                                          │
│ document_ids       VARCHAR[]           -- Scope filter               │
│ result_chunk_ids   VARCHAR[]           -- Cached chunk IDs           │
│ result_scores      FLOAT[]             -- Cached similarity scores   │
│ created_at         TIMESTAMP                                        │
│ expires_at         TIMESTAMP           -- TTL expiration             │
│ hit_count          INTEGER             -- Usage tracking             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Implementation Details

### 4.1 Configuration Updates

**File:** `app/config.py`

```python
class Settings(BaseSettings):
    # ... existing settings ...

    # ============================================
    # EMBEDDING CONFIGURATION
    # ============================================
    embedding_dimensions: int = 512
    """
    Matryoshka embedding dimension.
    Options: 256, 512, 1024, 1536
    Lower = faster search + less storage, slightly lower quality.
    512 recommended for cost/quality balance.
    """

    use_matryoshka: bool = True
    """Enable Matryoshka dimension reduction."""

    # ============================================
    # HYBRID SEARCH CONFIGURATION
    # ============================================
    use_hybrid_search: bool = True
    """Enable BM25 + vector hybrid search."""

    bm25_candidate_limit: int = 100
    """
    Number of candidates to retrieve via BM25 before vector reranking.
    Higher = better recall, slower search.
    100-200 recommended.
    """

    bm25_weight: float = 0.3
    """
    Weight for BM25 score in hybrid ranking (0.0 to 1.0).
    0.0 = pure vector, 1.0 = pure BM25.
    0.3 recommended for semantic-heavy queries.
    """

    # ============================================
    # QUERY CACHE CONFIGURATION
    # ============================================
    query_cache_enabled: bool = True
    """Enable query result caching."""

    query_cache_ttl_seconds: int = 3600
    """Cache TTL in seconds. Default: 1 hour."""

    semantic_cache_threshold: float = 0.95
    """
    Cosine similarity threshold for semantic cache matching.
    If a cached query embedding is >= this similar, return cached result.
    0.95 = very similar queries only.
    """

    # ============================================
    # SMART CHUNKING CONFIGURATION
    # ============================================
    min_chunk_size: int = 100
    """Minimum characters per chunk. Smaller blocks get merged."""

    max_chunk_size: int = 1000
    """Maximum characters per chunk. Larger blocks get split."""

    skip_headers_footers: bool = True
    """Skip header/footer blocks during chunking."""

    deduplicate_chunks: bool = True
    """Skip chunks with duplicate content hash."""
```

### 4.2 Embedding Service

**File:** `app/core/embeddings.py` (NEW)

```python
"""
Embedding service with Matryoshka dimension support.

Matryoshka embeddings allow truncating full-dimension vectors to smaller
dimensions with minimal quality loss. OpenAI's text-embedding-3-small
supports: 256, 512, 1024, 1536 dimensions.
"""

import hashlib
from typing import List, Optional

from openai import AsyncOpenAI

from app.config import get_settings


class EmbeddingService:
    """Generate embeddings with configurable dimensions."""

    def __init__(self, dimensions: Optional[int] = None):
        settings = get_settings()
        self.dimensions = dimensions or settings.embedding_dimensions
        self.model = "text-embedding-3-small"
        self._client: Optional[AsyncOpenAI] = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI()
        return self._client

    async def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding with Matryoshka dimension reduction.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector of configured dimension
        """
        response = await self.client.embeddings.create(
            input=text,
            model=self.model,
            dimensions=self.dimensions,  # Matryoshka truncation
        )
        return response.data[0].embedding

    async def get_embeddings_batch(
        self, texts: List[str], batch_size: int = 100
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Max texts per API call

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = await self.client.embeddings.create(
                input=batch,
                model=self.model,
                dimensions=self.dimensions,
            )
            embeddings.extend([item.embedding for item in response.data])
        return embeddings

    @staticmethod
    def truncate_embedding(
        embedding: List[float], target_dim: int
    ) -> List[float]:
        """
        Truncate existing embedding to smaller dimension.

        Useful for migrating existing 1536d embeddings to 512d.
        Matryoshka embeddings preserve quality when truncated.

        Args:
            embedding: Full-dimension embedding
            target_dim: Target dimension (must be <= current)

        Returns:
            Truncated embedding
        """
        if len(embedding) < target_dim:
            raise ValueError(
                f"Cannot truncate {len(embedding)}d to {target_dim}d"
            )
        return embedding[:target_dim]

    @staticmethod
    def compute_text_hash(text: str) -> str:
        """Compute SHA256 hash for content deduplication."""
        normalized = text.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()
```

### 4.3 Smart Chunking

**File:** `app/core/chunking.py` (NEW)

```python
"""
Intelligent document chunking with deduplication and block merging.

Key optimizations:
1. Skip low-value content (headers, footers, page numbers)
2. Merge small adjacent blocks to reduce vector count
3. Deduplicate identical content across documents
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set

from app.core.embeddings import EmbeddingService
from app.core.pdf_parser import TextBlock


@dataclass
class ChunkMetadata:
    """Metadata for a processed chunk."""
    chunk_type: str  # 'text', 'header', 'footer', 'merged'
    parent_id: Optional[str] = None
    content_hash: str = ""
    is_duplicate: bool = False
    merged_from: List[str] = field(default_factory=list)


@dataclass
class ChunkData:
    """Processed chunk ready for embedding and storage."""
    text: str
    page_number: int
    x0: float
    y0: float
    x1: float
    y1: float
    line_start: int
    line_end: int
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None


class SmartChunker:
    """
    Intelligent document chunker that reduces vector count while
    preserving retrieval quality.
    """

    # Patterns indicating low-value content
    PAGE_NUMBER_PATTERN = re.compile(r'^[\s\-–—]*\d+[\s\-–—]*$')
    HEADER_KEYWORDS = {'confidential', 'draft', 'page', 'copyright'}

    def __init__(
        self,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        skip_headers_footers: bool = True,
        deduplicate: bool = True,
    ):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.skip_headers_footers = skip_headers_footers
        self.deduplicate = deduplicate
        self._seen_hashes: Set[str] = set()

    def chunk_document(
        self,
        blocks: List[TextBlock],
        page_height: float = 792,  # Default letter size
    ) -> List[ChunkData]:
        """
        Process document blocks into optimized chunks.

        Args:
            blocks: Raw text blocks from PDF parser
            page_height: Page height for position-based classification

        Returns:
            List of processed chunks (deduplicated, merged)
        """
        self._seen_hashes.clear()

        # Step 1: Classify blocks
        classified = self._classify_blocks(blocks, page_height)

        # Step 2: Filter low-value content
        if self.skip_headers_footers:
            filtered = [
                c for c in classified
                if c.metadata.chunk_type not in ('header', 'footer', 'page_number')
            ]
        else:
            filtered = classified

        # Step 3: Merge small adjacent blocks
        merged = self._merge_small_blocks(filtered)

        # Step 4: Deduplicate
        if self.deduplicate:
            return self._deduplicate(merged)

        return merged

    def _classify_blocks(
        self,
        blocks: List[TextBlock],
        page_height: float,
    ) -> List[ChunkData]:
        """Classify each block by type based on position and content."""
        chunks = []

        for block in blocks:
            chunk_type = self._classify_block_type(block, page_height)
            content_hash = EmbeddingService.compute_text_hash(block.text)

            chunks.append(ChunkData(
                text=block.text,
                page_number=block.page_number,
                x0=block.x0,
                y0=block.y0,
                x1=block.x1,
                y1=block.y1,
                line_start=block.line_start,
                line_end=block.line_end,
                metadata=ChunkMetadata(
                    chunk_type=chunk_type,
                    content_hash=content_hash,
                ),
            ))

        return chunks

    def _classify_block_type(
        self,
        block: TextBlock,
        page_height: float,
    ) -> str:
        """
        Classify block as header/footer/body based on position and content.

        Heuristics:
        - Top 10% of page = likely header
        - Bottom 10% of page = likely footer
        - Single number = page number
        - Contains header keywords = header
        """
        text_lower = block.text.strip().lower()

        # Page numbers
        if self.PAGE_NUMBER_PATTERN.match(block.text.strip()):
            return 'page_number'

        # Position-based classification
        header_threshold = page_height * 0.1
        footer_threshold = page_height * 0.9

        if block.y0 < header_threshold:
            # Check for header keywords
            if any(kw in text_lower for kw in self.HEADER_KEYWORDS):
                return 'header'

        if block.y1 > footer_threshold:
            return 'footer'

        return 'text'

    def _merge_small_blocks(
        self,
        chunks: List[ChunkData],
    ) -> List[ChunkData]:
        """
        Merge adjacent small blocks on the same page.

        This reduces vector count while preserving context.
        """
        if not chunks:
            return []

        merged = []
        current = chunks[0]
        merged_ids = []

        for next_chunk in chunks[1:]:
            can_merge = (
                len(current.text) < self.min_chunk_size
                and current.page_number == next_chunk.page_number
                and len(current.text) + len(next_chunk.text) <= self.max_chunk_size
            )

            if can_merge:
                # Merge chunks
                merged_ids.append(str(id(current)))
                current = self._merge_two_chunks(current, next_chunk)
            else:
                # Finalize current chunk
                if merged_ids:
                    current.metadata.chunk_type = 'merged'
                    current.metadata.merged_from = merged_ids
                    merged_ids = []
                merged.append(current)
                current = next_chunk

        # Don't forget last chunk
        if merged_ids:
            current.metadata.chunk_type = 'merged'
            current.metadata.merged_from = merged_ids
        merged.append(current)

        return merged

    def _merge_two_chunks(
        self,
        a: ChunkData,
        b: ChunkData,
    ) -> ChunkData:
        """Merge two adjacent chunks into one."""
        return ChunkData(
            text=f"{a.text}\n\n{b.text}",
            page_number=a.page_number,
            x0=min(a.x0, b.x0),
            y0=min(a.y0, b.y0),
            x1=max(a.x1, b.x1),
            y1=max(a.y1, b.y1),
            line_start=a.line_start,
            line_end=b.line_end,
            metadata=ChunkMetadata(
                chunk_type='merged',
                content_hash=EmbeddingService.compute_text_hash(
                    f"{a.text}\n\n{b.text}"
                ),
            ),
        )

    def _deduplicate(
        self,
        chunks: List[ChunkData],
    ) -> List[ChunkData]:
        """Remove chunks with duplicate content hash."""
        result = []

        for chunk in chunks:
            if chunk.metadata.content_hash in self._seen_hashes:
                chunk.metadata.is_duplicate = True
            else:
                self._seen_hashes.add(chunk.metadata.content_hash)
                result.append(chunk)

        return result
```

### 4.4 Query Cache

**File:** `app/core/query_cache.py` (NEW)

```python
"""
Query result caching with semantic deduplication.

Features:
1. Exact match caching via query hash
2. Semantic similarity caching (similar questions = same answer)
3. TTL-based expiration
4. Cache invalidation on document updates
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy import delete, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models.cache import QueryCacheModel


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

    Usage:
        cache = QueryCache(session)

        # Check cache
        result = await cache.get(query, user_id, document_ids)
        if result:
            return result.chunk_ids  # Cache hit

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
        Check cache for query result.

        Checks:
        1. Exact query hash match
        2. Semantic similarity match (if threshold met)

        Args:
            query: User's query text
            user_id: User ID for scoping
            document_ids: Optional document filter

        Returns:
            CachedResult if found, None otherwise
        """
        query_hash = self._compute_query_hash(query, document_ids)
        now = datetime.now(timezone.utc)

        # Step 1: Exact hash match
        result = await self.session.execute(
            select(QueryCacheModel)
            .where(
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
            return CachedResult(
                chunk_ids=cached.result_chunk_ids,
                scores=cached.result_scores,
                created_at=cached.created_at,
                hit_count=cached.hit_count + 1,
            )

        # Step 2: Semantic similarity match (optional, more expensive)
        # This requires the query embedding, which we may not have yet
        # Skip for now - can be enabled if embedding is pre-computed

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

        Args:
            query_embedding: Embedding of user's query
            user_id: User ID for scoping
            document_ids: Optional document filter

        Returns:
            CachedResult if similar query found, None otherwise
        """
        now = datetime.now(timezone.utc)

        # Use pgvector to find similar cached queries
        result = await self.session.execute(
            text("""
                SELECT
                    id,
                    result_chunk_ids,
                    result_scores,
                    created_at,
                    hit_count,
                    1 - (query_embedding <=> :embedding) as similarity
                FROM query_cache
                WHERE user_id = :user_id
                  AND expires_at > :now
                  AND query_embedding IS NOT NULL
                  AND 1 - (query_embedding <=> :embedding) >= :threshold
                ORDER BY similarity DESC
                LIMIT 1
            """),
            {
                "embedding": str(query_embedding),
                "user_id": user_id,
                "now": now,
                "threshold": self.semantic_threshold,
            },
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
            document_ids: Document scope
            chunk_ids: Retrieved chunk IDs
            scores: Similarity scores

        Returns:
            Cache entry ID
        """
        query_hash = self._compute_query_hash(query, document_ids)
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=self.ttl_seconds)

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
        result = await self.session.execute(
            delete(QueryCacheModel).where(
                QueryCacheModel.document_ids.contains([document_id])
            )
        )
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
        return result.rowcount

    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent hashing."""
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
        scope = ",".join(sorted(document_ids)) if document_ids else ""
        combined = f"{normalized}|{scope}"
        return hashlib.sha256(combined.encode()).hexdigest()
```

### 4.5 Hybrid Search in Vector Store

**File:** `app/core/vector_store.py` (MODIFY)

Add the following methods to the existing `PGVectorStore` class:

```python
# Add to existing PGVectorStore class

async def hybrid_query(
    self,
    query_text: str,
    query_embedding: List[float],
    user_id: str,
    document_ids: Optional[List[str]] = None,
    top_k: int = 5,
    bm25_candidates: int = 100,
    bm25_weight: float = 0.3,
) -> List[RetrievedChunk]:
    """
    Two-stage hybrid retrieval: BM25 pre-filter → vector rerank.

    This dramatically reduces the number of vectors scanned while
    maintaining high retrieval quality.

    Args:
        query_text: Raw query text for BM25
        query_embedding: Query embedding for vector similarity
        user_id: User ID for isolation
        document_ids: Optional document filter
        top_k: Final number of results to return
        bm25_candidates: Number of BM25 candidates for reranking
        bm25_weight: Weight for BM25 score in final ranking (0-1)

    Returns:
        Top-K chunks ranked by hybrid score
    """
    # Stage 1: BM25 pre-filter
    candidates = await self._bm25_prefilter(
        query_text=query_text,
        user_id=user_id,
        document_ids=document_ids,
        limit=bm25_candidates,
    )

    if not candidates:
        # Fall back to pure vector search if BM25 returns nothing
        return await self.query(
            embedding=query_embedding,
            user_id=user_id,
            document_ids=document_ids,
            top_k=top_k,
        )

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
    user_id: str,
    document_ids: Optional[List[str]],
    limit: int,
) -> List[dict]:
    """
    Use PostgreSQL full-text search to get candidate chunks.

    This is much faster than vector similarity for initial filtering.

    Returns:
        List of {id, bm25_score} dicts
    """
    # Build document filter clause
    doc_filter = ""
    params = {
        "query": query_text,
        "user_id": user_id,
        "limit": limit,
    }

    if document_ids:
        doc_filter = "AND document_id = ANY(:doc_ids)"
        params["doc_ids"] = document_ids

    result = await self.session.execute(
        text(f"""
            SELECT
                id,
                ts_rank(content_tsv, plainto_tsquery('english', :query)) as bm25_score
            FROM document_chunks
            WHERE content_tsv @@ plainto_tsquery('english', :query)
              AND user_id = :user_id
              {doc_filter}
            ORDER BY bm25_score DESC
            LIMIT :limit
        """),
        params,
    )

    return [{"id": row.id, "bm25_score": row.bm25_score} for row in result]

async def _vector_rerank(
    self,
    query_embedding: List[float],
    candidate_ids: List[str],
    bm25_scores: dict[str, float],
    top_k: int,
    bm25_weight: float,
) -> List[RetrievedChunk]:
    """
    Rerank candidates using vector similarity + BM25 hybrid score.

    Only computes vector similarity on pre-filtered candidates,
    not the entire database.

    Returns:
        Top-K chunks by hybrid score
    """
    if not candidate_ids:
        return []

    # Compute vector similarity only on candidates
    result = await self.session.execute(
        text("""
            SELECT
                id,
                document_id,
                content,
                page_number,
                x0, y0, x1, y1,
                line_start,
                line_end,
                1 - (embedding_512 <=> :embedding) as vector_score
            FROM document_chunks
            WHERE id = ANY(:ids)
        """),
        {
            "embedding": str(query_embedding),
            "ids": candidate_ids,
        },
    )

    # Compute hybrid scores and rank
    chunks = []
    for row in result:
        bm25_score = bm25_scores.get(row.id, 0)
        vector_score = row.vector_score

        # Reciprocal Rank Fusion style hybrid score
        hybrid_score = (
            (1 - bm25_weight) * vector_score +
            bm25_weight * bm25_score
        )

        chunks.append(RetrievedChunk(
            id=row.id,
            document_id=row.document_id,
            content=row.content,
            page_number=row.page_number,
            x0=row.x0,
            y0=row.y0,
            x1=row.x1,
            y1=row.y1,
            line_start=row.line_start,
            line_end=row.line_end,
            score=hybrid_score,
        ))

    # Sort by hybrid score and return top-K
    chunks.sort(key=lambda c: c.score, reverse=True)
    return chunks[:top_k]
```

### 4.6 RAG Engine Integration

**File:** `app/core/rag_engine.py` (MODIFY)

Update the `query` method to use all optimizations:

```python
# Updated query method in RAGEngine class

async def query(
    self,
    question: str,
    document_id: Optional[str] = None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    use_cache: bool = True,
) -> RAGResponse:
    """
    Execute optimized RAG query.

    Pipeline:
    1. Check query cache (exact hash + semantic similarity)
    2. If miss: Run hybrid search (BM25 + vector rerank)
    3. Cache results
    4. Generate LLM response with retrieved context

    Args:
        question: User's question
        document_id: Optional single document filter
        llm_provider: LLM provider override
        llm_model: LLM model override
        use_cache: Whether to use query caching

    Returns:
        RAGResponse with answer and citations
    """
    settings = get_settings()
    document_ids = [document_id] if document_id else None

    # Initialize services
    query_cache = QueryCache(self.session)
    embedding_service = EmbeddingService(dimensions=settings.embedding_dimensions)

    # Step 1: Check cache (exact match first)
    if use_cache and settings.query_cache_enabled:
        cached = await query_cache.get(question, self.user_id, document_ids)
        if cached:
            logger.info(f"Cache hit (exact) for query, hit_count={cached.hit_count}")
            chunks = await self._fetch_chunks_by_ids(cached.chunk_ids)
            return await self._generate_response(question, chunks, llm_provider, llm_model)

    # Step 2: Generate query embedding
    query_embedding = await embedding_service.get_embedding(question)

    # Step 2b: Check semantic cache (if enabled)
    if use_cache and settings.query_cache_enabled:
        cached = await query_cache.get_semantic(query_embedding, self.user_id, document_ids)
        if cached:
            logger.info(f"Cache hit (semantic) for query, hit_count={cached.hit_count}")
            chunks = await self._fetch_chunks_by_ids(cached.chunk_ids)
            return await self._generate_response(question, chunks, llm_provider, llm_model)

    # Step 3: Hybrid search
    if settings.use_hybrid_search:
        retrieved = await self.vector_store.hybrid_query(
            query_text=question,
            query_embedding=query_embedding,
            user_id=self.user_id,
            document_ids=document_ids,
            top_k=settings.similarity_top_k,
            bm25_candidates=settings.bm25_candidate_limit,
            bm25_weight=settings.bm25_weight,
        )
    else:
        # Fall back to pure vector search
        retrieved = await self.vector_store.query(
            embedding=query_embedding,
            user_id=self.user_id,
            document_ids=document_ids,
            top_k=settings.similarity_top_k,
        )

    # Step 4: Cache results
    if use_cache and settings.query_cache_enabled and retrieved:
        await query_cache.set(
            query=question,
            query_embedding=query_embedding,
            user_id=self.user_id,
            document_ids=document_ids,
            chunk_ids=[c.id for c in retrieved],
            scores=[c.score for c in retrieved],
        )
        logger.info(f"Cached query result with {len(retrieved)} chunks")

    # Step 5: Generate response
    return await self._generate_response(question, retrieved, llm_provider, llm_model)

async def _fetch_chunks_by_ids(self, chunk_ids: List[str]) -> List[RetrievedChunk]:
    """Fetch chunks by ID list (for cache hits)."""
    result = await self.session.execute(
        text("""
            SELECT id, document_id, content, page_number,
                   x0, y0, x1, y1, line_start, line_end
            FROM document_chunks
            WHERE id = ANY(:ids)
        """),
        {"ids": chunk_ids},
    )

    # Preserve original order from cache
    chunk_map = {row.id: row for row in result}
    return [
        RetrievedChunk(
            id=chunk_map[cid].id,
            document_id=chunk_map[cid].document_id,
            content=chunk_map[cid].content,
            page_number=chunk_map[cid].page_number,
            x0=chunk_map[cid].x0,
            y0=chunk_map[cid].y0,
            x1=chunk_map[cid].x1,
            y1=chunk_map[cid].y1,
            line_start=chunk_map[cid].line_start,
            line_end=chunk_map[cid].line_end,
            score=1.0,  # Score not stored in cache
        )
        for cid in chunk_ids
        if cid in chunk_map
    ]
```

### 4.7 Document Worker Updates

**File:** `app/workers/document_worker.py` (MODIFY)

Update the processing pipeline to use smart chunking:

```python
# Updated indexing stage in document worker

async def _index_document(
    self,
    job_id: str,
    document_id: str,
    parsed: ParsedDocument,
) -> bool:
    """
    Index document with smart chunking and reduced embeddings.

    Optimizations applied:
    1. Block classification (skip headers/footers)
    2. Small block merging
    3. Content deduplication
    4. 512-dimension Matryoshka embeddings
    """
    settings = get_settings()

    # Initialize services
    chunker = SmartChunker(
        min_chunk_size=settings.min_chunk_size,
        max_chunk_size=settings.max_chunk_size,
        skip_headers_footers=settings.skip_headers_footers,
        deduplicate=settings.deduplicate_chunks,
    )
    embedding_service = EmbeddingService(dimensions=settings.embedding_dimensions)

    # Step 1: Smart chunking
    chunks = chunker.chunk_document(
        blocks=parsed.blocks,
        page_height=parsed.page_height,
    )

    # Log reduction metrics
    reduction_pct = (1 - len(chunks) / len(parsed.blocks)) * 100 if parsed.blocks else 0
    logger.info(
        f"Smart chunking: {len(parsed.blocks)} blocks → {len(chunks)} chunks "
        f"({reduction_pct:.1f}% reduction)"
    )

    # Step 2: Generate embeddings (batch for efficiency)
    texts = [c.text for c in chunks]
    embeddings = await embedding_service.get_embeddings_batch(texts)

    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding

    # Step 3: Store in database
    for chunk in chunks:
        chunk_model = DocumentChunkModel(
            document_id=document_id,
            user_id=self.user_id,
            content=chunk.text,
            embedding_512=chunk.embedding,  # Store 512d embedding
            page_number=chunk.page_number,
            x0=chunk.x0,
            y0=chunk.y0,
            x1=chunk.x1,
            y1=chunk.y1,
            line_start=chunk.line_start,
            line_end=chunk.line_end,
            chunk_type=chunk.metadata.chunk_type,
            is_merged=chunk.metadata.chunk_type == 'merged',
            content_hash=chunk.metadata.content_hash,
        )
        self.session.add(chunk_model)

    await self.session.flush()

    # Step 4: Invalidate any cached queries for this document
    cache = QueryCache(self.session)
    invalidated = await cache.invalidate_for_document(document_id)
    if invalidated:
        logger.info(f"Invalidated {invalidated} cached queries for document {document_id}")

    return True
```

---

## 5. Data Migration

### 5.1 Backfill Existing 512d Embeddings

**File:** `scripts/backfill_embeddings.py`

```python
"""
Backfill 512d embeddings from existing 1536d embeddings.

Matryoshka embeddings can be truncated without re-computing.
This script truncates existing embeddings to 512 dimensions.

Usage:
    python scripts/backfill_embeddings.py --batch-size 1000
"""

import asyncio
import argparse

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.config import get_settings


async def backfill_embeddings(batch_size: int = 1000):
    """Truncate 1536d embeddings to 512d in batches."""
    settings = get_settings()
    engine = create_async_engine(settings.database_url)
    async_session = sessionmaker(engine, class_=AsyncSession)

    async with async_session() as session:
        # Count total chunks needing backfill
        result = await session.execute(text("""
            SELECT COUNT(*) FROM document_chunks
            WHERE embedding IS NOT NULL AND embedding_512 IS NULL
        """))
        total = result.scalar()
        print(f"Total chunks to backfill: {total}")

        processed = 0
        while processed < total:
            # Backfill batch
            result = await session.execute(text("""
                UPDATE document_chunks
                SET embedding_512 = embedding[1:512]::vector(512)
                WHERE id IN (
                    SELECT id FROM document_chunks
                    WHERE embedding IS NOT NULL AND embedding_512 IS NULL
                    LIMIT :batch_size
                )
            """), {"batch_size": batch_size})

            await session.commit()
            processed += result.rowcount
            print(f"Backfilled {processed}/{total} chunks")

    print("Backfill complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()
    asyncio.run(backfill_embeddings(args.batch_size))
```

### 5.2 FTS Index Population

The FTS column uses `GENERATED ALWAYS`, so it auto-populates when the column is added. No manual backfill needed.

---

## 6. Rollout Plan

### Phase 1: Database Schema (Day 1)
1. Deploy migration for FTS column and index
2. Deploy migration for `embedding_512` column
3. Deploy migration for chunk metadata columns
4. Deploy migration for query cache table

### Phase 2: Read Path (Day 2-3)
1. Deploy hybrid search code with feature flag disabled
2. Enable `use_hybrid_search=true` in staging
3. Monitor latency and retrieval quality
4. Enable in production

### Phase 3: Write Path (Day 4-5)
1. Deploy smart chunking for new documents
2. Monitor vector count reduction metrics
3. Existing documents unchanged (backward compatible)

### Phase 4: Embedding Migration (Day 6-7)
1. Run backfill script for existing embeddings
2. Switch primary embedding column to `embedding_512`
3. Monitor storage reduction

### Phase 5: Caching (Day 8-10)
1. Deploy query cache
2. Tune TTL and semantic threshold
3. Monitor cache hit rates
4. Set up periodic cleanup job for expired entries

---

## 7. Verification

### 7.1 Unit Tests

```python
# tests/test_smart_chunking.py
async def test_header_footer_classification():
    """Blocks near page edges should be classified as headers/footers."""

async def test_small_block_merging():
    """Adjacent blocks smaller than min_chunk_size should merge."""

async def test_content_deduplication():
    """Duplicate content should be skipped."""

# tests/test_hybrid_search.py
async def test_bm25_prefilter():
    """BM25 should return relevant candidates."""

async def test_vector_rerank():
    """Vector reranking should improve relevance."""

async def test_hybrid_scoring():
    """Hybrid score should combine BM25 and vector scores."""

# tests/test_query_cache.py
async def test_exact_cache_hit():
    """Same query should return cached result."""

async def test_semantic_cache_hit():
    """Similar query should return cached result."""

async def test_cache_invalidation():
    """Document update should invalidate related cache."""
```

### 7.2 Integration Tests

```python
# tests/integration/test_rag_optimization.py
async def test_full_pipeline_with_caching():
    """Upload doc, query 3x, verify cache hits."""

async def test_chunk_reduction():
    """Upload 100-page PDF, verify 60% vector reduction."""

async def test_hybrid_vs_pure_vector():
    """Compare retrieval quality of hybrid vs pure vector."""
```

### 7.3 Manual Testing Checklist

- [ ] Upload a 100-page PDF
- [ ] Check `document_chunks` count (should be ~40% of original blocks)
- [ ] Verify `embedding_512` column is populated
- [ ] Verify `content_tsv` column is populated
- [ ] Run same query 3 times, observe latency decrease
- [ ] Check `query_cache` table for entries
- [ ] Delete document, verify cache invalidation
- [ ] Verify citations still have correct page/bbox

### 7.4 Metrics to Monitor

| Metric | Source | Target |
|--------|--------|--------|
| Vectors per document | `document_chunks` count | 60% reduction |
| Query latency (p50) | API response time | 5x improvement |
| Query latency (p99) | API response time | 3x improvement |
| Cache hit rate | `query_cache.hit_count` | 30-50% |
| Embedding API calls | OpenAI billing | 87% reduction |
| Storage per document | Database size | 87% reduction |

---

## 8. Rollback Plan

### If Hybrid Search Degrades Quality
1. Set `use_hybrid_search=false` in config
2. System falls back to pure vector search
3. No data migration needed

### If Smart Chunking Issues
1. New documents use original chunking
2. Existing documents unaffected
3. Re-process affected documents if needed

### If Cache Causes Stale Results
1. Set `query_cache_enabled=false`
2. Truncate `query_cache` table
3. All queries go to live search

---

## 9. Future Enhancements

### 9.1 Late Interaction Models (ColBERT)
- Store token-level embeddings instead of chunk-level
- MaxSim scoring for better semantic matching
- Higher storage but better quality

### 9.2 Hierarchical Chunking
- Document summary → Section summaries → Paragraphs
- Query routes to appropriate level
- Better for long documents

### 9.3 Adaptive Retrieval
- Simple questions → fewer chunks
- Complex questions → more chunks
- Query classification model

### 9.4 Compression
- Binary quantization for embeddings
- 32x storage reduction
- ~5% quality loss

-- Add hybrid search support for RAG cost optimization.
-- Includes: FTS column, 512d embedding, smart chunking metadata, query cache.

-- ============================================
-- STEP 1: Add FTS column for BM25 search
-- ============================================

-- Add tsvector column (auto-populated from content)
ALTER TABLE document_chunks
ADD COLUMN content_tsv tsvector
GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

-- Create GIN index for fast full-text search
CREATE INDEX idx_chunks_content_fts
ON document_chunks USING GIN(content_tsv);

-- ============================================
-- STEP 2: Add reduced-dimension embedding column
-- ============================================

-- Add 512d embedding column for Matryoshka embeddings
ALTER TABLE document_chunks
ADD COLUMN embedding_512 vector(512);

-- Create IVFFlat index for efficient vector search
CREATE INDEX idx_chunks_embedding_512
ON document_chunks
USING ivfflat (embedding_512 vector_cosine_ops)
WITH (lists = 100);

-- ============================================
-- STEP 3: Add smart chunking metadata columns
-- ============================================

-- Chunk type classification
ALTER TABLE document_chunks
ADD COLUMN chunk_type VARCHAR(20) DEFAULT 'text';

-- Hierarchical chunking support
ALTER TABLE document_chunks
ADD COLUMN parent_chunk_id VARCHAR;

-- Merge tracking
ALTER TABLE document_chunks
ADD COLUMN is_merged BOOLEAN DEFAULT FALSE;

-- Content deduplication hash
ALTER TABLE document_chunks
ADD COLUMN content_hash VARCHAR(64);

-- Index for deduplication lookups
CREATE INDEX idx_chunks_content_hash
ON document_chunks(content_hash);

-- ============================================
-- STEP 4: Create query cache table
-- ============================================

CREATE TABLE query_cache (
    id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::varchar,

    -- Query identification
    query_hash VARCHAR(64) NOT NULL,
    query_embedding vector(512),

    -- Scope
    user_id VARCHAR NOT NULL,
    document_ids VARCHAR[],

    -- Cached results
    result_chunk_ids VARCHAR[] NOT NULL,
    result_scores FLOAT[] NOT NULL,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 0,

    -- Constraints
    CONSTRAINT query_cache_expires_future CHECK (expires_at > created_at)
);

-- Fast lookup by query hash
CREATE INDEX idx_query_cache_hash
ON query_cache(query_hash, user_id);

-- Cleanup expired entries
CREATE INDEX idx_query_cache_expires
ON query_cache(expires_at);

-- Semantic similarity search for cache deduplication
CREATE INDEX idx_query_cache_embedding
ON query_cache
USING ivfflat (query_embedding vector_cosine_ops)
WITH (lists = 50);

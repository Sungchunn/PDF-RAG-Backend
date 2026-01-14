-- Add embedding column to document_chunks for pgvector storage.
-- Uses 1536 dimensions for OpenAI text-embedding-3-small model.

-- Enable pgvector extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS vector;

-- Add embedding column
ALTER TABLE document_chunks ADD COLUMN embedding vector(1536);

-- Create IVFFlat index for efficient cosine similarity search
-- Lists = 100 is suitable for up to ~1M vectors
CREATE INDEX idx_document_chunks_embedding ON document_chunks
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

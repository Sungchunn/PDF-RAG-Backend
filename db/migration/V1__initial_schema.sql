-- Initial schema for PDF-RAG backend.
-- Relationships are application-enforced (no FK constraints).
-- documents.id -> document_metadata.document_id, document_summaries.document_id,
-- document_stats.document_id, document_tag_links.document_id,
-- processing_jobs.document_id, document_pages.document_id,
-- document_chunks.document_id, citations.document_id,
-- conversations.document_id, messages.document_id
-- document_pages.id -> document_page_blocks.page_id
-- document_chunks.id -> chunk_vectors.chunk_id, citations.chunk_id (optional)
-- conversations.id -> messages.conversation_id
-- messages.id -> message_citations.message_id
-- citations.id -> message_citations.citation_id
-- processing_jobs.id -> processing_job_stages.job_id, processing_job_items.job_id

CREATE TABLE documents (
  id text PRIMARY KEY,
  name text NOT NULL,
  size_bytes bigint NOT NULL,
  mime_type text NOT NULL,
  status text NOT NULL,
  page_count integer,
  storage_key text NOT NULL,
  checksum_sha256 text,
  uploaded_at timestamptz NOT NULL,
  deleted_at timestamptz
);

CREATE TABLE document_metadata (
  id text PRIMARY KEY,
  document_id text NOT NULL,
  title text,
  author text,
  source_created_at date,
  extra_json jsonb,
  updated_at timestamptz NOT NULL
);

CREATE TABLE document_summaries (
  id text PRIMARY KEY,
  document_id text NOT NULL,
  summary_text text NOT NULL,
  model text,
  created_at timestamptz NOT NULL,
  updated_at timestamptz
);

CREATE TABLE document_stats (
  id text PRIMARY KEY,
  document_id text NOT NULL,
  chunk_count integer,
  word_count integer,
  message_count integer,
  updated_at timestamptz NOT NULL
);

CREATE TABLE tags (
  id text PRIMARY KEY,
  name text NOT NULL,
  created_at timestamptz NOT NULL
);

CREATE TABLE document_tag_links (
  id text PRIMARY KEY,
  document_id text NOT NULL,
  tag_id text NOT NULL,
  created_at timestamptz NOT NULL
);

CREATE TABLE processing_jobs (
  id text PRIMARY KEY,
  document_id text,
  job_type text NOT NULL,
  status text NOT NULL,
  status_url text,
  created_at timestamptz NOT NULL,
  started_at timestamptz,
  completed_at timestamptz,
  error_code text,
  error_message text,
  error_retryable boolean,
  progress_percent integer,
  idempotency_key text
);

CREATE TABLE processing_job_stages (
  id text PRIMARY KEY,
  job_id text NOT NULL,
  stage_name text NOT NULL,
  status text NOT NULL,
  started_at timestamptz,
  completed_at timestamptz,
  percent_complete integer
);

CREATE TABLE processing_job_items (
  id text PRIMARY KEY,
  job_id text NOT NULL,
  item_type text NOT NULL
);

CREATE TABLE document_pages (
  id text PRIMARY KEY,
  document_id text NOT NULL,
  page_number integer NOT NULL,
  width_points integer,
  height_points integer,
  created_at timestamptz NOT NULL
);

CREATE TABLE document_page_blocks (
  id text PRIMARY KEY,
  page_id text NOT NULL,
  block_index integer NOT NULL,
  block_type text NOT NULL,
  text text NOT NULL,
  x0 numeric NOT NULL,
  y0 numeric NOT NULL,
  x1 numeric NOT NULL,
  y1 numeric NOT NULL,
  confidence numeric,
  created_at timestamptz NOT NULL
);

CREATE TABLE document_chunks (
  id text PRIMARY KEY,
  document_id text NOT NULL,
  page_number integer,
  chunk_index integer NOT NULL,
  content text NOT NULL,
  token_count integer,
  x0 numeric,
  y0 numeric,
  x1 numeric,
  y1 numeric,
  created_at timestamptz NOT NULL
);

CREATE TABLE chunk_vectors (
  id text PRIMARY KEY,
  chunk_id text NOT NULL,
  vector_store text NOT NULL,
  vector_id text NOT NULL,
  indexed_at timestamptz NOT NULL
);

CREATE TABLE citations (
  id text PRIMARY KEY,
  document_id text NOT NULL,
  chunk_id text,
  page_number integer NOT NULL,
  text text NOT NULL,
  context_before text,
  context_after text,
  x0 numeric NOT NULL,
  y0 numeric NOT NULL,
  x1 numeric NOT NULL,
  y1 numeric NOT NULL,
  confidence numeric,
  created_at timestamptz NOT NULL
);

CREATE TABLE conversations (
  id text PRIMARY KEY,
  document_id text NOT NULL,
  created_at timestamptz NOT NULL,
  last_message_at timestamptz
);

CREATE TABLE messages (
  id text PRIMARY KEY,
  conversation_id text NOT NULL,
  document_id text NOT NULL,
  role text NOT NULL,
  content text NOT NULL,
  created_at timestamptz NOT NULL,
  model text,
  temperature numeric,
  max_tokens integer,
  tokens_prompt integer,
  tokens_completion integer,
  tokens_total integer,
  retrieval_latency_ms integer,
  generation_latency_ms integer
);

CREATE TABLE message_citations (
  id text PRIMARY KEY,
  message_id text NOT NULL,
  citation_id text NOT NULL,
  display_index integer NOT NULL,
  created_at timestamptz NOT NULL
);

CREATE TABLE idempotency_keys (
  id text PRIMARY KEY,
  idempotency_key text NOT NULL,
  scope text NOT NULL,
  request_hash text NOT NULL,
  response_code integer,
  response_body jsonb,
  created_at timestamptz NOT NULL,
  expires_at timestamptz
);

CREATE INDEX idx_documents_status ON documents (status);
CREATE INDEX idx_documents_uploaded_at ON documents (uploaded_at);

CREATE INDEX idx_document_metadata_document_id ON document_metadata (document_id);
CREATE INDEX idx_document_summaries_document_id ON document_summaries (document_id);
CREATE INDEX idx_document_stats_document_id ON document_stats (document_id);

CREATE INDEX idx_document_tag_links_document_id ON document_tag_links (document_id);
CREATE INDEX idx_document_tag_links_tag_id ON document_tag_links (tag_id);

CREATE INDEX idx_processing_jobs_document_id ON processing_jobs (document_id, status);
CREATE INDEX idx_processing_job_stages_job_id ON processing_job_stages (job_id);

CREATE INDEX idx_document_pages_document_id ON document_pages (document_id, page_number);
CREATE INDEX idx_document_page_blocks_page_id ON document_page_blocks (page_id);

CREATE INDEX idx_document_chunks_document_id ON document_chunks (document_id, page_number);
CREATE INDEX idx_chunk_vectors_chunk_id ON chunk_vectors (chunk_id);

CREATE INDEX idx_citations_document_id ON citations (document_id, page_number);

CREATE INDEX idx_conversations_document_id ON conversations (document_id);
CREATE INDEX idx_messages_conversation_id ON messages (conversation_id, created_at);
CREATE INDEX idx_message_citations_message_id ON message_citations (message_id);

CREATE INDEX idx_idempotency_key_scope ON idempotency_keys (idempotency_key, scope);

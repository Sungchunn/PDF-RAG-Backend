# Database Relations

This backend uses PostgreSQL with pgvector. SQLAlchemy models are in `app/db/models/*`, and migrations are in `db/migration/*`.

## Key design choices

- Relationships are application-enforced; the migrations do not add foreign key constraints.
- Soft-deletion is used for documents (`documents.deleted_at`).
- Vector embeddings are stored in PostgreSQL via pgvector (`document_chunks.embedding`).

## Core entities

### Users

- `users` stores login credentials and status flags.
- One user can own many documents and conversations.

### Documents

- `documents` is the root entity for uploaded PDFs.
- Related tables:
  - `document_metadata` (1:1)
  - `document_summaries` (1:1)
  - `document_stats` (1:1)
  - `document_pages` (1:M)
  - `document_chunks` (1:M)
  - `citations` (1:M)
  - `conversations` (1:M)
  - `document_tag_links` (1:M)

### Conversations and messages

- `conversations` group chat history per document (and optionally per user).
- `messages` store chat messages and token usage stats.
- `message_citations` links messages to citations with a display order.

### Chunks and vectors

- `document_chunks` stores extracted text blocks, bounding boxes, line numbers, and embeddings.
- `chunk_vectors` tracks where an embedding is indexed (currently `pgvector`).

### Processing jobs

- `processing_jobs` tracks async document processing.
- `processing_job_stages` stores per-stage status and progress.
- `processing_job_items` stores per-item processing metadata.

### Idempotency

- `idempotency_keys` is designed to dedupe API requests (not currently wired in routes).

## Relationship map (logical)

```
users (1) ----< documents
users (1) ----< conversations

documents (1) ----< document_pages ----< document_page_blocks

documents (1) ----< document_chunks ----< chunk_vectors
                         \
                          \----< citations ----< message_citations >---- messages ----< conversations

documents (1) ----< document_metadata

documents (1) ----< document_summaries

documents (1) ----< document_stats

documents (1) ----< document_tag_links >---- tags

documents (1) ----< processing_jobs ----< processing_job_stages
                            \
                             \----< processing_job_items
```

## Indexes and vector search

- `document_chunks.embedding` uses a pgvector `ivfflat` index with cosine ops (see `db/migration/V4__add_embedding_column.sql`).
- Common filters are indexed: document status, user ownership, and job status.

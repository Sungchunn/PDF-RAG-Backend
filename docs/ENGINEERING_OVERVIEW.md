# Engineering Overview

This document explains how the backend is assembled and how the primary workflows behave.

## Architecture map

- `app/main.py`: FastAPI app factory, CORS, lifecycle hooks, and route wiring.
- `app/config.py`: Pydantic settings for database, providers, JWT, and RAG tuning.
- `app/api/routes/*`: HTTP endpoints for auth, documents, chat, jobs, health.
- `app/api/deps.py`: auth dependencies, DB session management, API key checks.
- `app/db/*`: SQLAlchemy async engine + ORM models for persistence.
- `app/core/*`: PDF parsing, RAG pipeline, pgvector integration.
- `app/workers/document_worker.py`: background processing for parsing + indexing.

Note: `app/services/document_service.py` and `app/services/chat_service.py` are not wired into the current API routes; the active request paths use `app/api/routes/*` and `app/workers/document_worker.py`.

## Primary flows

### Document upload and processing

1. `POST /api/documents/upload` validates PDF type and size, computes a SHA-256 checksum, and saves the file to `settings.upload_dir`.
2. A `documents` row is inserted with `status = "processing"` and the current `user_id`.
3. A `processing_jobs` row is created with job type `document_upload`.
4. A background task kicks off `process_document_task`.
5. Background worker stages:
   - Parse PDF into `TextBlock` records (text + page + bbox + line numbers).
   - Create embeddings for each block and store them in `document_chunks.embedding`.
   - Update the document status to `ready` with a page count.

### Chat / RAG query

1. `POST /api/chat` validates provider keys and document ownership (when a documentId is provided).
2. `RAGEngine` embeds the question.
3. `PGVectorStore` performs cosine similarity search in `document_chunks` scoped by `user_id`.
4. Context blocks are composed into a prompt and sent to the LLM.
5. The response returns answer text and citations with page + bbox + line numbers.

### Job tracking

- `GET /api/jobs/{job_id}` returns status, progress, and stage breakdown.
- `GET /api/jobs/document/{document_id}` returns recent jobs for a document.

## Isolation and ownership

- Requests that require auth use `CurrentUserDep` to enforce JWT authentication.
- Documents and conversations are scoped by `user_id`.
- Vector search is scoped by `user_id` inside `PGVectorStore.query`, so retrieval only searches documents owned by the current user.

## Configuration highlights

- `LLM_PROVIDER`, `EMBEDDING_PROVIDER`: OpenAI or Gemini selection.
- `CHUNK_SIZE`, `CHUNK_OVERLAP`: LlamaIndex defaults; current pipeline uses PDF blocks as chunks.
- `SIMILARITY_TOP_K`: number of chunks retrieved.
- `JWT_*`: access and refresh token settings.

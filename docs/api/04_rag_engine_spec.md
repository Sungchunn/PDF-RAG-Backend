# RAG Engine Specification

This document details the internal specifications of the Retrieval-Augmented Generation (RAG) engine, including vector storage, similarity search, and hybrid provider support.

## 1. Vector Store (pgvector)

The system uses PostgreSQL with the `pgvector` extension for storing and searching document embeddings.

### Data Model: `document_chunks`
| Field | Type | Description |
|---|---|---|
| `id` | UUID | Unique chunk ID |
| `document_id` | UUID | Reference to parent document |
| `content` | Text | The raw text of the chunk |
| `embedding` | vector(1536) | 1536-dimensional vector (OpenAI default) |
| `page_number` | Int | Source page (1-indexed) |
| `x0, y0, x1, y1` | Numeric | Spatial bounding box |
| `line_start, line_end` | Int | Document-wide line range |

### Similarity Metric
- **Algorithm:** Cosine Distance (`<=>`)
- **Calculation:** `1 - (embedding <=> query_vector)`
- **Indexing:** IVFFlat or HNSW (depending on migration state)

## 2. Hybrid Provider Support

The engine is designed to swap between AI providers via environment configuration.

| Provider | Supported Models (LLM) | Supported Models (Embedding) |
|---|---|---|
| **OpenAI** | `gpt-4o`, `gpt-3.5-turbo` | `text-embedding-3-small` (1536d) |
| **Google Gemini** | `gemini-1.5-pro`, `gemini-1.5-flash` | `text-embedding-004` (768d) |

> **Note:** Switching providers requires re-indexing existing documents if the embedding dimensions change.

## 3. The Retrieval Algorithm

1.  **Query Embedding:** Convert the user's natural language question into a vector $\vec{q}$.
2.  **User Isolation:** Filter chunks where `document.user_id == current_user_id`.
3.  **Soft Delete Filter:** Filter chunks where `document.deleted_at IS NULL`.
4.  **Top-K Search:** Retrieve the top $k$ chunks (default $k=5$) ordered by cosine similarity.
5.  **Context Assembly:** Format retrieved chunks into a prompt-friendly string with metadata tags (e.g., `[Source: Page X, Lines Y-Z]`).

## 4. Internal API: `RAGEngine`

### `index_document(document_id, blocks)`
- **Purpose:** Converts parsed `TextBlock` objects into embedded chunks.
- **Workflow:**
    1. Loop through blocks.
    2. Call Embedding API for each text snippet.
    3. Bulk insert into `document_chunks`.

### `query(question, document_id=None)`
- **Purpose:** Executes the RAG pipeline.
- **Workflow:**
    1. Embed question.
    2. Query `PGVectorStore`.
    3. Construct LLM Prompt.
    4. Return `answer` + `contexts` (for citations).

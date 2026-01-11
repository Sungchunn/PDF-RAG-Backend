# DocChat Backend API Specification

> **Version:** 1.0.0
> **Based on:** Frontend Implementation Analysis
> **Principles:** Clarity, Modularity, Conditional Rendering, Logical Robustness

---

## Table of Contents

1. [Design Principles](#design-principles)
2. [API Modules Overview](#api-modules-overview)
3. [Module 1: Document Management](#module-1-document-management)
4. [Module 2: Chat & LLM](#module-2-chat--llm)
5. [Module 3: PDF Processing](#module-3-pdf-processing)
6. [Module 4: Citation & Retrieval](#module-4-citation--retrieval)
7. [Module 5: System & Health](#module-5-system--health)
8. [Shared Data Types](#shared-data-types)
9. [Error Handling](#error-handling)
10. [Conditional Response Patterns](#conditional-response-patterns)
11. [WebSocket Events](#websocket-events)

---

## Design Principles

### 1. Separation of Concerns
Each API module handles a single domain. Modules should **never** be tangled:

| Module | Responsibility | Does NOT Handle |
|--------|---------------|-----------------|
| Document Management | CRUD operations, metadata | PDF parsing, LLM calls |
| Chat & LLM | Message handling, AI responses | Document storage, PDF rendering |
| PDF Processing | Text extraction, page rendering | Storage, chat history |
| Citation & Retrieval | Vector search, source lookup | LLM generation, file storage |

### 2. Conditional Response Fields
All responses support conditional fields to minimize payload size:

```typescript
// Query parameter: ?include=metadata,citations,content
// Only requested fields are returned
```

### 3. Idempotency
All mutating operations (POST, PUT, DELETE) should support idempotency keys:
```
X-Idempotency-Key: <uuid>
```

### 4. Async Processing Pattern
Long-running operations return immediately with a job ID:
```json
{
  "jobId": "job_abc123",
  "status": "processing",
  "statusUrl": "/api/jobs/job_abc123"
}
```

---

## API Modules Overview

```
/api
├── /documents          # Module 1: Document Management
│   ├── POST   /upload
│   ├── GET    /
│   ├── GET    /:id
│   ├── DELETE /:id
│   └── GET    /:id/status
│
├── /chat               # Module 2: Chat & LLM
│   ├── POST   /message
│   ├── POST   /message/stream
│   ├── GET    /history/:documentId
│   └── DELETE /history/:documentId
│
├── /pdf                # Module 3: PDF Processing
│   ├── GET    /:documentId/render
│   ├── GET    /:documentId/page/:pageNum
│   ├── GET    /:documentId/thumbnail
│   └── GET    /:documentId/text
│
├── /retrieval          # Module 4: Citation & Retrieval
│   ├── POST   /search
│   ├── GET    /citation/:citationId
│   └── GET    /document/:documentId/chunks
│
└── /system             # Module 5: System & Health
    ├── GET    /health
    ├── GET    /health/detailed
    └── GET    /config
```

---

## Module 1: Document Management

Handles document lifecycle without concerning itself with content processing.

### POST `/api/documents/upload`

Upload a new document. Returns immediately with processing status.

**Request:**
```http
POST /api/documents/upload
Content-Type: multipart/form-data

file: <binary>
metadata[title]: "Custom Title" (optional)
metadata[tags]: ["research", "ml"] (optional)
```

**Response (202 Accepted):**
```json
{
  "success": true,
  "data": {
    "documentId": "doc_a1b2c3d4",
    "filename": "research-paper.pdf",
    "status": "uploading",
    "uploadedAt": "2024-01-15T10:30:00Z",
    "processingJob": {
      "jobId": "job_xyz789",
      "statusUrl": "/api/documents/doc_a1b2c3d4/status"
    }
  }
}
```

**Status Flow:**
```
uploading → processing → ready
                ↓
              error
```

---

### GET `/api/documents`

List all documents with optional filtering.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `status` | string | Filter by status: `uploading`, `processing`, `ready`, `error` |
| `limit` | number | Pagination limit (default: 20, max: 100) |
| `offset` | number | Pagination offset |
| `sort` | string | Sort field: `uploadedAt`, `name`, `size` |
| `order` | string | Sort order: `asc`, `desc` |
| `include` | string | Comma-separated: `metadata`, `stats`, `summary` |

**Response:**
```json
{
  "success": true,
  "data": {
    "documents": [
      {
        "id": "doc_a1b2c3d4",
        "name": "Introduction to Machine Learning.pdf",
        "size": 2456789,
        "uploadedAt": "2024-01-10T10:30:00Z",
        "status": "ready",
        "pageCount": 42,

        // Conditional: include=metadata
        "metadata": {
          "author": "John Doe",
          "createdAt": "2023-12-01",
          "title": "Introduction to Machine Learning"
        },

        // Conditional: include=stats
        "stats": {
          "chunkCount": 156,
          "wordCount": 28450,
          "messageCount": 12
        },

        // Conditional: include=summary
        "summary": "A comprehensive guide covering supervised, unsupervised..."
      }
    ],
    "pagination": {
      "total": 45,
      "limit": 20,
      "offset": 0,
      "hasMore": true
    }
  }
}
```

---

### GET `/api/documents/:id`

Get single document details.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `include` | string | Comma-separated: `metadata`, `stats`, `summary`, `recentMessages` |

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "doc_a1b2c3d4",
    "name": "Introduction to Machine Learning.pdf",
    "size": 2456789,
    "uploadedAt": "2024-01-10T10:30:00Z",
    "status": "ready",
    "pageCount": 42,
    "mimeType": "application/pdf",

    // Always included for ready documents
    "urls": {
      "download": "/api/pdf/doc_a1b2c3d4/render",
      "thumbnail": "/api/pdf/doc_a1b2c3d4/thumbnail"
    }
  }
}
```

---

### DELETE `/api/documents/:id`

Delete a document and all associated data (chunks, embeddings, chat history).

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "documentId": "doc_a1b2c3d4",
    "deletedAt": "2024-01-15T10:30:00Z",
    "cleanupJob": {
      "jobId": "cleanup_xyz",
      "itemsQueued": ["embeddings", "chunks", "chatHistory"]
    }
  }
}
```

---

### GET `/api/documents/:id/status`

Poll document processing status. Used for upload progress tracking.

**Response:**
```json
{
  "success": true,
  "data": {
    "documentId": "doc_a1b2c3d4",
    "status": "processing",
    "progress": {
      "stage": "embedding",
      "stagesCompleted": ["upload", "parse", "chunk"],
      "stagesRemaining": ["embedding", "index"],
      "percentComplete": 60,
      "estimatedSecondsRemaining": 45
    },

    // Only present if status === "error"
    "error": {
      "code": "PARSE_FAILED",
      "message": "Unable to extract text from PDF",
      "retryable": true
    }
  }
}
```

---

## Module 2: Chat & LLM

Handles all LLM interactions. Decoupled from document storage and retrieval logic.

### POST `/api/chat/message`

Send a message and receive AI response with citations.

**Request:**
```json
{
  "message": "What is supervised learning?",
  "documentId": "doc_a1b2c3d4",
  "conversationId": "conv_123",  // Optional: for multi-turn
  "options": {
    "temperature": 0.7,          // Optional: 0.0-1.0
    "maxTokens": 1024,           // Optional
    "includeCitations": true,    // Optional: default true
    "citationStyle": "inline"    // Optional: "inline" | "footnote" | "none"
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "messageId": "msg_abc123",
    "conversationId": "conv_123",
    "role": "assistant",
    "content": "Supervised learning is a type of machine learning where the model is trained on labeled data [1]. The algorithm learns to map inputs to outputs based on example input-output pairs [2].",
    "timestamp": "2024-01-15T10:35:00Z",

    // Conditional: includeCitations=true
    "citations": [
      {
        "citationId": "cite_001",
        "index": 1,
        "documentId": "doc_a1b2c3d4",
        "pageNumber": 12,
        "text": "Supervised learning uses labeled data to train models...",
        "boundingBox": {
          "x0": 100, "y0": 200,
          "x1": 450, "y1": 280
        },
        "confidence": 0.95
      },
      {
        "citationId": "cite_002",
        "index": 2,
        "documentId": "doc_a1b2c3d4",
        "pageNumber": 15,
        "text": "The algorithm learns to map inputs to outputs...",
        "boundingBox": {
          "x0": 80, "y0": 340,
          "x1": 480, "y1": 400
        },
        "confidence": 0.92
      }
    ],

    // Metadata for debugging/analytics
    "meta": {
      "model": "gpt-4",
      "tokensUsed": {
        "prompt": 1250,
        "completion": 180,
        "total": 1430
      },
      "retrievalLatencyMs": 120,
      "generationLatencyMs": 2400
    }
  }
}
```

---

### POST `/api/chat/message/stream`

Stream AI response for real-time display. Uses Server-Sent Events (SSE).

**Request:** Same as `/api/chat/message`

**Response (SSE Stream):**
```
event: start
data: {"messageId": "msg_abc123", "conversationId": "conv_123"}

event: token
data: {"content": "Supervised"}

event: token
data: {"content": " learning"}

event: token
data: {"content": " is"}

event: citation
data: {"index": 1, "citationId": "cite_001", "pageNumber": 12}

event: token
data: {"content": " [1]"}

event: done
data: {"tokensUsed": 180, "citations": [...full citation objects...]}

event: error
data: {"code": "CONTEXT_LENGTH_EXCEEDED", "message": "..."}
```

---

### GET `/api/chat/history/:documentId`

Get chat history for a document.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `conversationId` | string | Filter by specific conversation |
| `limit` | number | Number of messages (default: 50) |
| `before` | string | Cursor for pagination (messageId) |
| `includeCitations` | boolean | Include citation details (default: false) |

**Response:**
```json
{
  "success": true,
  "data": {
    "documentId": "doc_a1b2c3d4",
    "conversations": [
      {
        "conversationId": "conv_123",
        "createdAt": "2024-01-15T10:30:00Z",
        "messageCount": 8,
        "lastMessageAt": "2024-01-15T11:45:00Z"
      }
    ],
    "messages": [
      {
        "messageId": "msg_001",
        "conversationId": "conv_123",
        "role": "user",
        "content": "What is supervised learning?",
        "timestamp": "2024-01-15T10:30:00Z"
      },
      {
        "messageId": "msg_002",
        "conversationId": "conv_123",
        "role": "assistant",
        "content": "Supervised learning is...",
        "timestamp": "2024-01-15T10:30:15Z",
        "citationIds": ["cite_001", "cite_002"]  // Light reference
      }
    ],
    "pagination": {
      "hasMore": true,
      "nextCursor": "msg_001"
    }
  }
}
```

---

### DELETE `/api/chat/history/:documentId`

Clear chat history for a document.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `conversationId` | string | Delete specific conversation only |

**Response:**
```json
{
  "success": true,
  "data": {
    "deletedCount": 24,
    "documentId": "doc_a1b2c3d4"
  }
}
```

---

## Module 3: PDF Processing

Handles PDF rendering and text extraction. Stateless operations.

### GET `/api/pdf/:documentId/render`

Download or stream the original PDF file.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `disposition` | string | `inline` (view) or `attachment` (download) |

**Response:**
```http
Content-Type: application/pdf
Content-Disposition: inline; filename="document.pdf"

<binary PDF data>
```

---

### GET `/api/pdf/:documentId/page/:pageNum`

Render a single page as an image.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `format` | string | `png`, `jpeg`, `webp` (default: `png`) |
| `width` | number | Target width in pixels (default: 800) |
| `quality` | number | JPEG/WebP quality 1-100 (default: 85) |
| `highlight` | string | Citation ID to highlight on page |

**Response:**
```http
Content-Type: image/png
Cache-Control: public, max-age=86400

<binary image data>
```

---

### GET `/api/pdf/:documentId/thumbnail`

Get document thumbnail for preview.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `page` | number | Page to thumbnail (default: 1) |
| `size` | string | `small` (150px), `medium` (300px), `large` (600px) |

**Response:**
```http
Content-Type: image/webp
Cache-Control: public, max-age=604800

<binary image data>
```

---

### GET `/api/pdf/:documentId/text`

Extract text content from PDF.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `page` | number | Specific page (omit for all pages) |
| `format` | string | `plain`, `structured` (with positions) |

**Response (format=structured):**
```json
{
  "success": true,
  "data": {
    "documentId": "doc_a1b2c3d4",
    "pageCount": 42,
    "pages": [
      {
        "pageNumber": 1,
        "width": 612,
        "height": 792,
        "blocks": [
          {
            "type": "paragraph",
            "text": "Introduction to Machine Learning",
            "boundingBox": {"x0": 72, "y0": 72, "x1": 540, "y1": 100},
            "confidence": 0.98
          }
        ]
      }
    ]
  }
}
```

---

## Module 4: Citation & Retrieval

Handles vector search and citation management. Core RAG functionality.

### POST `/api/retrieval/search`

Search for relevant chunks across documents.

**Request:**
```json
{
  "query": "How does backpropagation work?",
  "documentIds": ["doc_a1b2c3d4"],  // Optional: filter to specific docs
  "options": {
    "topK": 5,                       // Number of results
    "minScore": 0.7,                 // Minimum similarity threshold
    "includeContent": true,          // Include chunk text
    "includeMetadata": true          // Include position info
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "query": "How does backpropagation work?",
    "results": [
      {
        "chunkId": "chunk_xyz",
        "documentId": "doc_a1b2c3d4",
        "documentName": "Neural Networks Deep Dive.pdf",
        "score": 0.94,
        "pageNumber": 23,

        // Conditional: includeContent=true
        "content": "Backpropagation computes gradients by applying the chain rule...",

        // Conditional: includeMetadata=true
        "metadata": {
          "boundingBox": {"x0": 72, "y0": 300, "x1": 540, "y1": 380},
          "chunkIndex": 45,
          "tokenCount": 128
        }
      }
    ],
    "meta": {
      "searchLatencyMs": 45,
      "totalChunksSearched": 1250
    }
  }
}
```

---

### GET `/api/retrieval/citation/:citationId`

Get full citation details for display/highlighting.

**Response:**
```json
{
  "success": true,
  "data": {
    "citationId": "cite_001",
    "documentId": "doc_a1b2c3d4",
    "documentName": "Introduction to Machine Learning.pdf",
    "pageNumber": 12,
    "text": "Supervised learning uses labeled data to train models that can predict outcomes for new, unseen data.",
    "boundingBox": {
      "x0": 100, "y0": 200,
      "x1": 450, "y1": 280
    },
    "context": {
      "before": "...previous sentence for context...",
      "after": "...following sentence for context..."
    },
    "confidence": 0.95,
    "createdAt": "2024-01-15T10:35:00Z"
  }
}
```

---

### GET `/api/retrieval/document/:documentId/chunks`

List all chunks for a document (for debugging/admin).

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `page` | number | Filter by page number |
| `limit` | number | Pagination limit |
| `offset` | number | Pagination offset |

**Response:**
```json
{
  "success": true,
  "data": {
    "documentId": "doc_a1b2c3d4",
    "totalChunks": 156,
    "chunks": [
      {
        "chunkId": "chunk_001",
        "pageNumber": 1,
        "content": "Chapter 1: Introduction...",
        "tokenCount": 128,
        "boundingBox": {"x0": 72, "y0": 100, "x1": 540, "y1": 200}
      }
    ],
    "pagination": {
      "limit": 20,
      "offset": 0,
      "hasMore": true
    }
  }
}
```

---

## Module 5: System & Health

System status and configuration endpoints.

### GET `/api/system/health`

Basic health check for load balancers.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Response (503 Service Unavailable):**
```json
{
  "status": "unhealthy",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

### GET `/api/system/health/detailed`

Detailed health check for monitoring.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "components": {
    "database": {
      "status": "healthy",
      "latencyMs": 5
    },
    "vectorStore": {
      "status": "healthy",
      "latencyMs": 12,
      "indexedDocuments": 45
    },
    "llmProvider": {
      "status": "healthy",
      "provider": "openai",
      "model": "gpt-4"
    },
    "storage": {
      "status": "healthy",
      "usedBytes": 1073741824,
      "availableBytes": 10737418240
    }
  }
}
```

---

### GET `/api/system/config`

Get client-safe configuration.

**Response:**
```json
{
  "success": true,
  "data": {
    "maxUploadSizeMb": 50,
    "supportedFormats": ["pdf"],
    "features": {
      "streaming": true,
      "multiDocument": true,
      "darkMode": true
    },
    "limits": {
      "maxDocuments": 100,
      "maxMessagesPerConversation": 1000,
      "maxConcurrentUploads": 3
    }
  }
}
```

---

## Shared Data Types

### Document Status Enum
```typescript
type DocumentStatus = 'uploading' | 'processing' | 'ready' | 'error';
```

### Bounding Box
```typescript
interface BoundingBox {
  x0: number;  // Left edge (PDF points from origin)
  y0: number;  // Top edge
  x1: number;  // Right edge
  y1: number;  // Bottom edge
}
```

### Citation
```typescript
interface Citation {
  citationId: string;
  index: number;           // Display number [1], [2], etc.
  documentId: string;
  pageNumber: number;
  text: string;            // Extracted source text
  boundingBox: BoundingBox;
  confidence: number;      // 0.0 - 1.0
}
```

### Message Role
```typescript
type MessageRole = 'user' | 'assistant' | 'system';
```

---

## Error Handling

### Error Response Format
```json
{
  "success": false,
  "error": {
    "code": "DOCUMENT_NOT_FOUND",
    "message": "The requested document does not exist",
    "details": {
      "documentId": "doc_invalid"
    },
    "requestId": "req_abc123",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid request parameters |
| `UNAUTHORIZED` | 401 | Missing or invalid authentication |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `DOCUMENT_NOT_FOUND` | 404 | Document does not exist |
| `CITATION_NOT_FOUND` | 404 | Citation does not exist |
| `CONFLICT` | 409 | Resource already exists |
| `PAYLOAD_TOO_LARGE` | 413 | File exceeds size limit |
| `UNSUPPORTED_FORMAT` | 415 | File type not supported |
| `RATE_LIMITED` | 429 | Too many requests |
| `PROCESSING_FAILED` | 500 | Document processing error |
| `LLM_ERROR` | 502 | LLM provider error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

---

## Conditional Response Patterns

### Pattern 1: Include Parameter
Request specific fields to reduce payload:

```http
GET /api/documents?include=metadata,stats
GET /api/documents/doc_123?include=summary,recentMessages
GET /api/chat/history/doc_123?includeCitations=true
```

### Pattern 2: Sparse Fieldsets
Request only needed fields:

```http
GET /api/documents?fields=id,name,status
```

### Pattern 3: Expand Related Resources
Inline related data:

```http
GET /api/chat/history/doc_123?expand=citations
```

### Pattern 4: Conditional Headers
Use ETags for caching:

```http
GET /api/documents/doc_123
If-None-Match: "abc123"

Response: 304 Not Modified (if unchanged)
```

---

## WebSocket Events

For real-time updates, connect to `ws://api/ws`.

### Connection
```javascript
const ws = new WebSocket('ws://api/ws?token=<auth_token>');
```

### Event Types

**Document Status Updates:**
```json
{
  "event": "document.status",
  "data": {
    "documentId": "doc_123",
    "status": "ready",
    "progress": null
  }
}
```

**Processing Progress:**
```json
{
  "event": "document.progress",
  "data": {
    "documentId": "doc_123",
    "stage": "embedding",
    "percentComplete": 75
  }
}
```

**Error Notification:**
```json
{
  "event": "document.error",
  "data": {
    "documentId": "doc_123",
    "error": {
      "code": "PARSE_FAILED",
      "message": "Unable to extract text"
    }
  }
}
```

---

## Implementation Recommendations

### 1. Module Isolation
```
┌────────────────────────────────────────────────────────────┐
│                      API Gateway                           │
├─────────────┬─────────────┬──────────────┬─────────────────┤
│  Document   │    Chat     │     PDF      │    Retrieval    │
│   Service   │   Service   │   Service    │    Service      │
├─────────────┼─────────────┼──────────────┼─────────────────┤
│  PostgreSQL │   Redis     │  File Store  │  Vector Store   │
│  (metadata) │  (sessions) │  (S3/local)  │  (Pinecone/     │
│             │             │              │   Qdrant)       │
└─────────────┴─────────────┴──────────────┴─────────────────┘
```

### 2. Processing Pipeline
```
Upload → Parse PDF → Chunk Text → Generate Embeddings → Index
         ↓
    Extract Metadata
         ↓
    Generate Summary (optional LLM call)
```

### 3. Caching Strategy
| Resource | Cache Duration | Invalidation |
|----------|---------------|--------------|
| PDF renders | 24 hours | On document delete |
| Thumbnails | 7 days | On document delete |
| Document list | 5 minutes | On any document change |
| Chat history | No cache | Real-time |
| Citations | 1 hour | Never (immutable) |

---

## Security Considerations

1. **Authentication**: All endpoints except `/api/system/health` require authentication
2. **Rate Limiting**: Apply per-user limits, especially on LLM endpoints
3. **File Validation**: Validate PDF magic bytes, not just extension
4. **Input Sanitization**: Sanitize all user input before LLM prompts
5. **CORS**: Restrict to known frontend origins

---

## Appendix: Frontend Type Mapping

```typescript
// Frontend types → API response mapping

// Document (frontend) ← GET /api/documents/:id
interface Document {
  id: string;           // ← data.id
  name: string;         // ← data.name
  size: number;         // ← data.size
  uploadedAt: string;   // ← data.uploadedAt
  status: DocumentStatus; // ← data.status
  pageCount?: number;   // ← data.pageCount
}

// ChatMessage (frontend) ← GET /api/chat/history
interface ChatMessage {
  id: string;           // ← messages[].messageId
  role: 'user' | 'assistant'; // ← messages[].role
  content: string;      // ← messages[].content
  timestamp: string;    // ← messages[].timestamp
  citations?: Citation[]; // ← expanded from citationIds
}

// Citation (frontend) ← Inline in chat response
interface Citation {
  documentId: string;   // ← citations[].documentId
  pageNumber: number;   // ← citations[].pageNumber
  boundingBox: BoundingBox; // ← citations[].boundingBox
  text: string;         // ← citations[].text
  confidence: number;   // ← citations[].confidence
}
```

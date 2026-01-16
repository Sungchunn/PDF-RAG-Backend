# Chat API Specification

This document details the API endpoints for the Conversational Interface (RAG Q&A).

**Base URL:** `/api/v1/chat`

## 1. Send Message (Q&A)

Sends a user question to the RAG engine and returns an AI-generated answer with citations.

- **Endpoint:** `POST /`
- **Auth Required:** Yes (Bearer Token)

### Request Body
| Field | Type | Required | Description |
|---|---|---|---|
| `message` | string | Yes | The user's question (max 5000 chars) |
| `documentId` | string | No | If provided, restricts search to this single document |
| `provider` | string | No | Override LLM provider (`openai` or `gemini`) |
| `model` | string | No | Override specific model name |

**Example Request:**
```json
{
  "message": "What is the net income for 2023?",
  "documentId": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Response (200 OK)
Returns the answer and a list of citation objects.

```json
{
  "answer": "The net income for 2023 was $12.5 million.",
  "citations": [
    {
      "documentId": "550e8400-e29b-41d4-a716-446655440000",
      "pageNumber": 14,
      "boundingBox": {
        "x0": 100.5,
        "y0": 200.0,
        "x1": 350.0,
        "y1": 220.0
      },
      "lineStart": 145,
      "lineEnd": 146,
      "text": "Net Income: $12.5M",
      "confidence": 0.89
    }
  ],
  "sourceDocuments": [
    "550e8400-e29b-41d4-a716-446655440000"
  ]
}
```

### Errors
- **404 Not Found:** If `documentId` is provided but not found/owned by user.
- **400 Bad Request:** If `documentId` is in `processing` or `error` state.
- **503 Service Unavailable:** If the configured AI provider is missing API keys.

---

## 2. Stream Message (Planned)

*Currently Not Implemented.*

- **Endpoint:** `POST /stream`
- **Response:** `501 Not Implemented`

Future design will support Server-Sent Events (SSE) for token-by-token streaming.

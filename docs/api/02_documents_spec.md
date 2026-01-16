# Document Management API Specification

This document details the API endpoints for managing PDF documents, including upload, retrieval, and deletion.

**Base URL:** `/api/v1/documents`

## 1. Upload Document

Uploads a PDF file for asynchronous processing.

- **Endpoint:** `POST /upload`
- **Auth Required:** Yes (Bearer Token)
- **Content-Type:** `multipart/form-data`

### Request Body
| Field | Type | Required | Description | Constraints |
|---|---|---|---|---|
| `file` | File | Yes | The PDF file to upload | `.pdf` extension, max size 10MB (configurable) |

### Response (200 OK)
Returns the initial status and a job ID for tracking.

```json
{
  "documentId": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "annual_report.pdf",
  "status": "processing",
  "jobId": "job_12345"
}
```

### Errors
- **400 Bad Request:** Invalid file type (not a PDF).
- **413 Request Entity Too Large:** File size exceeds the limit.
- **500 Internal Server Error:** File save failure.

---

## 2. List Documents

Retrieves a paginated list of the user's documents.

- **Endpoint:** `GET /`
- **Auth Required:** Yes

### Query Parameters
| Param | Type | Required | Default | Description |
|---|---|---|---|---|
| `status` | string | No | - | Filter by status (`processing`, `ready`, `error`) |
| `limit` | int | No | 50 | Max records to return (max 100) |
| `offset` | int | No | 0 | Number of records to skip |

### Response (200 OK)
```json
{
  "documents": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "annual_report.pdf",
      "sizeBytes": 1048576,
      "mimeType": "application/pdf",
      "status": "ready",
      "pageCount": 12,
      "uploadedAt": "2023-10-27T10:00:00Z",
      "deletedAt": null
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

---

## 3. Get Document Details

Retrieves metadata for a specific document.

- **Endpoint:** `GET /{document_id}`
- **Auth Required:** Yes

### Path Parameters
| Param | Type | Required | Description |
|---|---|---|---|
| `document_id` | string | Yes | Unique UUID of the document |

### Response (200 OK)
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "annual_report.pdf",
  "sizeBytes": 1048576,
  "mimeType": "application/pdf",
  "status": "ready",
  "pageCount": 12,
  "uploadedAt": "2023-10-27T10:00:00Z",
  "deletedAt": null
}
```

### Errors
- **404 Not Found:** Document does not exist, belongs to another user, or has been soft-deleted.

---

## 4. Delete Document

Soft-deletes a document. The file is marked as deleted but remains on disk initially. Vector embeddings are removed.

- **Endpoint:** `DELETE /{document_id}`
- **Auth Required:** Yes

### Path Parameters
| Param | Type | Required | Description |
|---|---|---|---|
| `document_id` | string | Yes | Unique UUID of the document |

### Response (200 OK)
```json
{
  "status": "deleted",
  "documentId": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Errors
- **404 Not Found:** Document not found.
- **400 Bad Request:** Document already deleted.

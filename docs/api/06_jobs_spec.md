# Job System API Specification

This document details the API endpoints for tracking asynchronous processing jobs.

**Base URL:** `/api/v1/jobs`

## 1. Get Job Status

Retrieves the detailed status of a specific processing job.

- **Endpoint:** `GET /{job_id}`
- **Auth Required:** Yes (Bearer Token)

### Path Parameters
| Param | Type | Required | Description |
|---|---|---|---|
| `job_id` | string | Yes | Unique UUID of the job |

### Response (200 OK)
```json
{
  "id": "job_12345",
  "documentId": "550e8400-e29b-41d4-a716-446655440000",
  "jobType": "document_upload",
  "status": "completed",
  "progressPercent": 100,
  "errorCode": null,
  "errorMessage": null,
  "createdAt": "2023-10-27T10:00:00Z",
  "startedAt": "2023-10-27T10:00:01Z",
  "completedAt": "2023-10-27T10:00:05Z",
  "stages": [
    {
      "id": "stage_1",
      "stageName": "parse_pdf",
      "status": "completed",
      "percentComplete": 100
    },
    {
      "id": "stage_2",
      "stageName": "index_vectors",
      "status": "completed",
      "percentComplete": 100
    }
  ]
}
```

### Errors
- **404 Not Found:** Job does not exist or user is unauthorized to view it.

---

## 2. List Jobs for Document

Retrieves all processing jobs associated with a specific document.

- **Endpoint:** `GET /document/{document_id}`
- **Auth Required:** Yes

### Path Parameters
| Param | Type | Required | Description |
|---|---|---|---|
| `document_id` | string | Yes | Unique UUID of the document |

### Query Parameters
| Param | Type | Default | Description |
|---|---|---|---|
| `limit` | int | 10 | Max number of jobs to return |

### Response (200 OK)
Returns a list of Job objects (same structure as above).

```json
[
  {
    "id": "job_98765",
    "jobType": "summary_generation",
    "status": "failed",
    "errorCode": "LLM_TIMEOUT",
    "errorMessage": "Provider timed out",
    ...
  },
  {
    "id": "job_12345",
    "jobType": "document_upload",
    "status": "completed",
    ...
  }
]
```

### Errors
- **404 Not Found:** Document does not exist or user is unauthorized.

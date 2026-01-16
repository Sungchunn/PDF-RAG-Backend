# Infrastructure API Specification

This document details the cross-cutting system infrastructure concerns including Rate Limiting, CORS, and Database Management.

## 1. Rate Limiting

The API implements token-bucket rate limiting to prevent abuse.

- **Library:** `slowapi` (backed by in-memory storage or Redis if configured).
- **Default Limit:** 100 requests per minute per IP address.
- **Identification:** Client IP address (`get_remote_address`).
- **Headers:** The following headers are returned on limit hit (429):
    - `Retry-After`: Seconds to wait before retrying.

### Errors
- **429 Too Many Requests:** When the rate limit is exceeded.
    ```json
    {
      "error": "Rate limit exceeded: 100/minute"
    }
    ```

## 2. CORS (Cross-Origin Resource Sharing)

The API is configured to allow requests from specific frontend origins.

- **Middleware:** `FastAPI.middleware.cors.CORSMiddleware`.
- **Allowed Origins:** Configurable via `CORS_ORIGINS` env var (default: `http://localhost:3000`).
- **Allowed Methods:** All (`*`).
- **Allowed Headers:** All (`*`).
- **Credentials:** Allowed (`Access-Control-Allow-Credentials: true`).

## 3. Database & Migrations

The system relies on PostgreSQL with the `pgvector` extension.

### Schema Management
- **ORM:** SQLAlchemy (Async).
- **Driver:** `asyncpg`.
- **Migrations:** While SQLAlchemy `create_all` is available for dev, production changes should use a migration tool (e.g., Flyway or Alembic). The system expects the `vector` extension to be enabled.

### Connection
- **Pooling:** Managed by `SQLAlchemy.create_async_engine`.
- **Session:** Per-request `AsyncSession` dependency injection.

## 4. Health Checks

- **Endpoint:** `GET /api/health` (Assuming standard implementation, though explicit code file wasn't shown in last read, `app/api/routes/health.py` exists in file tree).
- **Purpose:** Verifies DB connectivity and service uptime.

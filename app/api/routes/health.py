"""
Health check endpoints.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "message": "PDF-RAG API is running"}


@router.get("/ready")
async def readiness_check():
    """
    Readiness check - verify all critical services are available.
    Used by orchestration systems (K8s, Docker, etc.)
    """
    # Add database and other service checks here
    checks = {
        "api": "ready",
        # "database": check_database(),
        # "vector_store": check_vector_store(),
    }

    all_ready = all(v == "ready" for v in checks.values())

    return {
        "status": "ready" if all_ready else "not_ready",
        "checks": checks,
    }

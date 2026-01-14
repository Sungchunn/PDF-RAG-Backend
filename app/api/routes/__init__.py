"""
API router aggregating all route modules.
"""

from fastapi import APIRouter

from app.api.routes import health, documents, chat, auth, jobs

router = APIRouter()

# Include all route modules
router.include_router(health.router, prefix="/health", tags=["Health"])
router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
router.include_router(documents.router, prefix="/documents", tags=["Documents"])
router.include_router(chat.router, prefix="/chat", tags=["Chat"])
router.include_router(jobs.router, prefix="/jobs", tags=["Jobs"])

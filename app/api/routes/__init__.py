"""
API router aggregating all route modules.
"""

from fastapi import APIRouter

from app.api.routes import health, documents, chat

router = APIRouter()

# Include all route modules
router.include_router(health.router, prefix="/health", tags=["Health"])
router.include_router(documents.router, prefix="/documents", tags=["Documents"])
router.include_router(chat.router, prefix="/chat", tags=["Chat"])

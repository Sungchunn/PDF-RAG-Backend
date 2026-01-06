"""
API route dependencies.
"""

from typing import Annotated, Generator
from fastapi import Depends, HTTPException, status

from app.config import settings


def get_db():
    """Get database session dependency."""
    # Placeholder for database session
    # Will be properly implemented with SQLAlchemy
    try:
        yield None  # Replace with actual session
    finally:
        pass


def verify_api_key():
    """Verify OpenAI API key is configured (for RAG endpoints)."""
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY.",
        )
    return True


# Dependency annotations
DatabaseDep = Annotated[None, Depends(get_db)]  # Replace None with Session type
ApiKeyDep = Annotated[bool, Depends(verify_api_key)]

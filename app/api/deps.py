"""
API route dependencies.
"""

from typing import Annotated, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.auth import decode_token, is_token_expired
from app.db.database import get_db_session
from app.db.models import UserModel
from app.services.auth_service import AuthService


# HTTP Bearer scheme for JWT
security = HTTPBearer()
optional_security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    session: AsyncSession = Depends(get_db_session),
) -> UserModel:
    """
    Dependency to get current authenticated user.

    Requires valid JWT access token.
    """
    token_data = decode_token(credentials.credentials)

    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if is_token_expired(token_data):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if token_data.token_type != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"},
        )

    auth_service = AuthService(session)
    user = await auth_service.get_user_by_id(token_data.user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is deactivated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(optional_security),
    session: AsyncSession = Depends(get_db_session),
) -> Optional[UserModel]:
    """
    Optional authentication dependency.

    Returns user if authenticated, None otherwise.
    """
    if not credentials:
        return None

    try:
        token_data = decode_token(credentials.credentials)
        if not token_data or is_token_expired(token_data):
            return None

        if token_data.token_type != "access":
            return None

        auth_service = AuthService(session)
        user = await auth_service.get_user_by_id(token_data.user_id)

        if not user or not user.is_active:
            return None

        return user
    except Exception:
        return None


def verify_api_key():
    """Verify OpenAI API key is configured (for RAG endpoints)."""
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY.",
        )
    return True


# Dependency annotations
SessionDep = Annotated[AsyncSession, Depends(get_db_session)]
CurrentUserDep = Annotated[UserModel, Depends(get_current_user)]
OptionalUserDep = Annotated[Optional[UserModel], Depends(get_optional_user)]
ApiKeyDep = Annotated[bool, Depends(verify_api_key)]

"""
Authentication service for user management.
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import (
    hash_password,
    verify_password,
    create_token_pair,
    TokenPair,
)
from app.db.models import UserModel


class AuthService:
    """Service for authentication and user management."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def register(
        self,
        email: str,
        password: str,
        display_name: Optional[str] = None,
    ) -> UserModel:
        """
        Register a new user.

        Args:
            email: User's email address
            password: Plain text password
            display_name: Optional display name

        Returns:
            Created UserModel

        Raises:
            ValueError: If email already exists
        """
        # Check if email exists
        existing = await self.session.execute(
            select(UserModel).where(UserModel.email == email)
        )
        if existing.scalar_one_or_none():
            raise ValueError("Email already registered")

        now = datetime.now(timezone.utc)
        user = UserModel(
            id=str(uuid4()),
            email=email,
            password_hash=hash_password(password),
            display_name=display_name,
            is_active=True,
            is_verified=False,
            created_at=now,
            updated_at=now,
        )
        self.session.add(user)
        await self.session.flush()
        return user

    async def login(self, email: str, password: str) -> Optional[TokenPair]:
        """
        Authenticate user and return tokens.

        Args:
            email: User's email
            password: Plain text password

        Returns:
            TokenPair if credentials valid, None otherwise
        """
        result = await self.session.execute(
            select(UserModel).where(UserModel.email == email)
        )
        user = result.scalar_one_or_none()

        if not user:
            return None

        if not verify_password(password, user.password_hash):
            return None

        if not user.is_active:
            return None

        return create_token_pair(user.id, user.email)

    async def get_user_by_id(self, user_id: str) -> Optional[UserModel]:
        """Get user by ID."""
        return await self.session.get(UserModel, user_id)

    async def get_user_by_email(self, email: str) -> Optional[UserModel]:
        """Get user by email."""
        result = await self.session.execute(
            select(UserModel).where(UserModel.email == email)
        )
        return result.scalar_one_or_none()

    async def update_password(
        self,
        user_id: str,
        new_password: str,
    ) -> bool:
        """Update user's password."""
        user = await self.get_user_by_id(user_id)
        if not user:
            return False

        user.password_hash = hash_password(new_password)
        user.updated_at = datetime.now(timezone.utc)
        await self.session.flush()
        return True

    async def deactivate_user(self, user_id: str) -> bool:
        """Deactivate a user account."""
        user = await self.get_user_by_id(user_id)
        if not user:
            return False

        user.is_active = False
        user.updated_at = datetime.now(timezone.utc)
        await self.session.flush()
        return True

"""
API tests for authentication endpoints.
Tests registration, login, token refresh, and profile retrieval.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import status
from httpx import AsyncClient

from app.core.auth import hash_password, create_token_pair
from app.db.models import UserModel


class TestRegisterEndpoint:
    """Tests for POST /api/auth/register."""

    @pytest.mark.asyncio
    async def test_register_success(self, test_client, db_session):
        """Test successful user registration."""
        response = await test_client.post(
            "/api/auth/register",
            json={
                "email": "newuser@example.com",
                "password": "securepassword123",
                "displayName": "New User",
            },
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["email"] == "newuser@example.com"
        assert data["displayName"] == "New User"
        assert data["isActive"] is True
        assert "id" in data

    @pytest.mark.asyncio
    async def test_register_duplicate_email(self, test_client, test_user):
        """Test registration with existing email fails."""
        response = await test_client.post(
            "/api/auth/register",
            json={
                "email": test_user.email,  # Existing email
                "password": "newpassword123",
            },
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_register_invalid_email(self, test_client):
        """Test registration with invalid email format."""
        response = await test_client.post(
            "/api/auth/register",
            json={
                "email": "not-an-email",
                "password": "password123",
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_register_short_password(self, test_client):
        """Test registration with too short password."""
        response = await test_client.post(
            "/api/auth/register",
            json={
                "email": "user@example.com",
                "password": "short",  # Less than 8 characters
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_register_missing_password(self, test_client):
        """Test registration without password."""
        response = await test_client.post(
            "/api/auth/register",
            json={
                "email": "user@example.com",
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_register_without_display_name(self, test_client, db_session):
        """Test registration without display name is allowed."""
        response = await test_client.post(
            "/api/auth/register",
            json={
                "email": "nodisplay@example.com",
                "password": "password123",
            },
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["displayName"] is None


class TestLoginEndpoint:
    """Tests for POST /api/auth/login."""

    @pytest.mark.asyncio
    async def test_login_success(self, test_client, test_user, test_user_data):
        """Test successful login."""
        response = await test_client.post(
            "/api/auth/login",
            json={
                "email": test_user.email,
                "password": test_user_data["password"],
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "accessToken" in data
        assert "refreshToken" in data
        assert data["tokenType"] == "bearer"

    @pytest.mark.asyncio
    async def test_login_wrong_password(self, test_client, test_user):
        """Test login with wrong password."""
        response = await test_client.post(
            "/api/auth/login",
            json={
                "email": test_user.email,
                "password": "wrongpassword",
            },
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid email or password" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_login_nonexistent_user(self, test_client):
        """Test login with non-existent email."""
        response = await test_client.post(
            "/api/auth/login",
            json={
                "email": "nonexistent@example.com",
                "password": "password123",
            },
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_login_missing_email(self, test_client):
        """Test login without email."""
        response = await test_client.post(
            "/api/auth/login",
            json={
                "password": "password123",
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_login_inactive_user(self, test_client, db_session):
        """Test login with inactive user."""
        # Create inactive user
        inactive_user = UserModel(
            id="inactive-user-id",
            email="inactive@example.com",
            password_hash=hash_password("password123"),
            is_active=False,
            created_at=datetime.now(timezone.utc),
        )
        db_session.add(inactive_user)
        await db_session.flush()

        response = await test_client.post(
            "/api/auth/login",
            json={
                "email": "inactive@example.com",
                "password": "password123",
            },
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestRefreshEndpoint:
    """Tests for POST /api/auth/refresh."""

    @pytest.mark.asyncio
    async def test_refresh_success(self, test_client, test_user, test_user_tokens):
        """Test successful token refresh."""
        response = await test_client.post(
            "/api/auth/refresh",
            json={
                "refreshToken": test_user_tokens["refresh_token"],
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "accessToken" in data
        assert "refreshToken" in data
        # New tokens should be different
        assert data["accessToken"] != test_user_tokens["access_token"]

    @pytest.mark.asyncio
    async def test_refresh_invalid_token(self, test_client):
        """Test refresh with invalid token."""
        response = await test_client.post(
            "/api/auth/refresh",
            json={
                "refreshToken": "invalid.token.here",
            },
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_refresh_with_access_token(self, test_client, test_user_tokens):
        """Test refresh with access token (wrong type) fails."""
        response = await test_client.post(
            "/api/auth/refresh",
            json={
                "refreshToken": test_user_tokens["access_token"],  # Using access token
            },
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid token type" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_refresh_missing_token(self, test_client):
        """Test refresh without token."""
        response = await test_client.post(
            "/api/auth/refresh",
            json={},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestMeEndpoint:
    """Tests for GET /api/auth/me."""

    @pytest.mark.asyncio
    async def test_get_profile_success(self, test_client, test_user, auth_headers):
        """Test getting current user profile."""
        response = await test_client.get(
            "/api/auth/me",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == test_user.id
        assert data["email"] == test_user.email
        assert data["displayName"] == test_user.display_name
        assert data["isActive"] is True

    @pytest.mark.asyncio
    async def test_get_profile_unauthorized(self, test_client):
        """Test getting profile without authentication."""
        response = await test_client.get("/api/auth/me")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_get_profile_invalid_token(self, test_client):
        """Test getting profile with invalid token."""
        response = await test_client.get(
            "/api/auth/me",
            headers={"Authorization": "Bearer invalid.token"},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_get_profile_expired_token(self, test_client):
        """Test getting profile with expired token."""
        # Create an expired token
        with patch("app.core.auth.settings") as mock_settings:
            mock_settings.jwt_secret_key = "test-secret"
            mock_settings.jwt_algorithm = "HS256"
            mock_settings.jwt_access_expire_minutes = -1  # Already expired

            from app.core.auth import create_access_token
            expired_token = create_access_token("user-123", "user@example.com")

        response = await test_client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {expired_token}"},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

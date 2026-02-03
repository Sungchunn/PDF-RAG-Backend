"""
Unit tests for authentication module.
Tests password hashing, JWT token creation/validation, and token expiration.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

from app.core.auth import (
    verify_password,
    hash_password,
    create_access_token,
    create_refresh_token,
    create_token_pair,
    decode_token,
    is_token_expired,
    TokenData,
    TokenPair,
)


class TestPasswordHashing:
    """Tests for password hashing functions."""

    def test_hash_password_creates_hash(self):
        """Test that hash_password creates a bcrypt hash."""
        password = "my_secure_password"
        hashed = hash_password(password)

        assert hashed != password
        assert hashed.startswith("$2b$")  # bcrypt prefix
        assert len(hashed) == 60  # bcrypt hash length

    def test_hash_password_different_each_time(self):
        """Test that same password produces different hashes (due to salt)."""
        password = "my_secure_password"
        hash1 = hash_password(password)
        hash2 = hash_password(password)

        assert hash1 != hash2

    def test_verify_password_correct(self):
        """Test that verify_password returns True for correct password."""
        password = "my_secure_password"
        hashed = hash_password(password)

        assert verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Test that verify_password returns False for wrong password."""
        password = "my_secure_password"
        hashed = hash_password(password)

        assert verify_password("wrong_password", hashed) is False

    def test_verify_password_empty(self):
        """Test verification with empty password."""
        hashed = hash_password("actual_password")
        assert verify_password("", hashed) is False


class TestAccessToken:
    """Tests for access token creation and validation."""

    @patch("app.core.auth.settings")
    def test_create_access_token(self, mock_settings):
        """Test access token creation."""
        mock_settings.jwt_secret_key = "test-secret"
        mock_settings.jwt_algorithm = "HS256"
        mock_settings.jwt_access_expire_minutes = 30

        token = create_access_token("user-123", "user@example.com")

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    @patch("app.core.auth.settings")
    def test_access_token_decode(self, mock_settings):
        """Test that access token can be decoded."""
        mock_settings.jwt_secret_key = "test-secret"
        mock_settings.jwt_algorithm = "HS256"
        mock_settings.jwt_access_expire_minutes = 30

        token = create_access_token("user-123", "user@example.com")
        decoded = decode_token(token)

        assert decoded is not None
        assert decoded.user_id == "user-123"
        assert decoded.email == "user@example.com"
        assert decoded.token_type == "access"

    @patch("app.core.auth.settings")
    def test_access_token_expiration(self, mock_settings):
        """Test that access token has correct expiration."""
        mock_settings.jwt_secret_key = "test-secret"
        mock_settings.jwt_algorithm = "HS256"
        mock_settings.jwt_access_expire_minutes = 30

        token = create_access_token("user-123", "user@example.com")
        decoded = decode_token(token)

        now = datetime.now(timezone.utc)
        expected_exp = now + timedelta(minutes=30)

        # Allow 5 second tolerance
        assert abs((decoded.exp - expected_exp).total_seconds()) < 5


class TestRefreshToken:
    """Tests for refresh token creation and validation."""

    @patch("app.core.auth.settings")
    def test_create_refresh_token(self, mock_settings):
        """Test refresh token creation."""
        mock_settings.jwt_secret_key = "test-secret"
        mock_settings.jwt_algorithm = "HS256"
        mock_settings.jwt_refresh_expire_days = 7

        token = create_refresh_token("user-123", "user@example.com")

        assert token is not None
        assert isinstance(token, str)

    @patch("app.core.auth.settings")
    def test_refresh_token_decode(self, mock_settings):
        """Test that refresh token can be decoded."""
        mock_settings.jwt_secret_key = "test-secret"
        mock_settings.jwt_algorithm = "HS256"
        mock_settings.jwt_refresh_expire_days = 7

        token = create_refresh_token("user-123", "user@example.com")
        decoded = decode_token(token)

        assert decoded is not None
        assert decoded.user_id == "user-123"
        assert decoded.email == "user@example.com"
        assert decoded.token_type == "refresh"

    @patch("app.core.auth.settings")
    def test_refresh_token_longer_expiration(self, mock_settings):
        """Test that refresh token expires later than access token."""
        mock_settings.jwt_secret_key = "test-secret"
        mock_settings.jwt_algorithm = "HS256"
        mock_settings.jwt_access_expire_minutes = 30
        mock_settings.jwt_refresh_expire_days = 7

        access = decode_token(create_access_token("user-123", "user@example.com"))
        refresh = decode_token(create_refresh_token("user-123", "user@example.com"))

        assert refresh.exp > access.exp


class TestTokenPair:
    """Tests for token pair creation."""

    @patch("app.core.auth.settings")
    def test_create_token_pair(self, mock_settings):
        """Test creating both access and refresh tokens."""
        mock_settings.jwt_secret_key = "test-secret"
        mock_settings.jwt_algorithm = "HS256"
        mock_settings.jwt_access_expire_minutes = 30
        mock_settings.jwt_refresh_expire_days = 7

        pair = create_token_pair("user-123", "user@example.com")

        assert isinstance(pair, TokenPair)
        assert pair.access_token is not None
        assert pair.refresh_token is not None
        assert pair.token_type == "bearer"

    @patch("app.core.auth.settings")
    def test_token_pair_tokens_are_different(self, mock_settings):
        """Test that access and refresh tokens are different."""
        mock_settings.jwt_secret_key = "test-secret"
        mock_settings.jwt_algorithm = "HS256"
        mock_settings.jwt_access_expire_minutes = 30
        mock_settings.jwt_refresh_expire_days = 7

        pair = create_token_pair("user-123", "user@example.com")

        assert pair.access_token != pair.refresh_token


class TestDecodeToken:
    """Tests for token decoding."""

    @patch("app.core.auth.settings")
    def test_decode_valid_token(self, mock_settings):
        """Test decoding a valid token."""
        mock_settings.jwt_secret_key = "test-secret"
        mock_settings.jwt_algorithm = "HS256"
        mock_settings.jwt_access_expire_minutes = 30

        token = create_access_token("user-123", "user@example.com")
        decoded = decode_token(token)

        assert decoded is not None
        assert isinstance(decoded, TokenData)

    @patch("app.core.auth.settings")
    def test_decode_invalid_token(self, mock_settings):
        """Test decoding an invalid token returns None."""
        mock_settings.jwt_secret_key = "test-secret"
        mock_settings.jwt_algorithm = "HS256"

        decoded = decode_token("invalid.token.here")

        assert decoded is None

    @patch("app.core.auth.settings")
    def test_decode_wrong_secret(self, mock_settings):
        """Test decoding with wrong secret returns None."""
        mock_settings.jwt_secret_key = "test-secret"
        mock_settings.jwt_algorithm = "HS256"
        mock_settings.jwt_access_expire_minutes = 30

        token = create_access_token("user-123", "user@example.com")

        # Change the secret
        mock_settings.jwt_secret_key = "different-secret"
        decoded = decode_token(token)

        assert decoded is None

    @patch("app.core.auth.settings")
    def test_decode_expired_token(self, mock_settings):
        """Test decoding an expired token returns None."""
        mock_settings.jwt_secret_key = "test-secret"
        mock_settings.jwt_algorithm = "HS256"
        mock_settings.jwt_access_expire_minutes = -1  # Already expired

        token = create_access_token("user-123", "user@example.com")
        decoded = decode_token(token)

        # jose library returns None for expired tokens by default
        assert decoded is None


class TestIsTokenExpired:
    """Tests for token expiration checking."""

    def test_is_token_expired_not_expired(self):
        """Test that non-expired token returns False."""
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        token_data = TokenData(
            user_id="user-123",
            email="user@example.com",
            exp=future,
            token_type="access",
        )

        assert is_token_expired(token_data) is False

    def test_is_token_expired_expired(self):
        """Test that expired token returns True."""
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        token_data = TokenData(
            user_id="user-123",
            email="user@example.com",
            exp=past,
            token_type="access",
        )

        assert is_token_expired(token_data) is True

    def test_is_token_expired_just_now(self):
        """Test token that just expired."""
        just_past = datetime.now(timezone.utc) - timedelta(seconds=1)
        token_data = TokenData(
            user_id="user-123",
            email="user@example.com",
            exp=just_past,
            token_type="access",
        )

        assert is_token_expired(token_data) is True


class TestTokenData:
    """Tests for TokenData model."""

    def test_token_data_creation(self):
        """Test TokenData model creation."""
        now = datetime.now(timezone.utc)
        data = TokenData(
            user_id="user-123",
            email="user@example.com",
            exp=now,
            token_type="access",
        )

        assert data.user_id == "user-123"
        assert data.email == "user@example.com"
        assert data.exp == now
        assert data.token_type == "access"

    def test_token_data_types(self):
        """Test TokenData accepts different token types."""
        now = datetime.now(timezone.utc)

        access = TokenData(
            user_id="u", email="e@e.com", exp=now, token_type="access"
        )
        refresh = TokenData(
            user_id="u", email="e@e.com", exp=now, token_type="refresh"
        )

        assert access.token_type == "access"
        assert refresh.token_type == "refresh"

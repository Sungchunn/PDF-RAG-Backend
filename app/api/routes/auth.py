"""
Authentication endpoints for user registration and login.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

from app.api.deps import SessionDep, CurrentUserDep
from app.core.auth import create_token_pair, decode_token
from app.services.auth_service import AuthService


router = APIRouter()


# Request/Response Models
class RegisterRequest(BaseModel):
    """Registration request."""

    email: EmailStr
    password: str = Field(..., min_length=8, description="Minimum 8 characters")
    displayName: str | None = None


class LoginRequest(BaseModel):
    """Login request."""

    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    """Token refresh request."""

    refreshToken: str


class TokenResponse(BaseModel):
    """Authentication token response."""

    accessToken: str
    refreshToken: str
    tokenType: str = "bearer"


class UserResponse(BaseModel):
    """User information response."""

    id: str
    email: str
    displayName: str | None
    isActive: bool
    isVerified: bool


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register(request: RegisterRequest, session: SessionDep):
    """
    Register a new user account.

    Returns the created user profile (without tokens - login required).
    """
    auth = AuthService(session)
    try:
        user = await auth.register(
            email=request.email,
            password=request.password,
            display_name=request.displayName,
        )
        return UserResponse(
            id=user.id,
            email=user.email,
            displayName=user.display_name,
            isActive=user.is_active,
            isVerified=user.is_verified,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, session: SessionDep):
    """
    Login with email and password.

    Returns JWT access and refresh tokens.
    """
    auth = AuthService(session)
    tokens = await auth.login(request.email, request.password)

    if not tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    return TokenResponse(
        accessToken=tokens.access_token,
        refreshToken=tokens.refresh_token,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshRequest, session: SessionDep):
    """
    Refresh access token using refresh token.

    Returns new access and refresh tokens.
    """
    token_data = decode_token(request.refreshToken)

    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    if token_data.token_type != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
        )

    # Verify user still exists and is active
    auth = AuthService(session)
    user = await auth.get_user_by_id(token_data.user_id)

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    # Create new token pair
    tokens = create_token_pair(user.id, user.email)
    return TokenResponse(
        accessToken=tokens.access_token,
        refreshToken=tokens.refresh_token,
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(current_user: CurrentUserDep):
    """
    Get current authenticated user's profile.

    Requires valid access token.
    """
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        displayName=current_user.display_name,
        isActive=current_user.is_active,
        isVerified=current_user.is_verified,
    )

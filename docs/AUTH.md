# Authentication and Authorization

This document explains how authentication is implemented and how API routes enforce access.

## Components

- `app/core/auth.py`: password hashing and JWT creation/validation.
- `app/services/auth_service.py`: user registration and login logic.
- `app/api/routes/auth.py`: `/register`, `/login`, `/refresh`, `/me` endpoints.
- `app/api/deps.py`: `get_current_user` dependency used by protected routes.

## User model

Users are stored in the `users` table with:

- `email` (unique), `password_hash` (bcrypt), `display_name` (optional)
- `is_active`, `is_verified` flags
- `created_at`, `updated_at`

## Password hashing

- Passwords are hashed using `passlib` with the `bcrypt` scheme.
- Login validates with `verify_password` against `password_hash`.

## JWT tokens

Tokens are stateless, signed with `settings.jwt_secret_key` and `settings.jwt_algorithm`.

Payload fields:

- `sub`: user ID
- `email`: user email
- `exp`: expiration timestamp
- `type`: `access` or `refresh`
- `iat`: issued-at timestamp

Expiration settings (defaults):

- Access tokens: `jwt_access_expire_minutes` (30 minutes)
- Refresh tokens: `jwt_refresh_expire_days` (7 days)

## Endpoint flows

### Register

- `POST /api/auth/register` creates a user record.
- Returns the user profile (no tokens); client must login to receive tokens.

### Login

- `POST /api/auth/login` validates email + password.
- Returns `accessToken` + `refreshToken` pair.

### Refresh

- `POST /api/auth/refresh` validates the refresh token and user status.
- Issues a new access + refresh token pair.

### Current user

- `GET /api/auth/me` returns the current user's profile.
- Requires a valid access token.

## Route protection

`get_current_user` performs:

1. JWT decode and signature verification.
2. Expiration check (`is_token_expired`).
3. Token type check (`access` only).
4. User lookup and `is_active` check.

If any check fails, the request receives `401 Unauthorized`.

## Notes and limitations

- Refresh tokens are not persisted or revoked server-side; invalidation relies on expiry.
- `is_verified` is stored but not enforced in current routes.
- There is no password reset flow yet.

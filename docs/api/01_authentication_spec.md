# Authentication API Specification

This document details the API endpoints for user authentication and management.

**Base URL:** `/api/v1/auth` (Assuming `/api/v1` prefix based on project structure standards, though verified as just `/auth` relative to the router in `auth.py`. Standard convention usually implies a prefix, but the specification below will denote the relative path from the auth router mount point.)

## 1. Register User

Creates a new user account.

- **Endpoint:** `POST /register`
- **Auth Required:** No

### Request Body
| Field | Type | Required | Description | Constraints |
|---|---|---|---|---|
| `email` | string | Yes | User's email address | Valid email format |
| `password` | string | Yes | Account password | Min 8 characters |
| `displayName` | string | No | User's public display name | |

**Example Request:**
```json
{
  "email": "jane.doe@example.com",
  "password": "securePassword123!",
  "displayName": "Jane Doe"
}
```

### Response (201 Created)
Returns the created user profile. Note that registration does *not* automatically log the user in; a subsequent login call is required to obtain tokens.

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "jane.doe@example.com",
  "displayName": "Jane Doe",
  "isActive": true,
  "isVerified": false
}
```

### Errors
- **400 Bad Request:** Email already registered or invalid input.

---

## 2. Login

Authenticates a user and returns a pair of JWT tokens.

- **Endpoint:** `POST /login`
- **Auth Required:** No

### Request Body
| Field | Type | Required | Description |
|---|---|---|---|
| `email` | string | Yes | User's email |
| `password` | string | Yes | User's password |

**Example Request:**
```json
{
  "email": "jane.doe@example.com",
  "password": "securePassword123!"
}
```

### Response (200 OK)
Returns an Access Token (short-lived) and a Refresh Token (long-lived).

```json
{
  "accessToken": "ey...<jwt_access_token>...",
  "refreshToken": "ey...<jwt_refresh_token>...",
  "tokenType": "bearer"
}
```

### Errors
- **401 Unauthorized:** Invalid email or password.

---

## 3. Refresh Token

Obtains a new Access Token using a valid Refresh Token.

- **Endpoint:** `POST /refresh`
- **Auth Required:** No (Token passed in body)

### Request Body
| Field | Type | Required | Description |
|---|---|---|---|
| `refreshToken` | string | Yes | The refresh token obtained during login |

**Example Request:**
```json
{
  "refreshToken": "ey...<jwt_refresh_token>..."
}
```

### Response (200 OK)
Returns a *new* pair of tokens.

```json
{
  "accessToken": "ey...<new_access_token>...",
  "refreshToken": "ey...<new_refresh_token>...",
  "tokenType": "bearer"
}
```

### Errors
- **401 Unauthorized:** Invalid or expired refresh token, or user is inactive.

---

## 4. Get Current User Profile

Retrieves the profile of the currently authenticated user.

- **Endpoint:** `GET /me`
- **Auth Required:** Yes (Bearer Token)

### Headers
| Header | Value | Description |
|---|---|---|
| `Authorization` | `Bearer <accessToken>` | Valid access token |

### Response (200 OK)
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "jane.doe@example.com",
  "displayName": "Jane Doe",
  "isActive": true,
  "isVerified": false
}
```

### Errors
- **401 Unauthorized:** Missing or invalid access token.

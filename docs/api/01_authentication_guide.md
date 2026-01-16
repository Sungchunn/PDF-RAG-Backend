# Authentication Guide

This guide explains how authentication works in the PDF-RAG system. We use a **JWT (JSON Web Token)** based flow with separate **Access** and **Refresh** tokens to balance security and user experience.

## Overview

The system uses a standard OAuth2-style Bearer token implementation.
1.  **Access Token:** Short-lived (e.g., 30 minutes). Used to authorize API requests.
2.  **Refresh Token:** Long-lived (e.g., 7 days). Used to get new access tokens when the old one expires.

## Key Concepts

### 1. Registration vs. Login
- **Registration** (`/register`) creates the record in the database. It does *not* return tokens.
- **Login** (`/login`) validates credentials and mints the tokens.
- *Why?* This separation allows for email verification steps (if added later) before granting access.

### 2. The Token Lifecycle
Clients should store the Refresh Token securely (e.g., `HttpOnly` cookie or secure storage) and the Access Token in memory or a short-term store.

#### Flow Diagram: Login & Access

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant DB

    Note over Client, API: 1. User Enters Credentials
    Client->>API: POST /auth/login (email, password)
    API->>DB: Fetch User Hash
    DB-->>API: User Data
    API->>API: Verify Password Hash
    
    alt Invalid Credentials
        API-->>Client: 401 Unauthorized
    else Valid Credentials
        API->>API: Mint Access Token (30m)
        API->>API: Mint Refresh Token (7d)
        API-->>Client: 200 OK (Tokens)
    end

    Note over Client, API: 2. Accessing Protected Data
    Client->>API: GET /auth/me (Header: Bearer AccessToken)
    API->>API: Validate Token Signature & Expiry
    API-->>Client: 200 OK (User Data)
```

### 3. Handling Token Expiration

When the **Access Token** expires, the API returns a `401 Unauthorized`. The client should then use the **Refresh Token** to get a new pair without forcing the user to log in again.

#### Flow Diagram: Refreshing Tokens

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant DB

    Client->>API: GET /protected-resource
    API-->>Client: 401 Unauthorized (Token Expired)

    Note over Client: Detect 401 & has Refresh Token
    
    Client->>API: POST /auth/refresh (refreshToken)
    API->>API: Verify Refresh Token Signature & Expiry
    API->>DB: Check User Active Status
    
    alt Token Invalid / User Banned
        API-->>Client: 401 Unauthorized
        Note over Client: Redirect to Login
    else Valid
        API->>API: Mint NEW Access Token
        API->>API: Mint NEW Refresh Token
        API-->>Client: 200 OK (New Tokens)
        
        Note over Client: Retry Original Request
        Client->>API: GET /protected-resource (New Access Token)
        API-->>Client: 200 OK
    end
```

## Security Implementation Details

- **Hashing:** Passwords are hashed using `bcrypt` via the `passlib` library.
- **JWT Library:** We use `python-jose` for encoding and decoding tokens.
- **Statelessness:** Access tokens are stateless. The server validates them by checking the signature, not by looking up a session in the DB.
- **Revocation:** Refresh tokens check the user's status (`is_active`) in the DB. If a user is banned, their refresh token will stop working, effectively revoking access once the short-lived access token expires.

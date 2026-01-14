-- Add user authentication tables and user ownership to documents.
-- Enables multi-tenant document collections.

CREATE TABLE users (
  id text PRIMARY KEY,
  email text NOT NULL UNIQUE,
  password_hash text NOT NULL,
  display_name text,
  is_active boolean NOT NULL DEFAULT true,
  is_verified boolean NOT NULL DEFAULT false,
  created_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL
);

-- Add user ownership to documents
ALTER TABLE documents ADD COLUMN user_id text;

-- Add user ownership to conversations
ALTER TABLE conversations ADD COLUMN user_id text;

-- Indexes for efficient lookups
CREATE INDEX idx_users_email ON users (email);
CREATE INDEX idx_documents_user_id ON documents (user_id, uploaded_at);
CREATE INDEX idx_conversations_user_id ON conversations (user_id);

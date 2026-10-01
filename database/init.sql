-- DocuVortex Supabase PostgreSQL Schema Initialization

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- 1. Users Table
CREATE TABLE IF NOT EXISTS users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username      TEXT UNIQUE,
    email         TEXT UNIQUE,
    api_key_hash  TEXT,
    name          TEXT,
    is_verified   BOOLEAN NOT NULL DEFAULT true,
    is_active     BOOLEAN NOT NULL DEFAULT true,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 2. User Sessions Table
CREATE TABLE IF NOT EXISTS user_sessions (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             UUID REFERENCES users(id) ON DELETE CASCADE,
    refresh_token_hash  TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at          TIMESTAMPTZ NOT NULL,
    last_seen_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 3. Sessions Metadata Table
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    user_id     UUID REFERENCES users(id) ON DELETE CASCADE,
    title       TEXT DEFAULT 'New Chat',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 4. Messages Table (Persistent Chat History)
CREATE TABLE IF NOT EXISTS messages (
    id          BIGSERIAL PRIMARY KEY,
    session_id  TEXT REFERENCES sessions(session_id) ON DELETE CASCADE,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    attachments JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 5. Attachments Table (Associated Workspace Documents / Videos)
CREATE TABLE IF NOT EXISTS attachments (
    id          BIGSERIAL PRIMARY KEY,
    session_id  TEXT REFERENCES sessions(session_id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    collection  TEXT NOT NULL,
    source_type TEXT NOT NULL,
    extra       JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (session_id, collection)
);

-- 6. Session Summaries Table
CREATE TABLE IF NOT EXISTS session_summaries (
    session_id          TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
    summary             TEXT NOT NULL,
    summarized_through  TIMESTAMP NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for performance
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email) WHERE email IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users(username) WHERE username IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_users_api_key_hash ON users(api_key_hash);
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires ON user_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_attachments_session_id ON attachments(session_id);

-- Default guest account
INSERT INTO users (id, username, email, name, is_verified, is_active)
VALUES ('00000000-0000-0000-0000-000000000001', 'guest', 'guest@docuvortex.local', 'Guest User', true, true)
ON CONFLICT (id) DO UPDATE
SET username = 'guest', email = 'guest@docuvortex.local', is_active = true;

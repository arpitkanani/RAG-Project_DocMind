-- DocMind Postgres schema
-- Mirrors the current JSON-file structure in memory_manager.py 1:1,
-- so MemoryManager's public methods keep the same behavior once ported.

CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- for gen_random_uuid()

-- ============================================================
-- USERS  (Option A: API key auth, no login UI)
-- ============================================================
CREATE TABLE IF NOT EXISTS users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name          TEXT NOT NULL,
    api_key_hash  TEXT NOT NULL UNIQUE,   -- SHA-256 hex digest, never the plaintext key
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_active     BOOLEAN NOT NULL DEFAULT true
);

-- Fast lookup on every request: hash the incoming key, find the row
CREATE INDEX IF NOT EXISTS idx_users_api_key_hash ON users(api_key_hash);


-- ============================================================
-- SESSIONS  (replaces session_state_{id}.json)
-- ============================================================
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,           -- keep existing 8-char id format from app.py
    user_id      UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title        TEXT NOT NULL DEFAULT 'New Chat',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at DESC);


-- ============================================================
-- MESSAGES  (replaces chat_history_{id}.json)
-- ============================================================
CREATE TABLE IF NOT EXISTS messages (
    id            BIGSERIAL PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    role          TEXT NOT NULL CHECK (role IN ('human', 'ai')),
    content       TEXT NOT NULL,
    attachments   JSONB,                     -- only set on human messages, same shape as before
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);

-- This index is what makes the 7-day pruning query fast instead of
-- scanning every message every time (unlike the old JSON-file loop).
CREATE INDEX IF NOT EXISTS idx_messages_session_created
    ON messages(session_id, created_at DESC);


-- ============================================================
-- ATTACHMENTS  (replaces the "attachments" list inside session_state json)
-- ============================================================
CREATE TABLE IF NOT EXISTS attachments (
    id            BIGSERIAL PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    name          TEXT NOT NULL,
    collection    TEXT NOT NULL,             -- Qdrant collection name, unchanged
    source_type   TEXT NOT NULL,             -- "doc" | "youtube", same as today
    extra         JSONB,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (session_id, collection)          -- mirrors the dedupe-by-collection logic in add_attachment()
);

CREATE INDEX IF NOT EXISTS idx_attachments_session_id ON attachments(session_id);


-- ============================================================
-- QUERY CACHE — intentionally NOT created here.
-- It needs the pgvector extension (CREATE EXTENSION vector;) and a
-- vector column sized to match your embedding model's dimensions.
-- We'll add this table in its own migration during the caching phase,
-- once we've confirmed the embedding model (see embedding-model step).
-- ============================================================
-- DocuVortex Supabase PostgreSQL Schema Initialization

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- 1. Users Table
CREATE TABLE IF NOT EXISTS users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username      TEXT UNIQUE,
    email         TEXT UNIQUE,
    api_key_hash  TEXT,
    password_hash TEXT,
    role          TEXT NOT NULL DEFAULT 'user',
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
    id              BIGSERIAL PRIMARY KEY,
    session_id      TEXT REFERENCES sessions(session_id) ON DELETE CASCADE,
    name            TEXT NOT NULL,
    collection      TEXT NOT NULL,
    source_type     TEXT NOT NULL,
    embedding_model TEXT NOT NULL DEFAULT 'text-embedding-3-small',
    vector_dim      INTEGER NOT NULL DEFAULT 1536,
    extra           JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (session_id, collection)
);

-- 5b. User Settings Table
CREATE TABLE IF NOT EXISTS user_settings (
    user_id                   UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    active_embedding_provider TEXT NOT NULL DEFAULT 'azure',
    updated_at                TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 6. Session Summaries Table
CREATE TABLE IF NOT EXISTS session_summaries (
    session_id          TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
    summary             TEXT NOT NULL,
    summarized_through  TIMESTAMP NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 7. Deployed Custom Business Bots
CREATE TABLE IF NOT EXISTS deployed_bots (
    bot_id              TEXT PRIMARY KEY,
    admin_id            UUID REFERENCES users(id) ON DELETE CASCADE,
    business_name       TEXT NOT NULL,
    persona_name        TEXT NOT NULL,
    business_summary    TEXT NOT NULL DEFAULT '',
    website_url         TEXT,
    website_data        JSONB DEFAULT '{}'::jsonb,
    pdf_attachments     JSONB DEFAULT '[]'::jsonb,
    finalized_blueprint JSONB DEFAULT '{}'::jsonb,
    dynamic_prompt      TEXT NOT NULL DEFAULT '',
    initial_greeting    TEXT NOT NULL DEFAULT '',
    booking_contact     TEXT DEFAULT '',
    status              TEXT NOT NULL DEFAULT 'active',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 8. Customer Bot Chat Sessions
CREATE TABLE IF NOT EXISTS bot_chat_sessions (
    session_id          TEXT PRIMARY KEY,
    bot_id              TEXT REFERENCES deployed_bots(bot_id) ON DELETE CASCADE,
    customer_id         UUID REFERENCES users(id) ON DELETE SET NULL,
    customer_name       TEXT NOT NULL DEFAULT 'Customer',
    turn_count          INTEGER NOT NULL DEFAULT 0,
    conversation_summary TEXT DEFAULT '',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 9. Bot Chat Messages
CREATE TABLE IF NOT EXISTS bot_messages (
    id                  BIGSERIAL PRIMARY KEY,
    session_id          TEXT REFERENCES bot_chat_sessions(session_id) ON DELETE CASCADE,
    role                TEXT NOT NULL,
    content             TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 10. Bot Customer Inquiries & Appointment Leads
CREATE TABLE IF NOT EXISTS bot_leads (
    id                  BIGSERIAL PRIMARY KEY,
    bot_id              TEXT REFERENCES deployed_bots(bot_id) ON DELETE CASCADE,
    customer_username   TEXT NOT NULL,
    customer_name       TEXT,
    customer_email      TEXT,
    customer_phone      TEXT,
    session_id          TEXT,
    inquiry_type        TEXT NOT NULL DEFAULT 'appointment',
    message_excerpt     TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for performance
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email) WHERE email IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users(username) WHERE username IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_users_api_key_hash ON users(api_key_hash);
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires ON user_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user_updated ON sessions(user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_messages_session_created ON messages(session_id, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_messages_session_role_created ON messages(session_id, role, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_attachments_session_id ON attachments(session_id);
CREATE INDEX IF NOT EXISTS idx_attachments_session_created ON attachments(session_id, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_deployed_bots_admin ON deployed_bots(admin_id);
CREATE INDEX IF NOT EXISTS idx_deployed_bots_status ON deployed_bots(status);
CREATE INDEX IF NOT EXISTS idx_bot_chat_sessions_bot ON bot_chat_sessions(bot_id);
CREATE INDEX IF NOT EXISTS idx_bot_chat_sessions_user_lookup ON bot_chat_sessions(customer_id, bot_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_bot_chat_sessions_name_lookup ON bot_chat_sessions(customer_name, bot_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_bot_messages_session ON bot_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_bot_leads_bot ON bot_leads(bot_id);
CREATE INDEX IF NOT EXISTS idx_bot_leads_session ON bot_leads(session_id);

-- Default guest account
INSERT INTO users (id, username, email, name, is_verified, is_active)
VALUES ('00000000-0000-0000-0000-000000000001', 'guest', 'guest@docuvortex.local', 'Guest User', true, true)
ON CONFLICT (id) DO UPDATE
SET username = 'guest', email = 'guest@docuvortex.local', is_active = true;

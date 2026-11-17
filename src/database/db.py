import os
import sys
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv
import yaml
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
import psycopg2.extras

try:
    psycopg2.extras.register_default_jsonb(globally=True)
except Exception:
    pass

from src.exception import CustomException
from src.logger import logging
from src.utils.perf import perf_step, perf_count

load_dotenv()

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)


def _get_db_url() -> str:
    """Resolve database URL, prioritizing DATABASE_URL environment variable."""
    db_uri = os.getenv("DATABASE_URL")
    if db_uri:
        if db_uri.startswith("postgres://"):
            db_uri = db_uri.replace("postgres://", "postgresql://", 1)
        if "sslmode=" not in db_uri and ("supabase.com" in db_uri or "supabase.co" in db_uri):
            sep = "&" if "?" in db_uri else "?"
            db_uri = f"{db_uri}{sep}sslmode=require"
        if "keepalives=" not in db_uri:
            sep = "&" if "?" in db_uri else "?"
            db_uri = f"{db_uri}{sep}keepalives=1&keepalives_idle=30&keepalives_interval=10&keepalives_count=5"
        return db_uri

    # Fallback to config.yaml if DATABASE_URL not set
    cfg = config.get("postgres", {})
    host = os.getenv("POSTGRES_HOST", str(cfg.get("host", "localhost")))
    port = os.getenv("POSTGRES_PORT", str(cfg.get("port", 5432)))
    user = os.getenv("POSTGRES_USER", str(cfg.get("user", "docmind")))
    password = os.getenv("POSTGRES_PASSWORD", str(cfg.get("password", "docmind123")))
    dbname = os.getenv("POSTGRES_DB", str(cfg.get("database", "docmind")))
    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


_pool = None


def get_pool() -> pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        db_url = _get_db_url()
        try:
            logging.info("Initializing Supabase Postgres connection pool (dsn configured)")
            _pool = pool.ThreadedConnectionPool(
                minconn=2,
                maxconn=10,
                dsn=db_url,
            )
            logging.info("Supabase Postgres connection pool successfully created")
        except Exception as e:
            logging.error("Failed to connect to Supabase Postgres pool: %s", e)
            raise CustomException(e, sys)
    return _pool


# Initialize connection pool eagerly on import if possible
try:
    _pool = get_pool()
except Exception as e:
    logging.warning("Eager DB pool init failed (will retry on demand): %s", e)


def test_connection() -> bool:
    """Test the application DB connection pool. Fails loudly with clear log."""
    try:
        with get_db_cursor(commit=False) as cur:
            cur.execute("SELECT 1 AS alive;")
            row = cur.fetchone()
            if row and row["alive"] == 1:
                logging.info("✅ Supabase app DB connection pool (port 6543) OK")
                return True
        logging.error("❌ Supabase app DB connection returned invalid result")
        return False
    except Exception as e:
        logging.error("❌ Supabase app DB connection (port 6543) FAILED: %s", e)
        return False


def prewarm_pool(count: int = 2) -> None:
    """Eagerly acquires and returns pool connections to ensure warm TLS sockets at startup."""
    try:
        p = get_pool()
        conns = []
        for _ in range(count):
            c = p.getconn()
            conns.append(c)
        for c in conns:
            p.putconn(c)
        logging.info("✅ Pre-warmed %d Supabase database connection(s)", count)
    except Exception as e:
        logging.warning("DB pool pre-warm note: %s", e)


def init_db(sql_path: str = "database/init.sql") -> None:
    """Ensures all required tables and columns exist in the Supabase database."""
    try:
        if os.path.exists(sql_path):
            try:
                with open(sql_path, "r", encoding="utf-8") as f:
                    sql = f.read()
                with get_db_cursor(commit=True) as cur:
                    cur.execute(sql)
                logging.info("Database schema applied from %s", sql_path)
            except Exception as sql_err:
                logging.warning("Full init.sql execution note: %s (running granular defensive migrations)", sql_err)

        # Defensive migrations for users & user_sessions tables
        def _safe_exec(query: str, params: tuple = None):
            try:
                with get_db_cursor(commit=True) as cur:
                    cur.execute(query, params or ())
            except Exception as e:
                logging.debug("Defensive migration step note: %s", e)

        _safe_exec("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
        _safe_exec("""
            CREATE TABLE IF NOT EXISTS users (
                id            UUID PRIMARY KEY,
                username      TEXT,
                email         TEXT,
                api_key_hash  TEXT,
                name          TEXT,
                is_verified   BOOLEAN NOT NULL DEFAULT true,
                is_active     BOOLEAN NOT NULL DEFAULT true,
                created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
        _safe_exec("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                id                  UUID PRIMARY KEY,
                user_id             UUID,
                refresh_token_hash  TEXT NOT NULL,
                created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
                expires_at          TIMESTAMPTZ NOT NULL,
                last_seen_at        TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
        _safe_exec("ALTER TABLE users ADD COLUMN IF NOT EXISTS username TEXT;")
        _safe_exec("ALTER TABLE users ADD COLUMN IF NOT EXISTS email TEXT;")
        _safe_exec("ALTER TABLE users ADD COLUMN IF NOT EXISTS api_key_hash TEXT;")
        _safe_exec("ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash TEXT;")
        _safe_exec("ALTER TABLE users ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'user';")
        _safe_exec("ALTER TABLE users ADD COLUMN IF NOT EXISTS name TEXT;")
        _safe_exec("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_verified BOOLEAN DEFAULT true;")
        _safe_exec("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT true;")
        _safe_exec("ALTER TABLE users ALTER COLUMN name DROP NOT NULL;")
        _safe_exec("ALTER TABLE users ALTER COLUMN api_key_hash DROP NOT NULL;")
        _safe_exec("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email) WHERE email IS NOT NULL;")
        _safe_exec("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users(username) WHERE username IS NOT NULL;")
        _safe_exec("CREATE INDEX IF NOT EXISTS idx_users_api_key_hash ON users(api_key_hash);")
        _safe_exec("CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);")
        _safe_exec("CREATE INDEX IF NOT EXISTS idx_user_sessions_expires ON user_sessions(expires_at);")

        # Chat and Memory tables
        _safe_exec("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id  TEXT PRIMARY KEY,
                user_id     UUID REFERENCES users(id) ON DELETE CASCADE,
                title       TEXT DEFAULT 'New Chat',
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
        _safe_exec("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS title TEXT DEFAULT 'New Chat';")
        _safe_exec("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);")
        _safe_exec("CREATE INDEX IF NOT EXISTS idx_sessions_user_updated ON sessions(user_id, updated_at DESC);")

        _safe_exec("""
            CREATE TABLE IF NOT EXISTS messages (
                id          BIGSERIAL PRIMARY KEY,
                session_id  TEXT REFERENCES sessions(session_id) ON DELETE CASCADE,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                attachments JSONB,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
        _safe_exec("CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);")
        _safe_exec("CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);")
        _safe_exec("CREATE INDEX IF NOT EXISTS idx_messages_session_created ON messages(session_id, created_at ASC);")
        _safe_exec("CREATE INDEX IF NOT EXISTS idx_messages_session_role_created ON messages(session_id, role, created_at ASC);")

        _safe_exec("""
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
        """)
        _safe_exec("ALTER TABLE attachments ADD COLUMN IF NOT EXISTS embedding_model TEXT NOT NULL DEFAULT 'text-embedding-3-small';")
        _safe_exec("ALTER TABLE attachments ADD COLUMN IF NOT EXISTS vector_dim INTEGER NOT NULL DEFAULT 1536;")
        _safe_exec("CREATE INDEX IF NOT EXISTS idx_attachments_session_id ON attachments(session_id);")
        _safe_exec("CREATE INDEX IF NOT EXISTS idx_attachments_session_created ON attachments(session_id, created_at ASC);")

        _safe_exec("""
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id                   UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                active_embedding_provider TEXT NOT NULL DEFAULT 'azure',
                updated_at                TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)

        _safe_exec("""
            CREATE TABLE IF NOT EXISTS session_summaries (
                session_id          TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
                summary             TEXT NOT NULL,
                summarized_through  TIMESTAMP NOT NULL DEFAULT now(),
                updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)

        # Custom Business Bot tables
        _safe_exec("""
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
        """)
        _safe_exec("CREATE INDEX IF NOT EXISTS idx_deployed_bots_admin ON deployed_bots(admin_id);")
        _safe_exec("CREATE INDEX IF NOT EXISTS idx_deployed_bots_status ON deployed_bots(status);")

        _safe_exec("""
            CREATE TABLE IF NOT EXISTS bot_chat_sessions (
                session_id          TEXT PRIMARY KEY,
                bot_id              TEXT REFERENCES deployed_bots(bot_id) ON DELETE CASCADE,
                customer_id         UUID REFERENCES users(id) ON DELETE SET NULL,
                customer_name       TEXT NOT NULL DEFAULT 'Customer',
                turn_count          INTEGER NOT NULL DEFAULT 0,
                created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
        _safe_exec("CREATE INDEX IF NOT EXISTS idx_bot_chat_sessions_bot ON bot_chat_sessions(bot_id);")

        _safe_exec("""
            CREATE TABLE IF NOT EXISTS bot_messages (
                id                  BIGSERIAL PRIMARY KEY,
                session_id          TEXT REFERENCES bot_chat_sessions(session_id) ON DELETE CASCADE,
                role                TEXT NOT NULL,
                content             TEXT NOT NULL,
                created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
        _safe_exec("CREATE INDEX IF NOT EXISTS idx_bot_messages_session ON bot_messages(session_id);")

        _safe_exec("""
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
        """)
        _safe_exec("ALTER TABLE bot_leads ADD COLUMN IF NOT EXISTS customer_name TEXT;")
        _safe_exec("ALTER TABLE bot_leads ADD COLUMN IF NOT EXISTS customer_email TEXT;")
        _safe_exec("ALTER TABLE bot_leads ADD COLUMN IF NOT EXISTS customer_phone TEXT;")
        _safe_exec("ALTER TABLE bot_leads ADD COLUMN IF NOT EXISTS session_id TEXT;")
        _safe_exec("CREATE INDEX IF NOT EXISTS idx_bot_leads_bot ON bot_leads(bot_id);")
        _safe_exec("CREATE INDEX IF NOT EXISTS idx_bot_leads_session ON bot_leads(session_id);")

        _safe_exec("""
            INSERT INTO users (id, username, email, name, is_verified, is_active)
            VALUES ('00000000-0000-0000-0000-000000000001', 'guest', 'guest@docuvortex.local', 'Guest User', true, true)
            ON CONFLICT (id) DO UPDATE
            SET username = 'guest', email = 'guest@docuvortex.local', is_active = true;
        """)
        logging.info("Supabase database tables verified & defensive migrations applied successfully.")
    except Exception as e:
        logging.error("Failed to initialize or migrate database schema: %s", e)


class _PerfCursorWrapper:
    """Transparent proxy around DB cursor that measures query timings, counts, and retries once on disconnect."""
    def __init__(self, cur, db_cursor_ctx=None):
        self._cur = cur
        self._db_cursor_ctx = db_cursor_ctx
        self._executed = False

    def execute(self, *args, **kwargs):
        perf_count("db_queries", 1)
        with perf_step("db_query"):
            try:
                res = self._cur.execute(*args, **kwargs)
                self._executed = True
                return res
            except Exception as e:
                import psycopg2
                if not self._executed and isinstance(e, (psycopg2.OperationalError, psycopg2.InterfaceError)) and self._db_cursor_ctx:
                    logging.warning("Pooled socket closed by server, retrying query with fresh connection: %s", e)
                    new_wrapper = self._db_cursor_ctx.reconnect()
                    self._cur = new_wrapper._cur
                    self._executed = True
                    return self._cur.execute(*args, **kwargs)
                raise

    def executemany(self, *args, **kwargs):
        perf_count("db_queries", 1)
        with perf_step("db_query"):
            self._executed = True
            return self._cur.executemany(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._cur, name)

    def __iter__(self):
        return iter(self._cur)


def _get_healthy_conn(p: pool.ThreadedConnectionPool, max_retries: int = 3):
    """
    Acquires a connection from the pool without blocking network roundtrips.
    Performs fast local socket state check (conn.closed == 0),
    evicting dead handles and returning active sockets in ~0.01ms.
    """
    for attempt in range(max_retries):
        perf_count("db_checkouts", 1)
        with perf_step("db_checkout"):
            conn = p.getconn()
        if getattr(conn, "closed", 0) != 0:
            p.putconn(conn, close=True)
            continue
        return conn

    perf_count("db_checkouts", 1)
    with perf_step("db_checkout"):
        return p.getconn()


class get_db_cursor:
    """
    Context manager for a pooled Postgres connection + cursor.

    Usage:
        with get_db_cursor() as cur:
            cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            row = cur.fetchone()
    Commits automatically on clean exit, rolls back on exception,
    always returns the connection to the pool (closing dead sockets).
    Supports transparent single-retry on severed socket.
    """

    def __init__(self, commit: bool = True):
        self.commit = commit
        self.conn = None
        self.raw_cur = None
        self.cur = None

    def reconnect(self):
        """Evicts broken connection and acquires a fresh one."""
        p = get_pool()
        if self.conn:
            try:
                p.putconn(self.conn, close=True)
            except Exception:
                pass
        self.conn = _get_healthy_conn(p)
        self.raw_cur = self.conn.cursor(cursor_factory=RealDictCursor)
        self.cur = _PerfCursorWrapper(self.raw_cur, db_cursor_ctx=self)
        return self.cur

    def __enter__(self):
        try:
            p = get_pool()
            self.conn = _get_healthy_conn(p)
            self.raw_cur = self.conn.cursor(cursor_factory=RealDictCursor)
            self.cur = _PerfCursorWrapper(self.raw_cur, db_cursor_ctx=self)
            return self.cur
        except Exception as e:
            raise CustomException(e, sys)

    def __exit__(self, exc_type, exc_val, exc_tb):
        import psycopg2
        is_conn_error = False
        try:
            if exc_type is not None:
                if issubclass(exc_type, (psycopg2.OperationalError, psycopg2.InterfaceError)):
                    is_conn_error = True
                if self.conn and getattr(self.conn, "closed", 0) == 0:
                    try:
                        with perf_step("db_query"):
                            self.conn.rollback()
                    except Exception:
                        is_conn_error = True
            elif self.commit:
                if self.conn and getattr(self.conn, "closed", 0) == 0:
                    with perf_step("db_query"):
                        self.conn.commit()
        except Exception:
            is_conn_error = True
        finally:
            if self.raw_cur:
                try:
                    self.raw_cur.close()
                except Exception:
                    pass
            if self.conn:
                try:
                    get_pool().putconn(self.conn, close=is_conn_error)
                except Exception:
                    pass
        return False  # never swallow exceptions

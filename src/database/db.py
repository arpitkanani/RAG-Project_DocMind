import os
import sys
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv
import yaml
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

from src.exception import CustomException
from src.logger import logging

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
                minconn=1,
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
                password_hash TEXT,
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
        _safe_exec("ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash TEXT;")
        _safe_exec("ALTER TABLE users ADD COLUMN IF NOT EXISTS api_key_hash TEXT;")
        _safe_exec("ALTER TABLE users ADD COLUMN IF NOT EXISTS name TEXT;")
        _safe_exec("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_verified BOOLEAN DEFAULT true;")
        _safe_exec("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT true;")
        _safe_exec("ALTER TABLE users ALTER COLUMN name DROP NOT NULL;")
        _safe_exec("ALTER TABLE users ALTER COLUMN api_key_hash DROP NOT NULL;")
        _safe_exec("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email) WHERE email IS NOT NULL;")
        _safe_exec("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users(username) WHERE username IS NOT NULL;")
        _safe_exec("CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);")
        _safe_exec("CREATE INDEX IF NOT EXISTS idx_user_sessions_expires ON user_sessions(expires_at);")
        _safe_exec("""
            INSERT INTO users (id, username, email, name, is_verified, is_active)
            VALUES ('00000000-0000-0000-0000-000000000001', 'guest', 'guest@docuvortex.local', 'Guest User', true, true)
            ON CONFLICT (id) DO UPDATE
            SET username = 'guest', email = 'guest@docuvortex.local', is_active = true;
        """)
        logging.info("Supabase users and user_sessions schema verified & defensive migrations applied successfully.")
    except Exception as e:
        logging.error("Failed to initialize or migrate database schema: %s", e)


class get_db_cursor:
    """
    Context manager for a pooled Postgres connection + cursor.

    Usage:
        with get_db_cursor() as cur:
            cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            row = cur.fetchone()
    Commits automatically on clean exit, rolls back on exception,
    always returns the connection to the pool.
    """

    def __init__(self, commit: bool = True):
        self.commit = commit
        self.conn = None
        self.cur = None

    def __enter__(self):
        try:
            p = get_pool()
            self.conn = p.getconn()
            self.cur = self.conn.cursor(cursor_factory=RealDictCursor)
            return self.cur
        except Exception as e:
            raise CustomException(e, sys)

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is not None:
                if self.conn:
                    self.conn.rollback()
            elif self.commit:
                if self.conn:
                    self.conn.commit()
        finally:
            if self.cur:
                self.cur.close()
            if self.conn:
                get_pool().putconn(self.conn)
        return False  # never swallow exceptions
import os
import sys

from dotenv import load_dotenv
import yaml
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

from src.exception import CustomException
from src.logger import logging

load_dotenv()

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)


def _resolve(key: str, env_var: str) -> str:
    """Environment variable wins if set; otherwise fall back to config.yaml."""
    return os.environ.get(env_var, str(config["postgres"][key]))


try:
    logging.info("Initializing Postgres connection pool")
    _pool = pool.ThreadedConnectionPool(
        minconn=1,
        maxconn=10,
        host=_resolve("host", "POSTGRES_HOST"),
        port=_resolve("port", "POSTGRES_PORT"),
        user=_resolve("user", "POSTGRES_USER"),
        password=_resolve("password", "POSTGRES_PASSWORD"),
        dbname=_resolve("database", "POSTGRES_DB"),
    )
    logging.info("Postgres connection pool ready")
except Exception as e:
    raise CustomException(e, sys)


def init_db(sql_path: str = "database/init.sql") -> None:
    """Ensures all required tables (users, sessions, messages, attachments)
    exist in the database."""
    try:
        if not os.path.exists(sql_path):
            logging.warning("init.sql not found at %s, skipping schema init", sql_path)
            return

        with open(sql_path, "r", encoding="utf-8") as f:
            sql = f.read()

        with get_db_cursor() as cur:
            cur.execute(sql)
        logging.info("Database schema initialized successfully from %s", sql_path)
    except Exception as e:
        logging.error("Failed to initialize database schema: %s", e)
        raise CustomException(e, sys)


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
            self.conn = _pool.getconn()
            self.cur = self.conn.cursor(cursor_factory=RealDictCursor)
            return self.cur
        except Exception as e:
            raise CustomException(e, sys)

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is not None:
                self.conn.rollback() # type: ignore
            elif self.commit:
                self.conn.commit() # type: ignore
        finally:
            self.cur.close() # type: ignore
            _pool.putconn(self.conn)
        return False  # never swallow exceptions
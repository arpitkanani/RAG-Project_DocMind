import asyncio
import os
import warnings

import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

warnings.filterwarnings("ignore", category=UserWarning, module=r"langgraph.*")
warnings.filterwarnings("ignore", message=r".*allowed_objects.*")

from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.components.embedder import Embedder
from src.database.db import get_db_cursor, init_db, test_connection
from src.logger import logging
from src.routers.auth import router as auth_router
from src.routers.collections import router as collections_router
from src.routers.health import router as health_router
from src.routers.pages import router as pages_router
from src.routers.query import router as query_router
from src.routers.sessions import router as sessions_router
from src.routers.upload import router as upload_router
from src.routers.youtube import router as youtube_router

from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import yaml


def _get_checkpointer_conn_uri() -> str:
    """
    Direct connection (port 5432) — REQUIRED for LangGraph checkpointer.
    PgBouncer transaction pooler (port 6543) does not support prepared statements,
    which AsyncPostgresSaver relies on.
    """
    db_uri = os.getenv("LANGGRAPH_DATABASE_URL") or os.getenv("DATABASE_URL")
    if db_uri:
        if db_uri.startswith("postgres://"):
            db_uri = db_uri.replace("postgres://", "postgresql://", 1)
        if "sslmode=" not in db_uri and ("supabase.com" in db_uri or "supabase.co" in db_uri):
            sep = "&" if "?" in db_uri else "?"
            db_uri = f"{db_uri}{sep}sslmode=require"
        return db_uri

    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f).get("postgres", {})

    host = os.getenv("POSTGRES_HOST", str(cfg.get("host", "localhost")))
    port = os.getenv("POSTGRES_PORT", str(cfg.get("port", 5432)))
    user = os.getenv("POSTGRES_USER", str(cfg.get("user", "docmind")))
    password = os.getenv("POSTGRES_PASSWORD", str(cfg.get("password", "docmind123")))
    dbname = os.getenv("POSTGRES_DB", str(cfg.get("database", "docmind")))
    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


async def _run_data_retention_cleanup():
    """Wipes chat and uploaded document records after 3 months (90 days) of inactivity."""
    while True:
        try:
            # Sleep 24 hours between cleanups
            await asyncio.sleep(86400)
            with get_db_cursor(commit=True) as cur:
                cur.execute(
                    """
                    DELETE FROM sessions 
                    WHERE updated_at < now() - interval '90 days'
                    RETURNING session_id
                    """
                )
                deleted = cur.fetchall()
                if deleted:
                    logging.info("Data Retention: Cleaned %d expired sessions (>90 days inactive)", len(deleted))
        except asyncio.CancelledError:
            break
        except Exception as e:
            logging.error("Data retention background cleanup task failed: %s", e)
            await asyncio.sleep(3600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 0. Clean up any leftover temporary session files in root (*.ses, :memory:.ses, etc.)
    try:
        for fname in os.listdir("."):
            if fname.endswith(".ses") or ("memory" in fname and ".ses" in fname):
                try:
                    os.remove(fname)
                    logging.info("Cleaned up orphaned temporary file: %s", fname)
                except Exception as e:
                    logging.warning("Could not remove orphaned file %s: %s", fname, e)
    except Exception as e:
        logging.warning("Error checking orphaned session files: %s", e)

    # 1. Test app connection pool & initialize tables
    try:
        init_db()
        test_connection()
    except Exception as e:
        logging.error("Could not auto-initialize DB tables: %s", e)

    Embedder()

    # 2. Setup PostgreSQL Checkpointer Pool on Supabase direct connection (port 5432)
    db_uri = _get_checkpointer_conn_uri()
    logging.info("Initializing LangGraph checkpointer connection to Supabase...")

    # Start data retention background worker
    cleanup_task = asyncio.create_task(_run_data_retention_cleanup())

    try:
        async with AsyncConnectionPool(conninfo=db_uri, max_size=20, kwargs={"autocommit": True}) as pool:
            checkpointer = AsyncPostgresSaver(pool)
            await checkpointer.setup()
            app.state.checkpointer = checkpointer
            logging.info("✅ Supabase LangGraph checkpointer (port 5432) ready")
            yield
    except Exception as e:
        logging.error("❌ Failed to initialize AsyncPostgresSaver checkpointer pool: %s", e)
        app.state.checkpointer = None
        yield
    finally:
        cleanup_task.cancel()
        logging.info("DocuVortex API shut down.")


app = FastAPI(
    title="DocuVortex API",
    description="AI Document Intelligence System",
    version="2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directory is mounted at /static from templates/static
static_dir = "templates/static" if os.path.exists("templates/static") else "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Register modular routers
app.include_router(auth_router)
app.include_router(health_router)
app.include_router(pages_router)
app.include_router(query_router)
app.include_router(upload_router)
app.include_router(youtube_router)
app.include_router(collections_router)
app.include_router(sessions_router)
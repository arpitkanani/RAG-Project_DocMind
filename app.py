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

# Clean and guarantee LangSmith tracing environment configuration
for _k in [
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGCHAIN_PROJECT",
    "LANGSMITH_ENDPOINT",
    "LANGCHAIN_ENDPOINT",
]:
    _v = os.getenv(_k)
    if _v:
        os.environ[_k] = _v.strip().strip('"').strip("'")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_TRACING"] = "true"

from fastapi import FastAPI, Request
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
from src.routers.settings import router as settings_router
from src.routers.upload import router as upload_router
from src.routers.websocket import router as websocket_router
from src.routers.youtube import router as youtube_router

from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import yaml


def _get_checkpointer_conn_uri() -> str:
    """
    Direct connection (port 5432) — REQUIRED for LangGraph checkpointer.
    PgBouncer transaction pooler (port 6543) does not support prepared statements,
    so the checkpointer must connect directly to PostgreSQL on port 5432.
    """
    db_uri = os.getenv("CHECKPOINTER_DATABASE_URL") or os.getenv("LANGGRAPH_DATABASE_URL")
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

    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f).get("postgres", {})

    host = os.getenv("POSTGRES_HOST", str(cfg.get("host", "localhost")))
    port = os.getenv("POSTGRES_PORT", str(cfg.get("port", 5432)))
    user = os.getenv("POSTGRES_USER", str(cfg.get("user", "docmind")))
    password = os.getenv("POSTGRES_PASSWORD", str(cfg.get("password", "docmind123")))
    dbname = os.getenv("POSTGRES_DB", str(cfg.get("database", "docmind")))
    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


async def _run_data_retention_cleanup():
    """Wipes chat and uploaded document records after 7 days of inactivity."""
    while True:
        try:
            # Sleep 24 hours between cleanups
            await asyncio.sleep(86400)
            from src.components.memory_manager import MemoryManager
            from src.database.db import get_db_cursor
            with get_db_cursor(commit=False) as cur:
                cur.execute("SELECT id FROM users WHERE is_active = true")
                users = cur.fetchall()
            for u in users:
                await MemoryManager.aprune_expired_sessions(user_id=str(u["id"]))
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

    # Diagnostic: Scan recent logs and validate app_new.js syntax
    try:
        import subprocess, shutil
        log_files = [f for f in os.listdir("logs") if f.endswith(".log") and f not in ("latest.log", "latest_scan.log")]
        if log_files:
            log_files.sort(key=lambda x: os.path.getmtime(os.path.join("logs", x)), reverse=True)
            shutil.copyfile(os.path.join("logs", log_files[0]), "logs/latest_scan.log")
            logging.info("Copied most recent log %s to logs/latest_scan.log", log_files[0])
        try:
            for js_f in ["templates/static/js/app_new.js", "templates/static/js/landing.js"]:
                res = subprocess.run(["node", "-c", js_f], capture_output=True, text=True)
                if res.returncode != 0:
                    logging.error("❌ JS SYNTAX ERROR in %s:\n%s", js_f, res.stderr)
                else:
                    logging.info("✅ %s syntax check passed cleanly.", js_f)
        except Exception as e:
            logging.info("Node syntax validator note: %s", e)
    except Exception as e:
        logging.warning("Diagnostic check note: %s", e)

    # 1. Test app connection pool & initialize tables
    try:
        init_db()
        test_connection()
    except Exception as e:
        logging.error("Could not auto-initialize DB tables: %s", e)

    Embedder()

    # LangSmith observability confirmation
    if os.getenv("LANGCHAIN_TRACING_V2") == "true" or os.getenv("LANGSMITH_TRACING") == "true":
        ls_project = os.getenv("LANGCHAIN_PROJECT") or os.getenv("LANGSMITH_PROJECT") or "DocuVortex-Prod"
        logging.info("✅ LangSmith Tracing active & connected | Project: %s", ls_project)
    else:
        logging.warning("⚠️ LangSmith Tracing is inactive")

    # 2. Setup PostgreSQL Checkpointer Pool on Supabase direct connection (port 5432)
    db_uri = _get_checkpointer_conn_uri()
    logging.info("Initializing LangGraph checkpointer connection to Supabase...")

    # Start data retention background worker (7 days retention)
    cleanup_task = asyncio.create_task(_run_data_retention_cleanup())

    try:
        async with AsyncConnectionPool(
            conninfo=db_uri,
            min_size=0,
            max_size=5,
            max_idle=60.0,
            max_lifetime=300.0,
            reconnect_timeout=15.0,
            check=AsyncConnectionPool.check_connection,
            kwargs={
                "autocommit": True,
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
            },
        ) as pool:
            checkpointer = AsyncPostgresSaver(pool)
            await checkpointer.setup()
            app.state.checkpointer = checkpointer
            from src.graph.builder import build_rag_graph
            app.state.rag_graph = build_rag_graph(checkpointer=checkpointer)
            logging.info("✅ Supabase LangGraph checkpointer (port 5432) ready & attached to StateGraph")
            yield
    except Exception as e:
        logging.error("❌ Failed to initialize AsyncPostgresSaver checkpointer pool: %s", e)
        app.state.checkpointer = None
        from src.graph.builder import build_rag_graph
        app.state.rag_graph = build_rag_graph(checkpointer=None)
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

cors_origins_raw = os.getenv("CORS_ORIGINS", "*").strip()
if cors_origins_raw == "*":
    cors_origins = ["*"]
else:
    cors_origins = [o.strip() for o in cors_origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def no_cache_static_middleware(request: Request, call_next):
    """Ensures static assets and HTML templates are never cached with 304 in development."""
    response = await call_next(request)
    if request.url.path.startswith("/static/") or request.url.path in ("/", "/app", "/home", "/landing"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.post("/api/telemetry/log", include_in_schema=False)
async def log_client_telemetry(request: Request):
    """Captures frontend JavaScript errors, unhandled rejections, and milestone pings."""
    try:
        data = await request.json()
        log_type = str(data.get("type", "info")).lower()
        msg = data.get("message") or data.get("msg") or str(data)
        file_info = f" [{data.get('file', 'client')}:{data.get('line', '?')}:{data.get('col', '?')}]" if data.get("line") else ""
        stack = f"\n    Stack: {data.get('stack')}" if data.get("stack") else ""
        if log_type in ("error", "exception", "unhandledrejection"):
            logging.error("🔴 [CLIENT ERROR] %s%s%s", msg, file_info, stack)
        elif log_type == "warn":
            logging.warning("🟡 [CLIENT WARN] %s%s", msg, file_info)
        else:
            logging.debug("📱 [CLIENT LOG] %s%s", msg, file_info)
    except Exception as e:
        raw = await request.body()
        logging.warning("📱 [CLIENT LOG RAW] %s (err: %s)", raw, e)
    return {"status": "ok"}


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
app.include_router(settings_router)
app.include_router(websocket_router)

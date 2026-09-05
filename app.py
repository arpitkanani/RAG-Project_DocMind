import os
import warnings

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

# Suppress LangChain / LangGraph internal pending deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"langgraph.*")
warnings.filterwarnings("ignore", message=r".*allowed_objects.*")

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.components.embedder import Embedder
from src.logger import logging
from src.routers.collections import router as collections_router
from src.routers.health import router as health_router
from src.routers.pages import router as pages_router
from src.routers.query import router as query_router
from src.routers.sessions import router as sessions_router
from src.routers.upload import router as upload_router
from src.routers.youtube import router as youtube_router

load_dotenv()

app = FastAPI(
    title="DocMind API",
    description="Local AI Document Intelligence System",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="templates/static"), name="static")


@app.on_event("startup")
def _on_startup() -> None:
    """
    Initialize database tables and preload the embedding model on server boot.
    """
    try:
        from src.database.db import init_db
        init_db()
    except Exception as e:
        logging.warning("Could not auto-initialize DB tables at startup: %s", e)

    logging.info("Preloading embedding model at startup...")
    Embedder()
    logging.info("Embedding model preloaded, ready to serve requests.")


# Register modular routers
app.include_router(health_router)
app.include_router(pages_router)
app.include_router(query_router)
app.include_router(upload_router)
app.include_router(youtube_router)
app.include_router(collections_router)
app.include_router(sessions_router)
import os
import warnings

import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

import warnings

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

warnings.filterwarnings("ignore", category=UserWarning, module=r"langgraph.*")
warnings.filterwarnings("ignore", message=r".*allowed_objects.*")

from contextlib import asynccontextmanager
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    try:
        from src.database.db import init_db
        init_db()
    except Exception as e:
        logging.warning("Could not auto-initialize DB tables at startup: %s", e)

    logging.info("Preloading embedding model at startup...")
    Embedder()
    logging.info("DocuVortex API ready — embedding model preloaded.")

    yield  # Application runs here

    # ── Shutdown ──
    logging.info("DocuVortex API shutting down.")


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

app.mount("/static", StaticFiles(directory="templates/static"), name="static")

# Register modular routers
app.include_router(health_router)
app.include_router(pages_router)
app.include_router(query_router)
app.include_router(upload_router)
app.include_router(youtube_router)
app.include_router(collections_router)
app.include_router(sessions_router)
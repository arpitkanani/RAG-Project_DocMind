import uuid
from pathlib import Path
from typing import List, Optional
import re
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import yaml
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.auth import get_current_user
from src.components.embedder import Embedder
from src.components.memory_manager import MemoryManager
from src.components.vector_store import VectorStore
from src.exception import CollectionNotFoundError, CustomException, KnowledgeBaseEmptyError
from src.logger import logging
from src.pipelines.ingestion_pipeline import IngestionPipeline
from src.pipelines.qa_pipeline import QAPipeline
from src.utils.file_helper import clean_uploads, validate_file, validate_file_size
from src.utils.job_manager import upload_job_manager
from src.utils.rate_limiter import LLMRateLimitError
from src.utils.youtube_helper import extract_video_id
load_dotenv()

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

TEMPLATES_DIR = Path("templates")

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

@app.get("/favicon.ico", include_in_schema=False)
def _favicon():
    # Browsers request this automatically on every page load. We don't ship
    # an icon file, so just return an empty 204 instead of a 404 -- purely
    # cosmetic, keeps the console/log clean.
    return Response(status_code=204)
 
 
@app.on_event("startup")
def _preload_embedding_model() -> None:
    """
    Load the embedding model into memory once, during server boot, instead
    of paying for it inside the first request that happens to touch
    VectorStore/Embedder (e.g. GET /sessions). This is what was making the
    first request after every reload/refresh feel slow.
    """
    logging.info("Preloading embedding model at startup...")
    Embedder()
    logging.info("Embedding model preloaded, ready to serve requests.")


class QueryRequest(BaseModel):
    """User question payload."""

    query: str
    collection_name: Optional[str] = None
    collection_names: Optional[List[str]] = None
    message_attachments: Optional[List[dict]] = None
    session_id: Optional[str] = "default"


class YouTubeRequest(BaseModel):
    """YouTube URL processing payload."""

    url: str
    session_id: Optional[str] = "default"
    collection_name: Optional[str] = None


class SessionCreateResponse(BaseModel):
    session_id: str = Field(..., description="Unique 8-character session identifier")


def read_template(template_name: str) -> str:
    template_path = TEMPLATES_DIR / template_name
    with template_path.open("r", encoding="utf-8") as file_obj:
        return file_obj.read()


def build_error_response(
    *,
    status_code: int,
    error_code: str,
    message: str,
    extra: Optional[dict] = None,
) -> JSONResponse:
    payload = {"success": False, "error_code": error_code, "message": message}
    if extra:
        payload.update(extra)
    return JSONResponse(status_code=status_code, content=payload)


def normalize_collection_scope(request: QueryRequest) -> List[str] | None:
    if request.collection_names:
        return [name for name in request.collection_names if name]
    if request.collection_name:
        return [request.collection_name]
    return None


def build_collection_name(source_name: str, prefix: str = "doc") -> str:
    stem = Path(source_name or prefix).stem.lower()
    safe_stem = re.sub(r"[^a-z0-9_-]+", "_", stem).strip("_") or prefix
    return f"{prefix}_{safe_stem}_{uuid.uuid4().hex[:8]}"


def resolve_session_scope(
    session_id: str, requested_scope: Optional[List[str]], user_id: str
) -> List[str]:
    if requested_scope:
        return requested_scope

    available_collections = VectorStore().list_collections()
    memory = MemoryManager(session_id=session_id, user_id=user_id)
    memory.cleanup_attachments(available_collections)
    attachments = memory.get_attachment_collections()
    if attachments:
        return attachments

    raise KnowledgeBaseEmptyError(
        "Please upload a document or add a YouTube video first."
    )


def delete_collections(collection_names: List[str]) -> List[str]:
    deleted: List[str] = []
    seen = set()

    for collection_name in collection_names:
        if not collection_name or collection_name in seen:
            continue
        seen.add(collection_name)

        try:
            if VectorStore(collection_name=collection_name).delete_collection():
                deleted.append(collection_name)
        except Exception:
            logging.exception("Failed to delete collection during cleanup: %s", collection_name)

    return deleted


def get_available_collections() -> List[str]:
    try:
        return VectorStore().list_collections()
    except Exception:
        logging.exception("Failed to list available collections for synchronization")
        return []


def create_session_id() -> str:
    session_id = str(uuid.uuid4())[:8]
    logging.info("New session: %s", session_id)
    return session_id


def render_chat_template() -> str:
    try:
        return read_template("home.html")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Frontend template not found")
    except Exception as exc:
        logging.exception("Failed to render app template")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/", response_class=HTMLResponse)
@app.get("/app", response_class=HTMLResponse)
@app.get("/home", response_class=HTMLResponse)
async def chat_app():
    return render_chat_template()


@app.get("/landing", response_class=HTMLResponse)
async def landing_page():
    try:
        return read_template("index.html")
    except FileNotFoundError:
        return """
        <html><body>
            <h1>DocMind API Running</h1>
            <p>Visit <a href="/docs">/docs</a> for API documentation</p>
        </body></html>
        """
    except Exception as exc:
        logging.exception("Failed to render landing page")
        raise HTTPException(status_code=500, detail=str(exc))


def _run_upload_job(
    job_id: str,
    file_bytes: bytes,
    filename: str,
    collection_name: str,
    session_id: str,
    user_id: str,
):
    """Runs in the background (FastAPI BackgroundTasks) after the request
    has already returned a job_id to the client."""
    try:
        pipeline = IngestionPipeline()

        def on_retry(attempt: int, wait_seconds: float):
            upload_job_manager.update_progress(
                job_id,
                f"Rate limited by the embedding service — retrying in {int(wait_seconds)}s "
                f"(attempt {attempt}). This can take a few minutes for large files.",
            )

        result = pipeline.run_from_bytes(
            file_bytes,
            filename,
            collection_name=collection_name,
            on_retry=on_retry,
        )

        if result.get("success") and result.get("collection_name"):
            MemoryManager(session_id=session_id, user_id=user_id).add_attachment(
                name=filename or result["collection_name"],
                collection=result["collection_name"],
                source_type="doc",
            )

        logging.info("File indexed: %s", result.get("collection_name"))
        upload_job_manager.mark_ready(job_id, result)
    except Exception:
        logging.exception("Background upload job failed: %s", job_id)
        upload_job_manager.mark_failed(
            job_id, "Could not process this file. Please try uploading it again."
        )


@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_id: str = Form("default"),
    user_id: str = Depends(get_current_user),
):
    try:
        logging.info("File received: %s | session: %s", file.filename, session_id)

        file_bytes = await file.read()

        if not validate_file(file.filename or ""):
            return build_error_response(
                status_code=400,
                error_code="upload_failed",
                message=f"File type not allowed: {file.filename}",
            )
        if not validate_file_size(file_bytes, file.filename or ""):
            return build_error_response(
                status_code=400,
                error_code="upload_failed",
                message="File too large.",
            )

        collection_name = build_collection_name(file.filename or "document", prefix="doc")
        job_id = upload_job_manager.create_job()

        background_tasks.add_task(
            _run_upload_job,
            job_id,
            file_bytes,
            file.filename,
            collection_name,
            session_id,
            user_id,
        )

        return {"job_id": job_id, "status": "processing"}
    except Exception:
        logging.exception("Upload failed to start")
        return build_error_response(
            status_code=500,
            error_code="server_error",
            message="Server is down. Please try again.",
        )


@app.get("/upload/status/{job_id}")
async def get_upload_status(job_id: str, user_id: str = Depends(get_current_user)):
    job = upload_job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _run_youtube_job(
    job_id: str,
    url: str,
    video_id: str,
    collection_name: str,
    session_id: str,
    user_id: str,
    deleted_duplicates: List[str],
):
    """Runs in the background after the request has already returned a
    job_id -- same pattern as _run_upload_job."""
    try:
        pipeline = IngestionPipeline()
        memory = MemoryManager(session_id=session_id, user_id=user_id)

        def on_retry(attempt: int, wait_seconds: float):
            upload_job_manager.update_progress(
                job_id,
                f"Rate limited by the embedding service — retrying in {int(wait_seconds)}s "
                f"(attempt {attempt}). This can take a few minutes for long videos.",
            )

        result = pipeline.run(url, collection_name=collection_name, on_retry=on_retry)

        if result.get("success") and result.get("collection_name"):
            memory.add_attachment(
                name=f"YouTube {video_id}",
                collection=result["collection_name"],
                source_type="yt",
                extra={"video_id": video_id, "url": url},
            )

        logging.info("YouTube transcript indexed: %s", result.get("collection_name"))
        upload_job_manager.mark_ready(
            job_id, {**result, "replaced_collections": deleted_duplicates, "video_id": video_id}
        )
    except Exception:
        logging.exception("Background YouTube job failed: %s", job_id)
        upload_job_manager.mark_failed(
            job_id, "Could not process this YouTube transcript. Please try again."
        )


@app.post("/youtube")
async def process_youtube(
    request: YouTubeRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
):
    try:
        logging.info("YouTube URL received: %s | session: %s", request.url, request.session_id)

        video_id = extract_video_id(request.url)
        session_id = request.session_id or "default"
        memory = MemoryManager(session_id=session_id, user_id=user_id)
        existing_video_sources = [
            attachment
            for attachment in memory.get_attachments()
            if attachment.get("type") == "yt" and attachment.get("video_id") == video_id
        ]
        deleted_duplicates = delete_collections(
            [item.get("collection", "") for item in existing_video_sources]
        )
        for attachment in existing_video_sources:
            collection = attachment.get("collection")
            if collection:
                memory.remove_attachment(collection)

        collection_name = request.collection_name or build_collection_name(
            video_id,
            prefix="yt",
        )
        job_id = upload_job_manager.create_job()

        background_tasks.add_task(
            _run_youtube_job,
            job_id,
            request.url,
            video_id,
            collection_name,
            session_id,
            user_id,
            deleted_duplicates,
        )

        return {"job_id": job_id, "status": "processing", "video_id": video_id}
    except Exception:
        logging.exception("YouTube processing failed to start")
        return build_error_response(
            status_code=500,
            error_code="server_error",
            message="Server is down. Please try again.",
        )


@app.post("/query")
async def query(
    request: QueryRequest,
    user_id: str = Depends(get_current_user),
):
    try:
        logging.info("Query received: %s...", request.query[:50])

        collection_scope = resolve_session_scope(
            request.session_id or "default",
            normalize_collection_scope(request),
            user_id,
        )
        pipeline = QAPipeline(
            collection_names=collection_scope,
            session_id=request.session_id or "default",
            user_id=user_id,
        )
        result = pipeline.run(
            request.query,
            message_attachments=request.message_attachments,
        )

        logging.info(
            "Query succeeded | session: %s | scope: %s",
            result["session_id"],
            result["collection_scope"],
        )
        return result
    except LLMRateLimitError as exc:
        logging.warning("LLM rate limit hit during query | kind: %s", exc.kind)
        return build_error_response(
            status_code=429,
            error_code=f"llm_rate_limit_{exc.kind}",
            message=exc.message,
        )
    except CollectionNotFoundError as exc:
        logging.exception("Query failed because collection is missing")
        missing = getattr(exc, "missing_collections", [])
        return build_error_response(
            status_code=404,
            error_code="collection_not_found",
            message="This document was removed. Please upload a document to continue.",
            extra={"missing_collections": missing},
        )
    except KnowledgeBaseEmptyError:
        logging.exception("Query failed because the knowledge base is empty")
        return build_error_response(
            status_code=404,
            error_code="knowledge_base_empty",
            message="Please upload a document or add a YouTube video first.",
        )
    except CustomException as exc:
        logging.exception("Query failed with application error")
        return build_error_response(
            status_code=400,
            error_code="query_failed",
            message="I couldn't complete that request right now. Please try again.",
        )
    except Exception as exc:
        logging.exception("Query failed")
        return build_error_response(
            status_code=500,
            error_code="server_error",
            message="Server is down. Please try again.",
        )


@app.get("/collections")
async def list_collections():
    try:
        collections = VectorStore().list_collections()
        return {"collections": collections}
    except Exception as exc:
        logging.exception("Failed to list collections")
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    try:
        logging.info("Deleting collection: %s", collection_name)
        deleted = VectorStore(collection_name=collection_name).delete_collection()
        if not deleted:
            return build_error_response(
                status_code=404,
                error_code="collection_not_found",
                message="Collection not found.",
                extra={"missing_collections": [collection_name]},
            )
        return {"success": True, "deleted_collection": collection_name}
    except Exception as exc:
        logging.exception("Failed to delete collection")
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/sessions/{session_id}/attachments/{collection_name}")
async def delete_attachment(
    session_id: str,
    collection_name: str,
    user_id: str = Depends(get_current_user),
):
    try:
        logging.info(
            "Deleting attachment | session: %s | collection: %s",
            session_id,
            collection_name,
        )
        memory = MemoryManager(session_id=session_id, user_id=user_id)
        removed = memory.remove_attachment(collection_name)
        deleted_collections = delete_collections([collection_name])

        if not removed and not deleted_collections:
            return build_error_response(
                status_code=404,
                error_code="attachment_not_found",
                message="Document not found in this chat.",
            )

        return {
            "success": True,
            "removed_attachment": removed,
            "deleted_collections": deleted_collections,
        }
    except Exception as exc:
        logging.exception("Failed to delete attachment")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/sessions", response_model=SessionCreateResponse)
@app.post("/sessions/new", response_model=SessionCreateResponse)
async def new_session():
    try:
        return {"session_id": create_session_id()}
    except Exception as exc:
        logging.exception("Failed to create session")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/sessions")
async def list_sessions(user_id: str = Depends(get_current_user)):
    try:
        sessions = MemoryManager.list_sessions(
            user_id=user_id,
            valid_collections=get_available_collections(),
        )
        return {"sessions": sessions}
    except Exception as exc:
        logging.exception("Failed to list sessions")
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/sessions")
async def delete_all_sessions(user_id: str = Depends(get_current_user)):
    try:
        logging.info("Clearing all sessions | user: %s", user_id)
        user_sessions = MemoryManager.list_sessions(user_id=user_id)
        user_collections: List[str] = []
        for session in user_sessions:
            memory = MemoryManager(session_id=session["session_id"], user_id=user_id)
            user_collections.extend(memory.get_attachment_collections())

        deleted_collections = delete_collections(user_collections)
        MemoryManager.clear_all(user_id=user_id)
        clean_uploads()
        return {
            "success": True,
            "message": "All sessions cleared",
            "deleted_collections": deleted_collections,
        }
    except Exception as exc:
        logging.exception("Failed to clear sessions")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/sessions/{session_id}")
async def get_session(session_id: str, user_id: str = Depends(get_current_user)):
    try:
        memory = MemoryManager(session_id=session_id, user_id=user_id)
        memory.cleanup_attachments(get_available_collections())
        messages = memory.get_messages_payload()
        attachments = memory.get_attachments()
        title = next(
            (
                message["content"].strip()[:60]
                for message in messages
                if message.get("role") == "human" and message.get("content", "").strip()
            ),
            memory.get_title(),
        )
        return {
            "session_id": session_id,
            "title": title,
            "messages": messages,
            "attachments": attachments,
        }
    except Exception as exc:
        logging.exception("Failed to fetch session %s", session_id)
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user_id: str = Depends(get_current_user)):
    try:
        logging.info("Deleting session: %s", session_id)
        memory = MemoryManager(session_id=session_id, user_id=user_id)
        existing_messages = memory.get_messages_payload()
        existing_attachments = memory.get_attachments()
        if not existing_messages and not existing_attachments:
            return build_error_response(
                status_code=404,
                error_code="session_not_found",
                message="This chat no longer exists.",
            )
        deleted_collections = delete_collections(memory.get_attachment_collections())
        memory.clear()
        return {"success": True, "deleted_collections": deleted_collections}
    except Exception as exc:
        logging.exception("Failed to delete session %s", session_id)
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/memory")
async def clear_memory(user_id: str = Depends(get_current_user)):
    try:
        logging.info("Clearing all memory | user: %s", user_id)
        user_sessions = MemoryManager.list_sessions(user_id=user_id)
        user_collections: List[str] = []
        for session in user_sessions:
            memory = MemoryManager(session_id=session["session_id"], user_id=user_id)
            user_collections.extend(memory.get_attachment_collections())

        deleted_collections = delete_collections(user_collections)
        MemoryManager.clear_all(user_id=user_id)
        clean_uploads()
        return {
            "success": True,
            "message": "All memory cleared",
            "deleted_collections": deleted_collections,
        }
    except Exception as exc:
        logging.exception("Failed to clear memory")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "DocMind API"}
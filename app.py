import uuid
from pathlib import Path
from typing import List, Optional
import re

import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.components.memory_manager import MemoryManager
from src.components.vector_store import VectorStore
from src.exception import CollectionNotFoundError, CustomException, KnowledgeBaseEmptyError
from src.logger import logging
from src.pipelines.ingestion_pipeline import IngestionPipeline
from src.pipelines.qa_pipeline import QAPipeline

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


class QueryRequest(BaseModel):
    """User question payload."""

    query: str
    collection_name: Optional[str] = None
    collection_names: Optional[List[str]] = None
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


def resolve_session_scope(session_id: str, requested_scope: Optional[List[str]]) -> List[str]:
    if requested_scope:
        return requested_scope

    attachments = MemoryManager(session_id=session_id).get_attachment_collections()
    if attachments:
        return attachments

    raise KnowledgeBaseEmptyError(
        "You do not have an uploaded document in this chat. Please upload a document to continue."
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


def create_session_id() -> str:
    session_id = str(uuid.uuid4())[:8]
    logging.info("New session: %s", session_id)
    return session_id


@app.get("/", response_class=HTMLResponse)
async def root():
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


@app.get("/app", response_class=HTMLResponse)
@app.get("/home", response_class=HTMLResponse)
async def chat_app():
    try:
        return read_template("home_new.html")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Frontend template not found")
    except Exception as exc:
        logging.exception("Failed to render app template")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form("default"),
):
    try:
        logging.info("File received: %s | session: %s", file.filename, session_id)

        file_bytes = await file.read()
        pipeline = IngestionPipeline()
        collection_name = build_collection_name(file.filename or "document", prefix="doc")
        result = pipeline.run_from_bytes(
            file_bytes,
            file.filename,  # type: ignore[arg-type]
            collection_name=collection_name,
        )

        if result.get("success") and result.get("collection_name"):
            MemoryManager(session_id=session_id).add_attachment(
                name=file.filename or result["collection_name"],
                collection=result["collection_name"],
                source_type="doc",
            )

        logging.info("File indexed: %s", result.get("collection_name"))
        return result
    except CustomException as exc:
        logging.exception("Upload failed with application error")
        return build_error_response(
            status_code=400,
            error_code="upload_failed",
            message="Could not process this file. Please try uploading it again.",
        )
    except Exception as exc:
        logging.exception("Upload failed")
        return build_error_response(
            status_code=500,
            error_code="server_error",
            message="Server is down. Please try again.",
        )


@app.post("/youtube")
async def process_youtube(request: YouTubeRequest):
    try:
        logging.info("YouTube URL received: %s | session: %s", request.url, request.session_id)

        pipeline = IngestionPipeline()
        collection_name = request.collection_name or build_collection_name(
            request.url,
            prefix="yt",
        )
        result = pipeline.run(request.url, collection_name=collection_name)

        if result.get("success") and result.get("collection_name"):
            MemoryManager(session_id=request.session_id or "default").add_attachment(
                name=result["collection_name"],
                collection=result["collection_name"],
                source_type="yt",
            )

        logging.info("YouTube transcript indexed: %s", result.get("collection_name"))
        return result
    except Exception as exc:
        logging.exception("YouTube processing failed")
        return build_error_response(
            status_code=500,
            error_code="server_error",
            message="Server is down. Please try again.",
        )


@app.post("/query")
async def query(request: QueryRequest):
    try:
        logging.info("Query received: %s...", request.query[:50])

        collection_scope = resolve_session_scope(
            request.session_id or "default",
            normalize_collection_scope(request),
        )
        pipeline = QAPipeline(
            collection_names=collection_scope,
            session_id=request.session_id or "default",
        )
        result = pipeline.run(request.query)

        logging.info(
            "Query succeeded | session: %s | scope: %s",
            result["session_id"],
            result["collection_scope"],
        )
        return result
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
            message="Upload a document first to continue this chat.",
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
async def delete_attachment(session_id: str, collection_name: str):
    try:
        logging.info(
            "Deleting attachment | session: %s | collection: %s",
            session_id,
            collection_name,
        )
        memory = MemoryManager(session_id=session_id)
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
async def list_sessions():
    try:
        sessions = MemoryManager.list_sessions()
        return {"sessions": sessions}
    except Exception as exc:
        logging.exception("Failed to list sessions")
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/sessions")
async def delete_all_sessions():
    try:
        logging.info("Clearing all sessions")
        MemoryManager.clear_all()
        deleted_collections = VectorStore().delete_all_collections()
        return {
            "success": True,
            "message": "All sessions cleared",
            "deleted_collections": deleted_collections,
        }
    except Exception as exc:
        logging.exception("Failed to clear sessions")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    try:
        memory = MemoryManager(session_id=session_id)
        messages = memory.get_messages_payload()
        title = next(
            (
                message["content"].strip()[:60]
                for message in messages
                if message.get("role") == "human" and message.get("content", "").strip()
            ),
            "New Chat",
        )
        return {
            "session_id": session_id,
            "title": title,
            "messages": messages,
            "attachments": memory.get_attachments(),
        }
    except Exception as exc:
        logging.exception("Failed to fetch session %s", session_id)
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    try:
        logging.info("Deleting session: %s", session_id)
        memory = MemoryManager(session_id=session_id)
        deleted_collections = delete_collections(memory.get_attachment_collections())
        memory.clear()
        return {"success": True, "deleted_collections": deleted_collections}
    except Exception as exc:
        logging.exception("Failed to delete session %s", session_id)
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/memory")
async def clear_memory():
    try:
        logging.info("Clearing all memory")
        MemoryManager.clear_all()
        deleted_collections = VectorStore().delete_all_collections()
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

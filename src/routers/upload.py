from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from langsmith import traceable

from src.auth import get_current_user
from src.components.memory_manager import MemoryManager
from src.logger import logging
from src.pipelines.ingestion_pipeline import IngestionPipeline
from src.utils.helpers import build_collection_name, build_error_response
from src.utils.file_helper import validate_file, validate_file_size
from src.utils.job_manager import upload_job_manager

router = APIRouter(tags=["Upload"])


@traceable(
    run_type="chain",
    name="ingestion_upload",
    tags=["ingestion", "upload"],
)
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
        try:
            from langsmith.run_helpers import get_current_run_tree
            run_tree = get_current_run_tree()
            if run_tree:
                run_tree.name = f"Document Ingestion: {filename} [{session_id[:8]}]"
                run_tree.tags = list(set((run_tree.tags or []) + ["ingestion", "upload", f"session:{session_id}", f"user:{user_id}"]))
                run_tree.metadata = {
                    **(run_tree.metadata or {}),
                    "session_id": session_id,
                    "user_id": user_id,
                    "filename": filename,
                    "collection_name": collection_name,
                }
        except Exception:
            pass

        pipeline = IngestionPipeline()

        def on_retry(attempt: int, wait_seconds: float):
            upload_job_manager.update_progress(
                job_id,
                f"Rate limited by the embedding service — retrying in {int(wait_seconds)}s "
                f"(attempt {attempt}). This can take a few minutes for large files.",
            )

        def progress_callback(count: int):
            if count > 0:
                upload_job_manager.update_progress(
                    job_id,
                    f"Processing document... Embedded and stored {count} chunks so far."
                )

        result = pipeline.run_from_bytes(
            file_bytes,
            filename,
            collection_name=collection_name,
            on_retry=on_retry,
            progress_callback=progress_callback,
        )

        if result.get("success") and result.get("collection_name") and result.get("chunks_stored", 0) > 0:
            import re
            from src.components.embedder import Embedder
            raw_doc_name = filename or result["collection_name"]
            clean_doc_name = raw_doc_name.replace("\\", "/")
            if "/" in clean_doc_name:
                clean_doc_name = clean_doc_name.split("/")[-1]
            clean_doc_name = re.sub(r"^(?:data/)?uploads/", "", clean_doc_name, flags=re.IGNORECASE)
            clean_doc_name = re.sub(r"_[a-f0-9]{8}(\.[a-zA-Z0-9]+)$", r"\1", clean_doc_name)

            active_provider = Embedder.get_active_provider(user_id=user_id)
            specs = Embedder.get_model_specs(active_provider)

            MemoryManager(session_id=session_id, user_id=user_id).add_attachment(
                name=clean_doc_name or result["collection_name"],
                collection=result["collection_name"],
                source_type="doc",
                embedding_model=specs["model"],
                vector_dim=specs["dimension"],
            )
            logging.info("File indexed: %s (name: %s, model: %s, dim: %d)", result.get("collection_name"), clean_doc_name, specs["model"], specs["dimension"])
            result["embedding_model"] = specs["model"]
            result["vector_dim"] = specs["dimension"]
            upload_job_manager.mark_ready(job_id, result)
        else:
            err_msg = result.get("error") or "Could not process this file. Please try uploading it again."
            upload_job_manager.mark_failed(job_id, err_msg)
    except Exception as err:
        logging.exception("Background upload job failed: %s", job_id)
        err_str = str(err)
        from src.exception import EmbeddingUnavailableError
        if isinstance(err, EmbeddingUnavailableError) or ("embedding" in err_str.lower() and "unavailable" in err_str.lower()):
            user_msg = "The embedding model is unavailable. You can switch the embedding model in Settings."
        elif "connection" in err_str.lower() or "connect" in err_str.lower() or "6333" in err_str:
            user_msg = "Could not connect to the vector store. Please ensure Qdrant Cloud or local service is reachable."
        else:
            user_msg = "Could not process this file. Please try uploading it again."
        upload_job_manager.mark_failed(job_id, user_msg)


@router.post("/upload")
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

        is_large = len(file_bytes) > 10 * 1024 * 1024
        init_msg = (
            "Processing large document (>10MB)... parsing chunks and generating vector embeddings."
            if is_large
            else "Processing document..."
        )

        job_id = upload_job_manager.submit_job(
            lambda jid: _run_upload_job(
                jid,
                file_bytes,
                file.filename,
                collection_name,
                session_id,
                user_id,
            ),
            initial_message=init_msg,
        )

        return {"job_id": job_id, "status": "processing"}
    except Exception:
        logging.exception("Upload failed to start")
        return build_error_response(
            status_code=500,
            error_code="server_error",
            message="Server is down. Please try again.",
        )


@router.get("/upload/status/{job_id}")
async def get_upload_status(job_id: str, user_id: str = Depends(get_current_user)):
    job = upload_job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from langsmith import traceable

from src.auth import get_current_user
from src.components.memory_manager import MemoryManager
from src.logger import logging
from src.pipelines.ingestion_pipeline import IngestionPipeline
from src.routing_helpers import build_collection_name, build_error_response
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


@router.get("/upload/status/{job_id}")
async def get_upload_status(job_id: str, user_id: str = Depends(get_current_user)):
    job = upload_job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

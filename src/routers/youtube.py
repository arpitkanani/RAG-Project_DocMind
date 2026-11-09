from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends
from langsmith import traceable

from src.auth import get_current_user
from src.components.memory_manager import MemoryManager
from src.logger import logging
from src.pipelines.ingestion_pipeline import IngestionPipeline
from src.utils.helpers import (
    build_collection_name,
    build_error_response,
    delete_collections,
)
from src.schemas import YouTubeRequest
from src.utils.job_manager import upload_job_manager
from src.utils.youtube_helper import extract_video_id

router = APIRouter(tags=["YouTube"])


@traceable(
    run_type="chain",
    name="ingestion_youtube",
    tags=["ingestion", "youtube"],
)
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
        try:
            from langsmith.run_helpers import get_current_run_tree
            run_tree = get_current_run_tree()
            if run_tree:
                run_tree.name = f"YouTube Ingestion: {video_id} [{session_id[:8]}]"
                run_tree.tags = list(set((run_tree.tags or []) + ["ingestion", "youtube", f"session:{session_id}", f"user:{user_id}"]))
                run_tree.metadata = {
                    **(run_tree.metadata or {}),
                    "session_id": session_id,
                    "user_id": user_id,
                    "video_id": video_id,
                    "url": url,
                    "collection_name": collection_name,
                }
        except Exception:
            pass

        pipeline = IngestionPipeline()
        memory = MemoryManager(session_id=session_id, user_id=user_id)

        def on_retry(attempt: int, wait_seconds: float):
            upload_job_manager.update_progress(
                job_id,
                f"Rate limited by the embedding service — retrying in {int(wait_seconds)}s "
                f"(attempt {attempt}). This can take a few minutes for long videos.",
            )

        def progress_callback(count: int):
            if count > 0:
                upload_job_manager.update_progress(
                    job_id,
                    f"Processing transcript... Embedded and stored {count} chunks so far."
                )

        result = pipeline.run(url, collection_name=collection_name, on_retry=on_retry, progress_callback=progress_callback)

        if result.get("success") and result.get("collection_name") and result.get("chunks_stored", 0) > 0:
            from src.components.embedder import Embedder
            active_provider = Embedder.get_active_provider(user_id=user_id)
            specs = Embedder.get_model_specs(active_provider)

            memory.add_attachment(
                name=f"YouTube {video_id}",
                collection=result["collection_name"],
                source_type="yt",
                embedding_model=specs["model"],
                vector_dim=specs["dimension"],
                extra={"video_id": video_id, "url": url},
            )
            logging.info("YouTube transcript indexed: %s (model: %s, dim: %d)", result.get("collection_name"), specs["model"], specs["dimension"])
            result["embedding_model"] = specs["model"]
            result["vector_dim"] = specs["dimension"]
            upload_job_manager.mark_ready(
                job_id, {**result, "replaced_collections": deleted_duplicates, "video_id": video_id}
            )
        else:
            err_msg = result.get("error") or "Could not process this YouTube transcript. Please try again."
            upload_job_manager.mark_failed(job_id, err_msg)
    except Exception as err:
        logging.exception("Background YouTube job failed: %s", job_id)
        from src.exception import EmbeddingUnavailableError
        err_str = str(err)
        if isinstance(err, EmbeddingUnavailableError) or ("embedding" in err_str.lower() and "unavailable" in err_str.lower()):
            user_msg = "The embedding model is unavailable. You can switch the embedding model in Settings."
        else:
            user_msg = "Could not process this YouTube transcript. Please try again."
        upload_job_manager.mark_failed(job_id, user_msg)


@router.post("/youtube")
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

        job_id = upload_job_manager.submit_job(
            lambda jid: _run_youtube_job(
                jid,
                request.url,
                video_id,
                collection_name,
                session_id,
                user_id,
                deleted_duplicates,
            ),
            initial_message="Fetching and processing transcript...",
        )

        return {"job_id": job_id, "status": "processing", "video_id": video_id}
    except Exception:
        logging.exception("YouTube processing failed to start")
        return build_error_response(
            status_code=500,
            error_code="server_error",
            message="Server is down. Please try again.",
        )

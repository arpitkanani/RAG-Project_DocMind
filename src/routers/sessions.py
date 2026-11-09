import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Depends, HTTPException

from src.auth import get_current_user
from src.components.memory_manager import MemoryManager
from src.logger import logging
from src.utils.helpers import (
    adelete_collections,
    aget_available_collections,
    build_error_response,
    create_session_id,
    invalidate_collections_cache,
)
from src.schemas import SessionCreateResponse
from src.utils.file_helper import clean_uploads

router = APIRouter(tags=["Sessions & Memory"])

# In-memory session details cache for sub-5ms chat switching (zero infrastructure cost)
_SESSION_DETAILS_CACHE: Dict[Tuple[str, str], Tuple[float, Any]] = {}
_SESSION_CACHE_TTL_SECONDS = 30.0


def invalidate_session_cache(session_id: Optional[str] = None, user_id: Optional[str] = None):
    """Evicts cached session details on chat updates/deletes."""
    global _SESSION_DETAILS_CACHE
    if session_id and user_id:
        _SESSION_DETAILS_CACHE.pop((user_id, session_id), None)
    elif user_id:
        for k in [k for k in _SESSION_DETAILS_CACHE if k[0] == user_id]:
            _SESSION_DETAILS_CACHE.pop(k, None)
    elif session_id:
        for k in [k for k in _SESSION_DETAILS_CACHE if k[1] == session_id]:
            _SESSION_DETAILS_CACHE.pop(k, None)
    else:
        _SESSION_DETAILS_CACHE.clear()


@router.delete("/sessions/{session_id}/attachments/{collection_name}")
async def delete_attachment(
    session_id: str,
    collection_name: str,
    user_id: str = Depends(get_current_user),
):
    try:
        logging.info("Deleting attachment | session: %s | collection: %s", session_id, collection_name)
        memory = MemoryManager(session_id=session_id, user_id=user_id)
        removed = await memory.aremove_attachment(collection_name)
        deleted_collections = await adelete_collections([collection_name])
        invalidate_collections_cache()
        invalidate_session_cache(session_id=session_id, user_id=user_id)

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


@router.post("/sessions", response_model=SessionCreateResponse)
@router.post("/sessions/new", response_model=SessionCreateResponse)
async def new_session():
    try:
        return {"session_id": create_session_id()}
    except Exception as exc:
        logging.exception("Failed to create session")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/sessions")
async def list_sessions(user_id: str = Depends(get_current_user)):
    try:
        # 1. Automatically prune any sessions older than 7 days, deleting their Qdrant collections & DB records
        expired_sessions = await MemoryManager.aprune_expired_sessions(user_id=user_id)

        # 2. List remaining active sessions in a single fast query
        sessions = await MemoryManager.alist_sessions(user_id=user_id)
        return {
            "sessions": sessions,
            "expired_sessions": expired_sessions,
        }
    except Exception as exc:
        logging.exception("Failed to list sessions")
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/sessions")
async def delete_all_sessions(user_id: str = Depends(get_current_user)):
    try:
        logging.info("Clearing all sessions | user: %s", user_id)
        user_sessions = await MemoryManager.alist_sessions(user_id=user_id)
        user_collections: List[str] = []
        for session in user_sessions:
            memory = MemoryManager(session_id=session["session_id"], user_id=user_id)
            colls = await memory.aget_attachment_collections()
            user_collections.extend(colls)

        deleted_collections = await adelete_collections(user_collections)
        await MemoryManager.aclear_all(user_id=user_id)
        clean_uploads()
        invalidate_collections_cache()
        invalidate_session_cache(user_id=user_id)
        return {
            "success": True,
            "message": "All sessions cleared",
            "deleted_collections": deleted_collections,
        }
    except Exception as exc:
        logging.exception("Failed to clear sessions")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, user_id: str = Depends(get_current_user)):
    try:
        cache_key = (user_id, session_id)
        now = time.monotonic()
        if cache_key in _SESSION_DETAILS_CACHE:
            ts, cached_details = _SESSION_DETAILS_CACHE[cache_key]
            if now - ts < _SESSION_CACHE_TTL_SECONDS:
                return cached_details

        memory = MemoryManager(session_id=session_id, user_id=user_id)

        # Retrieve messages, attachments, and title directly from PostgreSQL (<20ms instant switch)
        details = await memory.aget_session_details(valid_collections=None)
        _SESSION_DETAILS_CACHE[cache_key] = (now, details)
        return details
    except Exception as exc:
        logging.exception("Failed to fetch session %s", session_id)
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user_id: str = Depends(get_current_user)):
    try:
        logging.info("Deleting session: %s | user: %s", session_id, user_id)
        memory = MemoryManager(session_id=session_id, user_id=user_id)
        colls = await memory.aget_attachment_collections()
        await memory.aclear()
        clean_uploads()
        invalidate_collections_cache()
        invalidate_session_cache(session_id=session_id, user_id=user_id)

        # Offload remote Qdrant collection deletions to non-blocking background task (<30ms endpoint response)
        if colls:
            asyncio.create_task(adelete_collections(colls))

        return {"success": True, "session_id": session_id, "deleted_collections": colls}
    except Exception as exc:
        logging.exception("Failed to delete session %s", session_id)
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/memory")
async def clear_memory(user_id: str = Depends(get_current_user)):
    try:
        logging.info("Clearing all memory | user: %s", user_id)
        user_sessions = await MemoryManager.alist_sessions(user_id=user_id)
        user_collections: List[str] = []
        for session in user_sessions:
            memory = MemoryManager(session_id=session["session_id"], user_id=user_id)
            colls = await memory.aget_attachment_collections()
            user_collections.extend(colls)

        await MemoryManager.aclear_all(user_id=user_id)
        clean_uploads()
        invalidate_collections_cache()
        invalidate_session_cache(user_id=user_id)

        # Offload remote Qdrant collection deletions to non-blocking background task
        if user_collections:
            asyncio.create_task(adelete_collections(user_collections))

        return {
            "success": True,
            "message": "All memory cleared",
            "deleted_collections": user_collections,
        }
    except Exception as exc:
        logging.exception("Failed to clear memory")
        raise HTTPException(status_code=500, detail=str(exc))

import asyncio
from datetime import datetime, timezone
import time
from typing import Any, Dict, List, Optional, Tuple

import anyio
# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Depends, HTTPException, Query

from src.auth import get_current_user
from src.components.memory_manager import MemoryManager
from src.database.db import get_db_cursor
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
_SESSION_CACHE_TTL_SECONDS = 60.0


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


@router.get("/sessions/bootstrap")
@router.get("/api/sessions/bootstrap")
async def bootstrap_workspace(
    preferred_session_id: Optional[str] = Query(None),
    user_id: str = Depends(get_current_user),
):
    """
    Consolidated single-roundtrip workspace bootstrapper:
    1. Fetches user identity.
    2. Fetches all sessions for sidebar (ordered by updated_at DESC).
    3. Fetches active/latest session messages and attachments in ONE single database transaction.
    Replaces 3-stage HTTP waterfall (session-status + sessions + default) for instant sub-second boot.
    """
    def _do_bootstrap():
        with get_db_cursor(commit=False) as cur:
            # 1. Fetch user info
            cur.execute("SELECT id, username, email, role FROM users WHERE id = %s LIMIT 1", (user_id,))
            user_row = cur.fetchone() or {}

            # 2. Fetch session list (limit 100)
            cur.execute(
                """
                SELECT session_id, COALESCE(NULLIF(TRIM(title), ''), 'New Chat') AS title, updated_at, created_at
                FROM sessions
                WHERE user_id = %s
                ORDER BY updated_at DESC
                LIMIT 100;
                """,
                (user_id,),
            )
            session_rows = cur.fetchall()
            sessions = []
            for row in session_rows:
                last_active = row["updated_at"] or row["created_at"]
                sessions.append({
                    "session_id": row["session_id"],
                    "title": row["title"],
                    "message_count": 1,
                    "attachment_count": 0,
                    "last_active": last_active.isoformat() if hasattr(last_active, "isoformat") else str(last_active),
                    "last_active_label": last_active.strftime("%b %d, %H:%M") if hasattr(last_active, "strftime") else str(last_active),
                })

            # 3. Active session candidate: preferred session if valid for this user, else first session in list
            active_session_id = None
            if preferred_session_id and any(s["session_id"] == preferred_session_id for s in sessions):
                active_session_id = preferred_session_id
            elif sessions:
                active_session_id = sessions[0]["session_id"]
            else:
                # User has zero sessions. Generate a fresh, isolated session ID for this user.
                new_sid = create_session_id()
                active_session_id = new_sid
                # Persist the new empty session for this user so it belongs strictly to them
                cur.execute(
                    """
                    INSERT INTO sessions (session_id, user_id, title, created_at, updated_at)
                    VALUES (%s, %s, 'New Chat', now(), now())
                    ON CONFLICT (session_id) DO NOTHING;
                    """,
                    (active_session_id, user_id),
                )
                sessions.append({
                    "session_id": active_session_id,
                    "title": "New Chat",
                    "message_count": 0,
                    "attachment_count": 0,
                    "last_active": datetime.now(timezone.utc).isoformat(),
                    "last_active_label": datetime.now(timezone.utc).strftime("%b %d, %H:%M"),
                })

            # Fetch active session record ensuring strict user ownership
            cur.execute(
                "SELECT title, created_at, updated_at FROM sessions WHERE session_id = %s AND user_id = %s LIMIT 1;",
                (active_session_id, user_id),
            )
            s_row = cur.fetchone() or {}

            # Fetch active session messages joining sessions to guarantee strict user isolation
            cur.execute(
                """
                SELECT m.role, m.content, m.attachments, m.created_at
                FROM messages m
                JOIN sessions s ON s.session_id = m.session_id
                WHERE m.session_id = %s AND s.user_id = %s
                ORDER BY m.created_at ASC
                LIMIT 100;
                """,
                (active_session_id, user_id),
            )
            msg_rows = cur.fetchall()
            messages = []
            for r in msg_rows:
                messages.append({
                    "role": r["role"],
                    "content": r["content"],
                    "attachments": r.get("attachments") or [],
                    "timestamp": r["created_at"].isoformat() if hasattr(r["created_at"], "isoformat") else str(r["created_at"]),
                })

            # Fetch active session attachments joining sessions
            cur.execute(
                """
                SELECT a.name, a.collection, a.source_type AS type,
                       COALESCE(a.embedding_model, 'text-embedding-3-small') AS embedding_model,
                       COALESCE(a.vector_dim, 1536) AS vector_dim,
                       a.extra, a.created_at
                FROM attachments a
                JOIN sessions s ON s.session_id = a.session_id
                WHERE a.session_id = %s AND s.user_id = %s
                ORDER BY a.created_at ASC;
                """,
                (active_session_id, user_id),
            )
            att_rows = cur.fetchall()
            attachments = []
            for r in att_rows:
                attachments.append({
                    "name": r["name"],
                    "collection": r["collection"],
                    "type": r.get("type", "file"),
                    "embedding_model": r.get("embedding_model"),
                    "vector_dim": r.get("vector_dim"),
                    "extra": r.get("extra") or {},
                })

            active_session = {
                "session_id": active_session_id,
                "title": s_row.get("title") or (sessions[0]["title"] if sessions else "New Chat"),
                "messages": messages,
                "attachments": attachments,
            }

            return {
                "authenticated": True,
                "user": {
                    "id": str(user_row.get("id", user_id)),
                    "username": user_row.get("username", "User"),
                    "email": user_row.get("email", ""),
                    "role": user_row.get("role", "user"),
                },
                "sessions": sessions,
                "active_session": active_session,
            }

    try:
        data = await anyio.to_thread.run_sync(_do_bootstrap)
        return data
    except Exception as exc:
        logging.exception("Failed to bootstrap workspace")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/sessions")
@router.get("/api/sessions")
async def list_sessions(user_id: str = Depends(get_current_user)):
    try:
        # 1. Offload 7-day expiration cleanup to background task (zero latency on HTTP request)
        asyncio.create_task(MemoryManager.aprune_expired_sessions(user_id=user_id))

        # 2. List remaining active sessions in a single fast indexed query
        sessions = await MemoryManager.alist_sessions(user_id=user_id)
        return {
            "sessions": sessions,
            "expired_sessions": [],
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
        # Strict user ownership verification: session must belong to user_id
        def _check_ownership():
            with get_db_cursor(commit=False) as cur:
                cur.execute("SELECT 1 FROM sessions WHERE session_id = %s AND user_id = %s LIMIT 1;", (session_id, user_id))
                return cur.fetchone() is not None

        is_owner = await anyio.to_thread.run_sync(_check_ownership)
        if not is_owner:
            raise HTTPException(status_code=404, detail="Chat session not found or unauthorized.")

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
    except HTTPException:
        raise
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

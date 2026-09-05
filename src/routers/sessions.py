from typing import List

# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Depends, HTTPException

from src.auth import get_current_user
from src.components.memory_manager import MemoryManager
from src.logger import logging
from src.routing_helpers import (
    adelete_collections,
    aget_available_collections,
    build_error_response,
    create_session_id,
)
from src.schemas import SessionCreateResponse
from src.utils.file_helper import clean_uploads

router = APIRouter(tags=["Sessions & Memory"])


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
        available_colls = await aget_available_collections()
        sessions = await MemoryManager.alist_sessions(
            user_id=user_id,
            valid_collections=available_colls,
        )
        return {"sessions": sessions}
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
        memory = MemoryManager(session_id=session_id, user_id=user_id)
        available_colls = await aget_available_collections()
        await memory.acleanup_attachments(available_colls)
        messages = await memory.aget_messages_payload()
        attachments = await memory.aget_attachments()
        title = next(
            (
                message["content"].strip()[:60]
                for message in messages
                if message.get("role") == "human" and message.get("content", "").strip()
            ),
            await memory.aget_title(),
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


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user_id: str = Depends(get_current_user)):
    try:
        logging.info("Deleting session: %s", session_id)
        memory = MemoryManager(session_id=session_id, user_id=user_id)
        existing_messages = await memory.aget_messages_payload()
        existing_attachments = await memory.aget_attachments()
        if not existing_messages and not existing_attachments:
            return build_error_response(
                status_code=404,
                error_code="session_not_found",
                message="This chat no longer exists.",
            )
        colls = await memory.aget_attachment_collections()
        deleted_collections = await adelete_collections(colls)
        await memory.aclear()
        return {"success": True, "deleted_collections": deleted_collections}
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

        deleted_collections = await adelete_collections(user_collections)
        await MemoryManager.aclear_all(user_id=user_id)
        clean_uploads()
        return {
            "success": True,
            "message": "All memory cleared",
            "deleted_collections": deleted_collections,
        }
    except Exception as exc:
        logging.exception("Failed to clear memory")
        raise HTTPException(status_code=500, detail=str(exc))

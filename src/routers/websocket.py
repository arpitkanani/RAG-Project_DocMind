"""WebSocket router for DocuVortex — /ws/chat
Provides persistent, multiplexed bidirectional streaming across multiple sessions.
"""

import asyncio
import hashlib
import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.components.embedder import Embedder
from src.components.memory_manager import MemoryManager
from src.components.vector_store import VectorStore
from src.database.db import get_db_cursor
from src.exception import (
    CollectionEmbeddingMismatchError,
    CollectionNotFoundError,
    KnowledgeBaseEmptyError,
)
from src.logger import logging
from src.utils.helpers import aresolve_session_scope, normalize_collection_scope
from src.utils.query_manager import QueryManager

router = APIRouter(tags=["WebSocket"])


async def _resolve_ws_user(websocket: WebSocket) -> Optional[str]:
    """Authenticates the WebSocket connection via api_key query param or cookies."""
    api_key = websocket.query_params.get("api_key")
    if not api_key:
        api_key = websocket.cookies.get("api_key")

    if api_key and api_key != "guest":
        key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
        try:
            with get_db_cursor() as cur:
                cur.execute(
                    "SELECT id FROM users WHERE api_key_hash = %s AND is_active = true",
                    (key_hash,),
                )
                row = cur.fetchone()
                if row:
                    return str(row["id"])
        except Exception as exc:
            logging.warning("WebSocket auth DB error: %s", exc)

    token = websocket.cookies.get("access_token")
    if token:
        try:
            from src.auth import decode_access_token, _verify_user_exists
            import anyio
            payload = decode_access_token(token)
            if payload and "sub" in payload:
                user_id = str(payload["sub"])
                user_exists = await anyio.to_thread.run_sync(_verify_user_exists, user_id)
                if user_exists:
                    return user_id
        except Exception as e:
            logging.warning("WebSocket cookie auth error: %s", e)

    return None


@router.websocket("/ws/chat")
async def chat_websocket_endpoint(websocket: WebSocket):
    """Persistent bidirectional WebSocket connection for chat query streaming."""
    user_id = await _resolve_ws_user(websocket)
    if not user_id:
        logging.warning("Rejecting unauthorized WebSocket connection")
        await websocket.close(code=4401, reason="Unauthorized: Valid API key required")
        return

    await websocket.accept()
    logging.info("WebSocket connection established for user: %s", user_id)

    # Active stream reader tasks per session: session_id -> asyncio.Task
    subscribed_tasks: Dict[str, asyncio.Task] = {}

    async def _stream_session_events(session_id: str, queue: asyncio.Queue):
        try:
            while True:
                event = await queue.get()
                # Send frame over WebSocket
                try:
                    await websocket.send_text(json.dumps(event))
                except Exception:
                    break
                if event.get("type") in ("done", "error"):
                    break
        except asyncio.CancelledError:
            pass
        finally:
            active_q = await QueryManager.get_active_query(user_id, session_id)
            if active_q:
                active_q.remove_subscriber(queue)
            subscribed_tasks.pop(session_id, None)

    try:
        while True:
            raw_text = await websocket.receive_text()
            if not raw_text.strip():
                continue

            try:
                msg = json.loads(raw_text)
            except Exception:
                await websocket.send_text(
                    json.dumps({"type": "error", "error_code": "invalid_json", "message": "Invalid JSON frame"})
                )
                continue

            action = msg.get("action", "")

            # ── 1. Heartbeat ping/pong ──
            if action == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                continue

            # ── 2. Cancel in-flight query ──
            elif action == "cancel":
                target_session_id = msg.get("session_id", "default")
                active_q = await QueryManager.get_active_query(user_id, target_session_id)
                if active_q and active_q.task and not active_q.task.done():
                    active_q.task.cancel()
                    await websocket.send_text(
                        json.dumps({
                            "session_id": target_session_id,
                            "type": "error",
                            "error_code": "cancelled",
                            "message": "Query cancelled by user",
                        })
                    )
                continue

            # ── 3. Subscribe to existing running query (after chat switch / reconnect) ──
            elif action == "subscribe":
                target_session_id = msg.get("session_id", "default")
                active_q = await QueryManager.get_active_query(user_id, target_session_id)
                if active_q and not active_q.is_completed:
                    if target_session_id in subscribed_tasks:
                        subscribed_tasks[target_session_id].cancel()
                    q = active_q.add_subscriber()
                    subscribed_tasks[target_session_id] = asyncio.create_task(
                        _stream_session_events(target_session_id, q)
                    )
                else:
                    await websocket.send_text(
                        json.dumps({
                            "session_id": target_session_id,
                            "type": "status",
                            "stage": "idle",
                            "message": "No active generation in progress",
                        })
                    )
                continue

            # ── 4. Launch new query ──
            elif action == "query":
                session_id = msg.get("session_id", "default")
                query_text = (msg.get("query") or "").strip()
                requested_collections = msg.get("collection_names") or []
                message_attachments = msg.get("message_attachments")

                if not query_text:
                    await websocket.send_text(
                        json.dumps({
                            "session_id": session_id,
                            "type": "error",
                            "error_code": "empty_query",
                            "message": "Query text cannot be empty",
                        })
                    )
                    continue

                # Pre-flight embedding dimension check
                active_provider = Embedder.get_active_provider(user_id=user_id)
                active_specs = Embedder.get_model_specs(active_provider)

                try:
                    collection_scope = await aresolve_session_scope(
                        session_id,
                        [c for c in requested_collections if c],
                        user_id,
                    )
                except Exception:
                    collection_scope = []

                if collection_scope:
                    mismatched = []
                    memory = MemoryManager(session_id=session_id, user_id=user_id)
                    attachments = await memory.aget_attachments()
                    att_map = {a["collection"]: a for a in attachments if "collection" in a}

                    for col in collection_scope:
                        att = att_map.get(col)
                        indexed_dim = att.get("vector_dim") if att else None
                        if not indexed_dim:
                            indexed_dim = VectorStore(collection_name=col).get_collection_vector_dimension(col)

                        if indexed_dim and indexed_dim != active_specs["dimension"]:
                            doc_name = att.get("name", col) if att else col
                            indexed_model = att.get("embedding_model") if (att and att.get("embedding_model")) else ("BAAI/bge-small-en-v1.5" if indexed_dim == 384 else "text-embedding-3-small")
                            mismatched.append({
                                "collection": col,
                                "name": doc_name,
                                "indexed_model": indexed_model,
                                "indexed_dim": indexed_dim,
                                "current_model": active_specs["model"],
                                "current_dim": active_specs["dimension"],
                            })

                    if mismatched:
                        first_m = mismatched[0]
                        detail_msg = (
                            f"The document '{first_m['name']}' was indexed using '{first_m['indexed_model']}' ({first_m['indexed_dim']}d), "
                            f"but your currently active model is '{first_m['current_model']}' ({first_m['current_dim']}d). "
                            f"Please switch to '{first_m['indexed_model']}' from Settings or re-upload the document."
                        )
                        await websocket.send_text(
                            json.dumps({
                                "session_id": session_id,
                                "type": "error",
                                "error_code": "collection_needs_reindex",
                                "message": detail_msg,
                                "extra": {"mismatched_collections": mismatched},
                            })
                        )
                        continue

                # Launch shielded detached background generation
                app_graph = getattr(websocket.app.state, "rag_graph", None)
                active_query = await QueryManager.start_query(
                    user_id=user_id,
                    session_id=session_id,
                    query=query_text,
                    collection_scope=collection_scope,
                    message_attachments=message_attachments,
                    app_graph=app_graph,
                )

                # Subscribe this WebSocket to the session's stream
                if session_id in subscribed_tasks:
                    subscribed_tasks[session_id].cancel()

                sub_queue = active_query.add_subscriber()
                subscribed_tasks[session_id] = asyncio.create_task(
                    _stream_session_events(session_id, sub_queue)
                )

    except WebSocketDisconnect:
        logging.info("WebSocket client disconnected | user: %s", user_id)
    except Exception as exc:
        logging.exception("WebSocket loop error: %s", exc)
    finally:
        # Cancel forwarding tasks without terminating background LangGraph execution
        for t in subscribed_tasks.values():
            t.cancel()
        logging.info("Cleaned up WebSocket subscriber tasks for user %s", user_id)

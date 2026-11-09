import asyncio
from datetime import datetime, timedelta, timezone
import json
import sys
from typing import Any, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from psycopg2.extras import Json
import yaml

from src.database.db import get_db_cursor
from src.exception import CustomException
from src.logger import logging

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)


class MemoryManager:
    """Manages multi-session chat history and attachments with sync and async interfaces."""

    def __init__(self, session_id: str = "default", user_id: str = None):
        if not user_id:
            raise CustomException(ValueError("MemoryManager requires a user_id"), sys)
        self.session_id = session_id
        self.user_id = user_id
        self.window_days = config.get("memory", {}).get("window_days", 7)
        self.recent_turns_to_keep = config.get("memory", {}).get("recent_messages_verbatim", 6)

    def _ensure_session(self, cur, title: Optional[str] = None):
        cur.execute(
            """
            INSERT INTO sessions (session_id, user_id, title, created_at, updated_at)
            VALUES (%s, %s, COALESCE(%s, 'New Chat'), now(), now())
            ON CONFLICT (session_id) DO UPDATE
                SET title = COALESCE(%s, sessions.title),
                    updated_at = now()
            """,
            (self.session_id, self.user_id, title, title),
        )

    def save_message(
        self,
        role: str,
        content: str,
        attachments: Optional[List[dict[str, Any]]] = None,
    ):
        try:
            title = content.strip()[:60] if (role == "human" and content.strip()) else None
            attachments_json = (
                Json([item for item in attachments if isinstance(item, dict)])
                if attachments and role == "human"
                else None
            )

            with get_db_cursor() as cur:
                self._ensure_session(cur, title=title)
                cur.execute(
                    """
                    INSERT INTO messages (session_id, role, content, attachments)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (self.session_id, role, content, attachments_json),
                )
                self._prune_expired(cur)

            logging.info("Message saved | role: %s | session: %s", role, self.session_id)
        except Exception as e:
            raise CustomException(e, sys)

    async def asave_message(
        self,
        role: str,
        content: str,
        attachments: Optional[List[dict[str, Any]]] = None,
    ):
        return await asyncio.to_thread(self.save_message, role, content, attachments)

    def get_history(self) -> List[BaseMessage]:
        try:
            payload = self.get_messages_payload()
            if len(payload) <= self.recent_turns_to_keep:
                return self._to_base_messages(payload)

            older = payload[: -self.recent_turns_to_keep]
            recent = payload[-self.recent_turns_to_keep :]
            summary_text = self._get_or_build_summary(older)

            messages: List[BaseMessage] = []
            if summary_text:
                messages.append(
                    SystemMessage(
                        content=f"Summary of earlier parts of this conversation:\n{summary_text}"
                    )
                )
            messages.extend(self._to_base_messages(recent))
            return messages
        except Exception as e:
            raise CustomException(e, sys)

    async def aget_history(self) -> List[BaseMessage]:
        return await asyncio.to_thread(self.get_history)

    @staticmethod
    def _to_base_messages(payload: List[dict[str, Any]]) -> List[BaseMessage]:
        messages: List[BaseMessage] = []
        for msg in payload:
            if msg["role"] == "human":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "ai":
                messages.append(AIMessage(content=msg["content"]))
        return messages

    @staticmethod
    def _ensure_summary_table(cur):
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS session_summaries (
                session_id TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                summarized_through TIMESTAMP NOT NULL,
                updated_at TIMESTAMP DEFAULT now()
            )
            """
        )

    def _get_cached_summary(self, cur) -> Optional[dict]:
        cur.execute(
            "SELECT summary, summarized_through FROM session_summaries WHERE session_id = %s",
            (self.session_id,),
        )
        return cur.fetchone()

    def _save_summary(self, cur, summary: str, summarized_through: datetime):
        cur.execute(
            """
            INSERT INTO session_summaries (session_id, summary, summarized_through, updated_at)
            VALUES (%s, %s, %s, now())
            ON CONFLICT (session_id) DO UPDATE
                SET summary = %s, summarized_through = %s, updated_at = now()
            """,
            (self.session_id, summary, summarized_through, summary, summarized_through),
        )

    @staticmethod
    def _normalize_dt(dt: datetime) -> datetime:
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt

    def _get_or_build_summary(self, older_messages: List[dict[str, Any]]) -> str:
        if not older_messages:
            return ""

        latest_covered = self._normalize_dt(datetime.fromisoformat(older_messages[-1]["timestamp"]))

        with get_db_cursor(commit=False) as cur:
            self._ensure_summary_table(cur)
            cached = self._get_cached_summary(cur)

        if cached and self._normalize_dt(cached["summarized_through"]) >= latest_covered:
            return cached["summary"]

        already_covered_through = (
            self._normalize_dt(cached["summarized_through"]) if cached else None
        )
        to_fold_in = [
            m for m in older_messages
            if already_covered_through is None
            or self._normalize_dt(datetime.fromisoformat(m["timestamp"])) > already_covered_through
        ]

        new_summary = self._summarize_messages(
            to_fold_in, previous_summary=cached["summary"] if cached else ""
        )

        with get_db_cursor() as cur:
            self._ensure_summary_table(cur)
            self._save_summary(cur, new_summary, latest_covered)

        return new_summary

    @staticmethod
    def _summarize_messages(messages: List[dict[str, Any]], previous_summary: str = "") -> str:
        from src.chains.qa_chain import _build_llm
        from src.utils.rate_limiter import llm_rate_limiter

        conversation_text = "\n".join(
            f"{'User' if m['role'] == 'human' else 'Assistant'}: {m['content']}"
            for m in messages
        )

        prompt = (
            "Summarize the following part of a conversation in 3-5 short sentences. "
            "Keep only facts, decisions, and topics that matter for later context.\n\n"
        )
        if previous_summary:
            prompt += f"Existing summary:\n{previous_summary}\n\n"
        prompt += f"Conversation to fold in:\n{conversation_text}\n\nUpdated summary:"

        from src.utils.helpers import extract_text

        llm = _build_llm()
        llm_rate_limiter.acquire()
        response = llm.invoke(prompt, config={"tags": ["memory_summary"]})
        return extract_text(getattr(response, "content", response))

    def get_messages_payload(self) -> List[dict[str, Any]]:
        try:
            with get_db_cursor(commit=False) as cur:
                cur.execute(
                    """
                    SELECT m.role, m.content, m.attachments, m.created_at
                    FROM messages m
                    JOIN sessions s ON s.session_id = m.session_id
                    WHERE m.session_id = %s
                      AND s.user_id = %s
                      AND m.created_at >= (now() - (%s || ' days')::interval)
                    ORDER BY m.created_at ASC
                    """,
                    (self.session_id, self.user_id, str(self.window_days)),
                )
                rows = cur.fetchall()

            result = []
            for row in rows:
                raw_attachments = row["attachments"]
                if isinstance(raw_attachments, str):
                    try:
                        raw_attachments = json.loads(raw_attachments)
                    except Exception:
                        pass
                payload = {
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["created_at"].isoformat() if hasattr(row["created_at"], "isoformat") else str(row["created_at"]),
                }
                if raw_attachments:
                    payload["attachments"] = raw_attachments
                result.append(payload)
            return result
        except Exception as e:
            raise CustomException(e, sys)

    async def aget_messages_payload(self) -> List[dict[str, Any]]:
        return await asyncio.to_thread(self.get_messages_payload)

    def _prune_expired(self, cur):
        cur.execute(
            """
            DELETE FROM messages
            WHERE session_id = %s
              AND created_at < (now() - (%s || ' days')::interval)
            """,
            (self.session_id, str(self.window_days)),
        )

    def add_attachment(
        self,
        name: str,
        collection: str,
        source_type: str,
        extra: Optional[dict[str, Any]] = None,
        embedding_model: Optional[str] = None,
        vector_dim: Optional[int] = None,
    ):
        try:
            from src.components.embedder import Embedder
            provider = Embedder.get_active_provider(user_id=self.user_id)
            specs = Embedder.get_model_specs(provider)
            actual_model = embedding_model or specs["model"]
            actual_dim = vector_dim or specs["dimension"]

            with get_db_cursor() as cur:
                self._ensure_session(cur)
                cur.execute(
                    """
                    INSERT INTO attachments (session_id, name, collection, source_type, embedding_model, vector_dim, extra)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (session_id, collection) DO UPDATE
                        SET name = EXCLUDED.name,
                            source_type = EXCLUDED.source_type,
                            embedding_model = EXCLUDED.embedding_model,
                            vector_dim = EXCLUDED.vector_dim,
                            extra = EXCLUDED.extra
                    """,
                    (self.session_id, name, collection, source_type, actual_model, actual_dim, Json(extra) if extra else None),
                )
                # Explicitly touch updated_at so session activity timestamp is refreshed
                cur.execute(
                    "UPDATE sessions SET updated_at = now() WHERE session_id = %s",
                    (self.session_id,),
                )
        except Exception as e:
            raise CustomException(e, sys)

    async def aadd_attachment(
        self,
        name: str,
        collection: str,
        source_type: str,
        extra: Optional[dict[str, Any]] = None,
        embedding_model: Optional[str] = None,
        vector_dim: Optional[int] = None,
    ):
        return await asyncio.to_thread(self.add_attachment, name, collection, source_type, extra, embedding_model, vector_dim)

    def get_attachments(self) -> List[dict[str, Any]]:
        try:
            from src.components.embedder import Embedder
            active_provider = Embedder.get_active_provider(user_id=self.user_id)
            active_specs = Embedder.get_model_specs(active_provider)

            with get_db_cursor(commit=False) as cur:
                cur.execute(
                    """
                    SELECT a.name, a.collection, a.source_type AS type,
                           COALESCE(a.embedding_model, 'text-embedding-3-small') AS embedding_model,
                           COALESCE(a.vector_dim, 1536) AS vector_dim,
                           a.extra
                    FROM attachments a
                    JOIN sessions s ON s.session_id = a.session_id
                    WHERE a.session_id = %s AND s.user_id = %s
                    ORDER BY a.created_at ASC
                    """,
                    (self.session_id, self.user_id),
                )
                rows = cur.fetchall()

            results = []
            for row in rows:
                needs_reindex = (row["vector_dim"] != active_specs["dimension"])
                item = {
                    "name": row["name"],
                    "collection": row["collection"],
                    "type": row["type"],
                    "embedding_model": row["embedding_model"],
                    "vector_dim": row["vector_dim"],
                    "needs_reindex": needs_reindex,
                }
                if row["extra"]:
                    if isinstance(row["extra"], str):
                        try:
                            item.update(json.loads(row["extra"]))
                        except Exception:
                            pass
                    elif isinstance(row["extra"], dict):
                        item.update(row["extra"])
                results.append(item)
            return results
        except Exception as e:
            raise CustomException(e, sys)

    async def aget_attachments(self) -> List[dict[str, str]]:
        return await asyncio.to_thread(self.get_attachments)

    def remove_attachment(self, collection: str) -> bool:
        try:
            with get_db_cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM attachments
                    WHERE session_id = %s
                      AND collection = %s
                      AND session_id IN (SELECT session_id FROM sessions WHERE user_id = %s)
                    """,
                    (self.session_id, collection, self.user_id),
                )
                removed = cur.rowcount > 0
                if removed:
                    cur.execute(
                        "UPDATE sessions SET updated_at = now() WHERE session_id = %s",
                        (self.session_id,),
                    )
            return removed
        except Exception as e:
            raise CustomException(e, sys)

    async def aremove_attachment(self, collection: str) -> bool:
        return await asyncio.to_thread(self.remove_attachment, collection)

    def get_attachment_collections(self) -> List[str]:
        try:
            return [item["collection"] for item in self.get_attachments() if item.get("collection")]
        except Exception as e:
            raise CustomException(e, sys)

    async def aget_attachment_collections(self) -> List[str]:
        return await asyncio.to_thread(self.get_attachment_collections)

    def cleanup_attachments(self, valid_collections: List[str]) -> List[str]:
        try:
            if not valid_collections:
                # Safety guard: if valid_collections is empty, do NOT delete attachments
                return []
            valid_set = set(valid_collections)
            current = self.get_attachments()
            to_remove = [
                item["collection"] for item in current
                if item.get("collection") and item["collection"] not in valid_set
            ]
            for collection in to_remove:
                self.remove_attachment(collection)
            return to_remove
        except Exception as e:
            raise CustomException(e, sys)

    async def acleanup_attachments(self, valid_collections: List[str]) -> List[str]:
        return await asyncio.to_thread(self.cleanup_attachments, valid_collections)

    def get_title(self) -> str:
        try:
            with get_db_cursor(commit=False) as cur:
                cur.execute(
                    "SELECT title FROM sessions WHERE session_id = %s AND user_id = %s",
                    (self.session_id, self.user_id),
                )
                row = cur.fetchone()
                return row["title"] if row else "New Chat"
        except Exception as e:
            raise CustomException(e, sys)

    async def aget_title(self) -> str:
        return await asyncio.to_thread(self.get_title)

    def get_session_details(self, valid_collections: Optional[List[str]] = None) -> dict:
        """
        Batches messages, attachments, title, and optional stale attachment cleanup
        into a single DB query/transaction for fast chat loading.
        """
        try:
            with get_db_cursor(commit=bool(valid_collections)) as cur:
                # 1. Fetch messages
                cur.execute(
                    """
                    SELECT m.role, m.content, m.attachments, m.created_at
                    FROM messages m
                    JOIN sessions s ON s.session_id = m.session_id
                    WHERE m.session_id = %s
                      AND s.user_id = %s
                      AND m.created_at >= (now() - (%s || ' days')::interval)
                    ORDER BY m.created_at ASC
                    """,
                    (self.session_id, self.user_id, str(self.window_days)),
                )
                msg_rows = cur.fetchall()

                # 2. Fetch attachments
                cur.execute(
                    """
                    SELECT a.name, a.collection, a.source_type AS type,
                           COALESCE(a.embedding_model, 'text-embedding-3-small') AS embedding_model,
                           COALESCE(a.vector_dim, 1536) AS vector_dim,
                           a.extra
                    FROM attachments a
                    JOIN sessions s ON s.session_id = a.session_id
                    WHERE a.session_id = %s AND s.user_id = %s
                    ORDER BY a.created_at ASC
                    """,
                    (self.session_id, self.user_id),
                )
                att_rows = cur.fetchall()

                # 3. Fetch session record for timestamps and title
                cur.execute(
                    "SELECT title, created_at, updated_at FROM sessions WHERE session_id = %s AND user_id = %s",
                    (self.session_id, self.user_id),
                )
                s_row = cur.fetchone() or {}

                # Derive title
                title = next(
                    (
                        r["content"].strip()[:60]
                        for r in msg_rows
                        if r.get("role") == "human" and r.get("content", "").strip()
                    ),
                    None,
                )
                if not title:
                    title = s_row.get("title") if s_row else "New Chat"

                # Check session inactivity age (7 days)
                now_dt = datetime.now(timezone.utc)
                timestamps = []
                for m in msg_rows:
                    if m.get("created_at"):
                        timestamps.append(m["created_at"])
                if s_row.get("updated_at"):
                    timestamps.append(s_row["updated_at"])
                if s_row.get("created_at"):
                    timestamps.append(s_row["created_at"])

                latest_activity = max(timestamps) if timestamps else now_dt
                if latest_activity.tzinfo is None:
                    latest_activity = latest_activity.replace(tzinfo=timezone.utc)

                session_age_seconds = (now_dt - latest_activity).total_seconds()
                is_older_than_7_days = session_age_seconds >= (7 * 86400)

            messages = []
            for row in msg_rows:
                raw_attachments = row["attachments"]
                if isinstance(raw_attachments, str):
                    try:
                        raw_attachments = json.loads(raw_attachments)
                    except Exception:
                        pass
                item = {
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["created_at"].isoformat() if hasattr(row["created_at"], "isoformat") else str(row["created_at"]),
                }
                if raw_attachments:
                    item["attachments"] = raw_attachments
                messages.append(item)

            from src.components.embedder import Embedder
            active_provider = Embedder.get_active_provider(user_id=self.user_id)
            active_specs = Embedder.get_model_specs(active_provider)

            valid_set = set(valid_collections) if valid_collections is not None else None
            has_expired_attachments = False

            attachments = []
            for row in att_rows:
                extra = row.get("extra")
                if isinstance(extra, str):
                    try:
                        extra = json.loads(extra)
                    except Exception:
                        pass
                v_dim = row.get("vector_dim", 1536)
                needs_reindex = (v_dim != active_specs["dimension"])

                col = row.get("collection")
                is_col_missing = bool(valid_set is not None and len(valid_set) > 0 and col and col not in valid_set)
                is_expired = is_older_than_7_days or is_col_missing
                if is_expired:
                    has_expired_attachments = True

                item = {
                    "name": row["name"],
                    "collection": col,
                    "type": row["type"],
                    "embedding_model": row.get("embedding_model", "text-embedding-3-small"),
                    "vector_dim": v_dim,
                    "needs_reindex": needs_reindex,
                    "expired": is_expired,
                }
                if extra and isinstance(extra, dict):
                    item.update(extra)
                attachments.append(item)

            return {
                "session_id": self.session_id,
                "title": title or "New Chat",
                "messages": messages,
                "attachments": attachments,
                "is_older_than_7_days": is_older_than_7_days,
                "has_expired_attachments": has_expired_attachments,
            }
        except Exception as e:
            raise CustomException(e, sys)

    async def aget_session_details(self, valid_collections: Optional[List[str]] = None) -> dict:
        return await asyncio.to_thread(self.get_session_details, valid_collections)

    def clear(self):
        try:
            with get_db_cursor() as cur:
                self._ensure_summary_table(cur)
                cur.execute("DELETE FROM session_summaries WHERE session_id = %s", (self.session_id,))
                cur.execute("DELETE FROM sessions WHERE session_id = %s AND user_id = %s", (self.session_id, self.user_id))
        except Exception as e:
            raise CustomException(e, sys)

    async def aclear(self):
        return await asyncio.to_thread(self.clear)

    @staticmethod
    def clear_all(user_id: str):
        try:
            with get_db_cursor() as cur:
                MemoryManager._ensure_summary_table(cur)
                cur.execute(
                    "DELETE FROM session_summaries WHERE session_id IN (SELECT session_id FROM sessions WHERE user_id = %s)",
                    (user_id,),
                )
                cur.execute("DELETE FROM sessions WHERE user_id = %s", (user_id,))
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    async def aclear_all(user_id: str):
        return await asyncio.to_thread(MemoryManager.clear_all, user_id)

    @staticmethod
    def prune_expired_sessions(user_id: str) -> List[dict]:
        """
        Finds all sessions for this user where ALL activity (latest message,
        latest attachment, updated_at, and created_at) is strictly older than window_days (7 days).
        Deletes all their Qdrant collections and removes their database records.
        Returns a list of pruned session dictionaries: [{'session_id': ..., 'title': ...}]
        """
        try:
            window_days = config.get("memory", {}).get("window_days", 7)

            with get_db_cursor(commit=True) as cur:
                # Find expired sessions where ALL activity is strictly older than window_days
                cur.execute(
                    """
                    SELECT
                        s.session_id,
                        COALESCE(
                            (
                                SELECT content FROM messages
                                WHERE session_id = s.session_id AND role = 'human'
                                ORDER BY created_at ASC LIMIT 1
                            ),
                            s.title,
                            'New Chat'
                        ) AS title,
                        ARRAY_AGG(DISTINCT a.collection) FILTER (WHERE a.collection IS NOT NULL) AS collections
                    FROM sessions s
                    LEFT JOIN messages m ON m.session_id = s.session_id
                    LEFT JOIN attachments a ON a.session_id = s.session_id
                    WHERE s.user_id = %s
                    GROUP BY s.session_id, s.title, s.updated_at, s.created_at
                    HAVING GREATEST(
                        COALESCE(MAX(m.created_at), '1970-01-01 00:00:00+00'::timestamptz),
                        COALESCE(MAX(a.created_at), '1970-01-01 00:00:00+00'::timestamptz),
                        COALESCE(s.updated_at, '1970-01-01 00:00:00+00'::timestamptz),
                        COALESCE(s.created_at, '1970-01-01 00:00:00+00'::timestamptz)
                    ) < (now() - (%s || ' days')::interval)
                    """,
                    (user_id, str(window_days)),
                )
                expired_rows = cur.fetchall()

                if not expired_rows:
                    return []

                expired_ids = [r["session_id"] for r in expired_rows]
                collections_to_delete = []
                expired_info = []

                for r in expired_rows:
                    clean_title = (r["title"] or "New Chat").strip()[:60]
                    expired_info.append({
                        "session_id": r["session_id"],
                        "title": clean_title,
                    })
                    if r["collections"]:
                        collections_to_delete.extend(r["collections"])

                # Atomic cascade deletion in PostgreSQL
                cur.execute(
                    "DELETE FROM sessions WHERE session_id = ANY(%s) AND user_id = %s",
                    (expired_ids, user_id),
                )

            # Purge associated vector collections from Qdrant
            if collections_to_delete:
                from src.utils.helpers import delete_collections
                delete_collections(collections_to_delete)

            logging.info(
                "Pruned %d expired sessions (> %d days) for user %s: %s",
                len(expired_info),
                window_days,
                user_id,
                expired_ids,
            )
            return expired_info

        except Exception as e:
            logging.warning("Error pruning expired sessions: %s", e)
            return []

    @staticmethod
    async def aprune_expired_sessions(user_id: str) -> List[dict]:
        return await asyncio.to_thread(MemoryManager.prune_expired_sessions, user_id)

    @staticmethod
    def list_sessions(user_id: str, valid_collections: Optional[List[str]] = None) -> List[dict]:
        """
        Fast single-query listing of active sessions without N+1 loops.
        """
        try:
            with get_db_cursor(commit=True) as cur:
                # Clean up truly empty ghost sessions only if older than 2 hours in DB time and completely inactive
                cur.execute(
                    """
                    DELETE FROM sessions
                    WHERE user_id = %s
                      AND NOT EXISTS (SELECT 1 FROM messages WHERE session_id = sessions.session_id)
                      AND NOT EXISTS (SELECT 1 FROM attachments WHERE session_id = sessions.session_id)
                      AND created_at < (now() - interval '2 hours')
                    """,
                    (user_id,),
                )

                cur.execute(
                    """
                    SELECT
                        s.session_id,
                        s.title,
                        s.updated_at,
                        s.created_at,
                        COUNT(DISTINCT m.id) AS message_count,
                        COUNT(DISTINCT a.id) AS attachment_count,
                        MAX(m.created_at) AS last_message_at,
                        MAX(a.created_at) AS last_attachment_at,
                        (
                            SELECT content FROM messages
                            WHERE session_id = s.session_id AND role = 'human'
                            ORDER BY created_at ASC LIMIT 1
                        ) AS first_human_message
                    FROM sessions s
                    LEFT JOIN messages m ON m.session_id = s.session_id
                    LEFT JOIN attachments a ON a.session_id = s.session_id
                    WHERE s.user_id = %s
                    GROUP BY s.session_id
                    """,
                    (user_id,),
                )
                rows = cur.fetchall()

            sessions = []
            for row in rows:
                last_active = (
                    row["last_message_at"]
                    or row["last_attachment_at"]
                    or row["updated_at"]
                    or row["created_at"]
                )
                title = (
                    (row["first_human_message"] or "").strip()[:60]
                    or row["title"]
                    or "New Chat"
                )

                sessions.append(
                    {
                        "session_id": row["session_id"],
                        "title": title,
                        "message_count": row["message_count"],
                        "attachment_count": row["attachment_count"],
                        "last_active": last_active.isoformat() if hasattr(last_active, "isoformat") else str(last_active),
                        "last_active_label": last_active.strftime("%b %d, %H:%M") if hasattr(last_active, "strftime") else str(last_active),
                    }
                )

            sessions.sort(key=lambda item: item["last_active"], reverse=True)
            return sessions
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    async def alist_sessions(user_id: str, valid_collections: Optional[List[str]] = None) -> List[dict]:
        return await asyncio.to_thread(MemoryManager.list_sessions, user_id, valid_collections)

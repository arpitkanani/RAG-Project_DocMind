import asyncio
from datetime import datetime, timedelta, timezone
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
        self.window_days = config["memory"]["window_days"]
        self.recent_turns_to_keep = config["memory"].get("recent_messages_verbatim", 6)

    def _ensure_session(self, cur, title: Optional[str] = None):
        cur.execute(
            """
            INSERT INTO sessions (session_id, user_id, title)
            VALUES (%s, %s, COALESCE(%s, 'New Chat'))
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

        llm = _build_llm()
        llm_rate_limiter.acquire()
        response = llm.invoke(prompt, config={"tags": ["memory_summary"]})
        return (response.content if hasattr(response, "content") else str(response)).strip()

    def get_messages_payload(self) -> List[dict[str, Any]]:
        try:
            cutoff = datetime.now() - timedelta(days=self.window_days)
            with get_db_cursor(commit=False) as cur:
                cur.execute(
                    """
                    SELECT m.role, m.content, m.attachments, m.created_at
                    FROM messages m
                    JOIN sessions s ON s.session_id = m.session_id
                    WHERE m.session_id = %s
                      AND s.user_id = %s
                      AND m.created_at >= %s
                    ORDER BY m.created_at ASC
                    """,
                    (self.session_id, self.user_id, cutoff),
                )
                rows = cur.fetchall()

            result = []
            for row in rows:
                payload = {
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["created_at"].isoformat(),
                }
                if row["attachments"]:
                    payload["attachments"] = row["attachments"]
                result.append(payload)
            return result
        except Exception as e:
            raise CustomException(e, sys)

    async def aget_messages_payload(self) -> List[dict[str, Any]]:
        return await asyncio.to_thread(self.get_messages_payload)

    def _prune_expired(self, cur):
        cutoff = datetime.now() - timedelta(days=self.window_days)
        cur.execute(
            "DELETE FROM messages WHERE session_id = %s AND created_at < %s",
            (self.session_id, cutoff),
        )

    def add_attachment(
        self,
        name: str,
        collection: str,
        source_type: str,
        extra: Optional[dict[str, Any]] = None,
    ):
        try:
            with get_db_cursor() as cur:
                self._ensure_session(cur)
                cur.execute(
                    """
                    INSERT INTO attachments (session_id, name, collection, source_type, extra)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (session_id, collection) DO UPDATE
                        SET name = EXCLUDED.name,
                            source_type = EXCLUDED.source_type,
                            extra = EXCLUDED.extra
                    """,
                    (self.session_id, name, collection, source_type, Json(extra) if extra else None),
                )
        except Exception as e:
            raise CustomException(e, sys)

    async def aadd_attachment(
        self,
        name: str,
        collection: str,
        source_type: str,
        extra: Optional[dict[str, Any]] = None,
    ):
        return await asyncio.to_thread(self.add_attachment, name, collection, source_type, extra)

    def get_attachments(self) -> List[dict[str, str]]:
        try:
            with get_db_cursor(commit=False) as cur:
                cur.execute(
                    """
                    SELECT a.name, a.collection, a.source_type AS type, a.extra
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
                item = {"name": row["name"], "collection": row["collection"], "type": row["type"]}
                if row["extra"]:
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
    def list_sessions(user_id: str, valid_collections: Optional[List[str]] = None) -> List[dict]:
        try:
            window_days = config["memory"]["window_days"]
            cutoff = datetime.now() - timedelta(days=window_days)

            with get_db_cursor(commit=False) as cur:
                cur.execute(
                    """
                    SELECT
                        s.session_id,
                        s.title,
                        s.updated_at,
                        s.created_at,
                        COUNT(DISTINCT m.id) FILTER (WHERE m.created_at >= %s) AS message_count,
                        COUNT(DISTINCT a.id) AS attachment_count,
                        MAX(m.created_at) AS last_message_at,
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
                    (cutoff, user_id),
                )
                rows = cur.fetchall()

            sessions = []
            for row in rows:
                if row["message_count"] == 0 and row["attachment_count"] == 0:
                    MemoryManager(session_id=row["session_id"], user_id=user_id).clear()
                    continue

                if valid_collections is not None:
                    MemoryManager(
                        session_id=row["session_id"], user_id=user_id
                    ).cleanup_attachments(valid_collections)

                last_active = row["last_message_at"] or row["updated_at"] or row["created_at"]
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
                        "last_active": last_active.isoformat(),
                        "last_active_label": last_active.strftime("%b %d, %H:%M"),
                    }
                )

            sessions.sort(key=lambda item: item["last_active"], reverse=True)
            return sessions
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    async def alist_sessions(user_id: str, valid_collections: Optional[List[str]] = None) -> List[dict]:
        return await asyncio.to_thread(MemoryManager.list_sessions, user_id, valid_collections)
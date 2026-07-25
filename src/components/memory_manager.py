import sys
from datetime import datetime, timedelta
from typing import Any, List, Optional

import yaml
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from psycopg2.extras import Json

from src.database.db import get_db_cursor
from src.exception import CustomException
from src.logger import logging

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)


class MemoryManager:
    """
    Manage persistent multi-session chat history and session attachments,
    backed by PostgreSQL. Every operation is scoped to (session_id, user_id)
    so one user can never read, modify, or delete another user's data.
    """

    def __init__(self, session_id: str = "default", user_id: str = None):
        if not user_id:
            raise CustomException(
                ValueError("MemoryManager requires a user_id"), sys
            )
        try:
            logging.info(
                "Initializing MemoryManager | session: %s | user: %s",
                session_id,
                user_id,
            )
            self.session_id = session_id
            self.user_id = user_id
            self.window_days = config["memory"]["window_days"]
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Session ownership / existence
    # ------------------------------------------------------------------
    def _ensure_session(self, cur, title: Optional[str] = None):
        """Create the session row if it doesn't exist yet (upsert on touch)."""
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

    def _owns_session(self, cur) -> bool:
        cur.execute(
            "SELECT 1 FROM sessions WHERE session_id = %s AND user_id = %s",
            (self.session_id, self.user_id),
        )
        return cur.fetchone() is not None

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------
    def save_message(
        self,
        role: str,
        content: str,
        attachments: Optional[List[dict[str, Any]]] = None,
    ):
        try:
            title = content.strip()[:60] if (role == "human" and content.strip()) else None
            attachments_json = None
            if attachments and role == "human":
                attachments_json = Json(
                    [item for item in attachments if isinstance(item, dict)]
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

    def get_history(self) -> List[BaseMessage]:
        try:
            messages: List[BaseMessage] = []
            for msg in self.get_messages_payload():
                if msg["role"] == "human":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "ai":
                    messages.append(AIMessage(content=msg["content"]))

            logging.info(
                "History loaded | session: %s | messages: %s",
                self.session_id,
                len(messages),
            )
            return messages
        except Exception as e:
            raise CustomException(e, sys)

    def get_messages_payload(self) -> List[dict[str, Any]]:
        """Return recent raw messages (within retention window) for this user's session."""
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

    def get_message_count(self) -> int:
        try:
            return len(self.get_messages_payload())
        except Exception as e:
            raise CustomException(e, sys)

    def _prune_expired(self, cur):
        """Delete messages older than the retention window for this session."""
        cutoff = datetime.now() - timedelta(days=self.window_days)
        cur.execute(
            "DELETE FROM messages WHERE session_id = %s AND created_at < %s",
            (self.session_id, cutoff),
        )

    # ------------------------------------------------------------------
    # Attachments
    # ------------------------------------------------------------------
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
            logging.info(
                "Attachment saved | session: %s | collection: %s",
                self.session_id,
                collection,
            )
        except Exception as e:
            raise CustomException(e, sys)

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
            logging.info(
                "Attachment removed | session: %s | collection: %s | removed: %s",
                self.session_id,
                collection,
                removed,
            )
            return removed
        except Exception as e:
            raise CustomException(e, sys)

    def get_attachment_collections(self) -> List[str]:
        try:
            return [item["collection"] for item in self.get_attachments() if item.get("collection")]
        except Exception as e:
            raise CustomException(e, sys)

    def cleanup_attachments(self, valid_collections: List[str]) -> List[str]:
        """Remove attachment rows that point to vector collections that no longer exist."""
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

    # ------------------------------------------------------------------
    # Session metadata / lifecycle
    # ------------------------------------------------------------------
    def touch(self, *, title: Optional[str] = None, timestamp: Optional[str] = None):
        try:
            with get_db_cursor() as cur:
                self._ensure_session(cur, title=title[:60] if title else None)
        except Exception as e:
            raise CustomException(e, sys)

    def get_title(self) -> str:
        """Return the session's stored title (used when no human message exists yet)."""
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

    def has_persisted_state(self) -> bool:
        try:
            with get_db_cursor(commit=False) as cur:
                cur.execute(
                    "SELECT 1 FROM sessions WHERE session_id = %s AND user_id = %s",
                    (self.session_id, self.user_id),
                )
                return cur.fetchone() is not None
        except Exception as e:
            raise CustomException(e, sys)

    def clear(self):
        """Delete only the active session (messages + attachments cascade automatically)."""
        try:
            with get_db_cursor() as cur:
                cur.execute(
                    "DELETE FROM sessions WHERE session_id = %s AND user_id = %s",
                    (self.session_id, self.user_id),
                )
            logging.info("Session cleared: %s", self.session_id)
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def clear_all(user_id: str):
        """Delete every session (and cascading messages/attachments) for ONE user only."""
        try:
            with get_db_cursor() as cur:
                cur.execute("DELETE FROM sessions WHERE user_id = %s", (user_id,))
            logging.info("All chat history cleared | user: %s", user_id)
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def list_sessions(user_id: str, valid_collections: Optional[List[str]] = None) -> List[dict]:
        """List all non-expired sessions belonging to ONE user, for the sidebar."""
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
                    # Mirrors old behavior: empty, stale sessions get cleaned up
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
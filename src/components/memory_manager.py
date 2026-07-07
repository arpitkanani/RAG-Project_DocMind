import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any, List, Optional

import yaml
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.exception import CustomException
from src.logger import logging

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)


class MemoryManager:
    """Manage persistent multi-session chat history and session attachments."""

    DEFAULT_STATE = {
        "attachments": [],
        "title": "New Chat",
        "created_at": None,
        "updated_at": None,
    }

    def __init__(self, session_id: str = "default"):
        try:
            logging.info("Initializing MemoryManager | session: %s", session_id)
            self.persist_dir = config["memory"]["persist_directory"]
            self.window_days = config["memory"]["window_days"]
            self.session_id = session_id
            self.memory_file = os.path.join(
                self.persist_dir,
                f"chat_history_{session_id}.json",
            )
            self.state_file = os.path.join(
                self.persist_dir,
                f"session_state_{session_id}.json",
            )
            os.makedirs(self.persist_dir, exist_ok=True)
        except Exception as e:
            raise CustomException(e, sys)

    def save_message(
        self,
        role: str,
        content: str,
        attachments: Optional[List[dict[str, Any]]] = None,
    ):
        """Append one message to the active session file."""
        try:
            history = self._load_raw(prune_expired=True)
            timestamp = datetime.now().isoformat()
            payload = {
                "role": role,
                "content": content,
                "timestamp": timestamp,
            }
            if attachments and role == "human":
                payload["attachments"] = [
                    item for item in attachments if isinstance(item, dict)
                ]
            history.append(payload)
            self._write_raw(history)
            title = None
            if role == "human" and content.strip():
                title = content.strip()[:60]
            self.touch(title=title, timestamp=timestamp)
            logging.info("Message saved | role: %s | session: %s", role, self.session_id)
        except Exception as e:
            raise CustomException(e, sys)

    def get_history(self) -> List[BaseMessage]:
        """Load active-session history from the configured retention window."""
        try:
            messages: List[BaseMessage] = []
            for msg in self._load_raw(prune_expired=True):
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
        """Return recent raw messages for the frontend history viewer."""
        try:
            return self._load_raw(prune_expired=True)
        except Exception as e:
            raise CustomException(e, sys)

    def add_attachment(
        self,
        name: str,
        collection: str,
        source_type: str,
        extra: Optional[dict[str, Any]] = None,
    ):
        """Persist attachment metadata so each session knows its document scope."""
        try:
            state = self._load_state()
            attachments = state.get("attachments", [])
            filtered = [
                item for item in attachments if item.get("collection") != collection
            ]
            attachment = {
                "name": name,
                "collection": collection,
                "type": source_type,
            }
            if extra:
                attachment.update(extra)
            filtered.append(attachment)
            state["attachments"] = filtered
            self._ensure_state_defaults(state)
            timestamp = datetime.now().isoformat()
            state["created_at"] = state.get("created_at") or timestamp
            state["updated_at"] = timestamp
            self._write_state(state)
            logging.info(
                "Attachment saved | session: %s | collection: %s",
                self.session_id,
                collection,
            )
        except Exception as e:
            raise CustomException(e, sys)

    def get_attachments(self) -> List[dict[str, str]]:
        try:
            state = self._load_state()
            attachments = state.get("attachments", [])
            return [item for item in attachments if isinstance(item, dict)]
        except Exception as e:
            raise CustomException(e, sys)

    def remove_attachment(self, collection: str) -> bool:
        """Remove one attachment from the active session state."""
        try:
            state = self._load_state()
            attachments = state.get("attachments", [])
            filtered = [
                item for item in attachments if item.get("collection") != collection
            ]
            removed = len(filtered) != len(attachments)
            state["attachments"] = filtered
            if removed:
                state["updated_at"] = datetime.now().isoformat()
            self._write_state(state)
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
        """Return collection ids linked to the active session."""
        try:
            return [
                item["collection"]
                for item in self.get_attachments()
                if item.get("collection")
            ]
        except Exception as e:
            raise CustomException(e, sys)

    def touch(
        self,
        *,
        title: Optional[str] = None,
        timestamp: Optional[str] = None,
    ):
        """Persist lightweight workspace metadata for refresh recovery."""
        try:
            state = self._load_state()
            self._ensure_state_defaults(state)
            now = timestamp or datetime.now().isoformat()
            state["created_at"] = state.get("created_at") or now
            state["updated_at"] = now
            if title:
                state["title"] = title[:60]
            self._write_state(state, keep_if_empty=True)
        except Exception as e:
            raise CustomException(e, sys)

    def cleanup_attachments(self, valid_collections: List[str]) -> List[str]:
        """Remove attachment metadata that points to missing vector collections."""
        try:
            state = self._load_state()
            attachments = state.get("attachments", [])
            valid_set = set(valid_collections)
            filtered = [
                item
                for item in attachments
                if item.get("collection") and item.get("collection") in valid_set
            ]
            removed = [
                item.get("collection")
                for item in attachments
                if item.get("collection") and item.get("collection") not in valid_set
            ]

            if len(filtered) != len(attachments):
                state["attachments"] = filtered
                state["updated_at"] = datetime.now().isoformat()
                self._write_state(
                    state,
                    keep_if_empty=bool(filtered or self._load_raw(prune_expired=True)),
                )

            return [item for item in removed if item]
        except Exception as e:
            raise CustomException(e, sys)

    def has_persisted_state(self) -> bool:
        try:
            has_history = bool(self._load_raw(prune_expired=True))
            state = self._load_state()
            attachments = state.get("attachments", [])
            metadata_only = bool(
                state.get("created_at") or state.get("updated_at") or state.get("title")
            )
            return has_history or bool(attachments) or metadata_only
        except Exception as e:
            raise CustomException(e, sys)

    def clear(self):
        """Delete only the active session's chat history and attachment state."""
        try:
            for path in (self.memory_file, self.state_file):
                if os.path.exists(path):
                    os.remove(path)
            logging.info("Session cleared: %s", self.session_id)
        except Exception as e:
            raise CustomException(e, sys)

    def get_message_count(self) -> int:
        try:
            return len(self.get_history())
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def clear_all(persist_dir: str | None = None):
        """Delete every saved session history and session-state file."""
        try:
            persist_dir, _ = MemoryManager._resolve_settings(persist_dir)
            if not os.path.exists(persist_dir):
                return

            for filename in os.listdir(persist_dir):
                if (
                    filename.startswith("chat_history_")
                    or filename.startswith("session_state_")
                ) and filename.endswith(".json"):
                    os.remove(os.path.join(persist_dir, filename))

            logging.info("All chat history cleared")
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def list_sessions(
        persist_dir: str | None = None,
        valid_collections: Optional[List[str]] = None,
    ) -> List[dict]:
        """List all non-expired sessions for the sidebar."""
        try:
            persist_dir, window_days = MemoryManager._resolve_settings(persist_dir)
            if not os.path.exists(persist_dir):
                return []

            session_ids = set()
            valid_collection_set = set(valid_collections or [])

            for filename in os.listdir(persist_dir):
                if filename.startswith("chat_history_") and filename.endswith(".json"):
                    session_ids.add(
                        filename.replace("chat_history_", "").replace(".json", "")
                    )
                if filename.startswith("session_state_") and filename.endswith(".json"):
                    session_ids.add(
                        filename.replace("session_state_", "").replace(".json", "")
                    )

            sessions = []

            for session_id in session_ids:
                memory = MemoryManager(session_id=session_id)
                history = memory._load_raw(prune_expired=True)
                state = memory._load_state()

                if valid_collections is not None:
                    memory.cleanup_attachments(list(valid_collection_set))
                    state = memory._load_state()

                attachments = [
                    item
                    for item in state.get("attachments", [])
                    if isinstance(item, dict)
                ]

                if not history and not state.get("updated_at"):
                    memory.clear()
                    continue

                timestamps = [
                    message.get("timestamp")
                    for message in history
                    if message.get("timestamp")
                ]
                if state.get("updated_at"):
                    timestamps.append(state["updated_at"])
                if state.get("created_at"):
                    timestamps.append(state["created_at"])
                if not timestamps:
                    memory.clear()
                    continue

                last_active = max(datetime.fromisoformat(value) for value in timestamps)
                first_human = next(
                    (
                        message["content"].strip()[:60]
                        for message in history
                        if message.get("role") == "human"
                        and message.get("content", "").strip()
                    ),
                    "",
                )
                title = first_human or state.get("title") or "New Chat"

                sessions.append(
                    {
                        "session_id": session_id,
                        "title": title,
                        "message_count": len(history),
                        "attachment_count": len(attachments),
                        "last_active": last_active.isoformat(),
                        "last_active_label": last_active.strftime("%b %d, %H:%M"),
                    }
                )

            sessions.sort(key=lambda item: item["last_active"], reverse=True)
            return sessions
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def _resolve_settings(persist_dir: str | None = None) -> tuple[str, int]:
        if persist_dir is not None:
            return persist_dir, config["memory"]["window_days"]

        return (
            config["memory"]["persist_directory"],
            config["memory"]["window_days"],
        )

    @staticmethod
    def _filter_recent_messages(history: Any, window_days: int) -> List[dict[str, Any]]:
        cutoff = datetime.now() - timedelta(days=window_days)
        recent_messages: List[dict[str, Any]] = []

        if not isinstance(history, list):
            return recent_messages

        for message in history:
            if not isinstance(message, dict):
                continue

            timestamp = message.get("timestamp")
            role = message.get("role")
            content = message.get("content")

            if not timestamp or not role or content is None:
                continue

            try:
                message_time = datetime.fromisoformat(timestamp)
            except ValueError:
                continue

            if message_time >= cutoff:
                normalized = {
                    "role": str(role),
                    "content": str(content),
                    "timestamp": message_time.isoformat(),
                }
                attachments = message.get("attachments")
                if isinstance(attachments, list):
                    normalized["attachments"] = [
                        item for item in attachments if isinstance(item, dict)
                    ]
                recent_messages.append(normalized)

        return recent_messages

    def _load_raw(self, prune_expired: bool = False) -> List[dict[str, Any]]:
        try:
            if not os.path.exists(self.memory_file):
                return []

            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except (json.JSONDecodeError, OSError, ValueError):
                return []

            recent_history = self._filter_recent_messages(history, self.window_days)

            if prune_expired and len(recent_history) != len(history):
                self._write_raw(recent_history)

            return recent_history if prune_expired else history
        except Exception as e:
            raise CustomException(e, sys)

    def _write_raw(self, history: List[dict[str, Any]]):
        try:
            if not history:
                if os.path.exists(self.memory_file):
                    os.remove(self.memory_file)
                return

            os.makedirs(self.persist_dir, exist_ok=True)
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise CustomException(e, sys)

    def _load_state(self) -> dict[str, Any]:
        try:
            if not os.path.exists(self.state_file):
                return dict(self.DEFAULT_STATE)

            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)

            if not isinstance(state, dict):
                return dict(self.DEFAULT_STATE)

            self._ensure_state_defaults(state)
            return state
        except (json.JSONDecodeError, OSError, ValueError):
            return dict(self.DEFAULT_STATE)
        except Exception as e:
            raise CustomException(e, sys)

    def _write_state(self, state: dict[str, Any], keep_if_empty: bool = False):
        try:
            self._ensure_state_defaults(state)
            attachments = state.get("attachments", [])
            has_metadata = bool(
                state.get("created_at") or state.get("updated_at") or state.get("title")
            )
            if not attachments and not keep_if_empty and not has_metadata:
                if os.path.exists(self.state_file):
                    os.remove(self.state_file)
                return

            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise CustomException(e, sys)

    @classmethod
    def _ensure_state_defaults(cls, state: dict[str, Any]) -> dict[str, Any]:
        for key, value in cls.DEFAULT_STATE.items():
            if key not in state:
                state[key] = value
        if not isinstance(state.get("attachments"), list):
            state["attachments"] = []
        return state

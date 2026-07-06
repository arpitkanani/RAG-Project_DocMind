import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any, List

import yaml
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.exception import CustomException
from src.logger import logging

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)


class MemoryManager:
    """Manage persistent multi-session chat history and session attachments."""

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

    def save_message(self, role: str, content: str):
        """Append one message to the active session file."""
        try:
            history = self._load_raw(prune_expired=True)
            history.append(
                {
                    "role": role,
                    "content": content,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            self._write_raw(history)
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

    def get_messages_payload(self) -> List[dict[str, str]]:
        """Return recent raw messages for the frontend history viewer."""
        try:
            return self._load_raw(prune_expired=True)
        except Exception as e:
            raise CustomException(e, sys)

    def add_attachment(self, name: str, collection: str, source_type: str):
        """Persist attachment metadata so each session knows its document scope."""
        try:
            state = self._load_state()
            attachments = state.get("attachments", [])
            filtered = [
                item for item in attachments if item.get("collection") != collection
            ]
            filtered.append(
                {
                    "name": name,
                    "collection": collection,
                    "type": source_type,
                }
            )
            state["attachments"] = filtered
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
    def list_sessions(persist_dir: str | None = None) -> List[dict]:
        """List all non-expired sessions for the sidebar."""
        try:
            persist_dir, window_days = MemoryManager._resolve_settings(persist_dir)
            if not os.path.exists(persist_dir):
                return []

            sessions = []

            for filename in os.listdir(persist_dir):
                if not (
                    filename.startswith("chat_history_") and filename.endswith(".json")
                ):
                    continue

                filepath = os.path.join(persist_dir, filename)
                session_id = filename.replace("chat_history_", "").replace(".json", "")

                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        history = json.load(f)
                except (json.JSONDecodeError, OSError, ValueError):
                    continue

                recent_history = MemoryManager._filter_recent_messages(
                    history,
                    window_days,
                )

                if not recent_history:
                    os.remove(filepath)
                    continue

                if len(recent_history) != len(history):
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(recent_history, f, indent=2, ensure_ascii=False)

                last_time = datetime.fromisoformat(recent_history[-1]["timestamp"])
                first_human = next(
                    (
                        message["content"].strip()[:60]
                        for message in recent_history
                        if message.get("role") == "human"
                        and message.get("content", "").strip()
                    ),
                    "New Chat",
                )

                sessions.append(
                    {
                        "session_id": session_id,
                        "title": first_human,
                        "message_count": len(recent_history),
                        "last_active": last_time.isoformat(),
                        "last_active_label": last_time.strftime("%b %d, %H:%M"),
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
    def _filter_recent_messages(history: Any, window_days: int) -> List[dict[str, str]]:
        cutoff = datetime.now() - timedelta(days=window_days)
        recent_messages: List[dict[str, str]] = []

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
                recent_messages.append(
                    {
                        "role": str(role),
                        "content": str(content),
                        "timestamp": message_time.isoformat(),
                    }
                )

        return recent_messages

    def _load_raw(self, prune_expired: bool = False) -> List[dict[str, str]]:
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

    def _write_raw(self, history: List[dict[str, str]]):
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
                return {"attachments": []}

            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)

            if not isinstance(state, dict):
                return {"attachments": []}

            return state
        except (json.JSONDecodeError, OSError, ValueError):
            return {"attachments": []}
        except Exception as e:
            raise CustomException(e, sys)

    def _write_state(self, state: dict[str, Any]):
        try:
            attachments = state.get("attachments", [])
            if not attachments:
                if os.path.exists(self.state_file):
                    os.remove(self.state_file)
                return

            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump({"attachments": attachments}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise CustomException(e, sys)

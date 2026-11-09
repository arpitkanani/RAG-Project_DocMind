"""DocuVortex Query Manager — Detached background generation task manager
with event broadcast and disconnect resilience.

Ensures that:
1. Generation runs in an un-cancellable background task (asyncio.shield).
2. Client disconnects (chat switches, page refreshes) do not cancel generation.
3. finalize_node always completes and persists human + AI messages to PostgreSQL.
4. Multiple clients or reconnecting clients can subscribe to in-flight query events.
"""

import asyncio
import json
import re
import sys
from typing import Any, AsyncGenerator, Dict, List, Optional, Set
from langchain_core.runnables import RunnableConfig

from src.components.embedder import Embedder
from src.components.memory_manager import MemoryManager
from src.components.vector_store import VectorStore
from src.exception import (
    CollectionEmbeddingMismatchError,
    CollectionNotFoundError,
    CustomException,
    EmbeddingUnavailableError,
    KnowledgeBaseEmptyError,
)
from src.graph.builder import rag_graph
from src.logger import logging
from src.utils.helpers import extract_text
from src.utils.rate_limiter import LLMRateLimitError

# Stage labels for status events
_STAGE_LABELS = {
    "load_context": ("classifying", "Analyzing query..."),
    "classify_intent": ("classifying", "Analyzing query..."),
    "clarify_question": ("classifying", "Checking question clarity..."),
    "chitchat": ("thinking", "Thinking..."),
    "run_react_agent": ("thinking", "Thinking..."),
    "agent": ("thinking", "Thinking..."),
    "tools": ("tool", "Executing tool..."),
    "retrieve_qa": ("retrieving", "Retrieving context..."),
    "retrieve_summary": ("retrieving", "Retrieving context..."),
    "grade_documents": ("grading", "Evaluating relevance..."),
    "generate": ("structuring", "Structuring answer..."),
    "fallback_response": ("structuring", "Structuring answer..."),
}

TOOL_DISPLAY_MAP = {
    "search_tool": ("search", "🔍 Searching the web..."),
    "duckduckgo_search": ("search", "🔍 Searching the web..."),
    "calculator_tool": ("calc", "🧮 Calculating..."),
    "calculator": ("calc", "🧮 Calculating..."),
    "stock_price_tool": ("stock", "📈 Fetching stock price..."),
    "get_stock_price": ("stock", "📈 Fetching stock price..."),
    "get_weather": ("weather", "🌤️ Checking weather..."),
    "search_arxiv": ("research", "📚 Searching academic papers..."),
    "rag_query": ("rag", "Searching knowledge base..."),
    "summarize_document": ("rag", "Reading document context..."),
}


class ActiveSessionQuery:
    """Encapsulates a running detached query task and its subscriber queues."""

    def __init__(self, user_id: str, session_id: str, query: str):
        self.user_id = user_id
        self.session_id = session_id
        self.query = query
        self.subscribers: Set[asyncio.Queue] = set()
        self.event_history: List[Dict[str, Any]] = []
        self.is_completed = False
        self.final_answer: Optional[str] = None
        self.task: Optional[asyncio.Task] = None

    def add_subscriber(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        # Replay past events to this subscriber
        for ev in self.event_history:
            q.put_nowait(ev)
        self.subscribers.add(q)
        return q

    def remove_subscriber(self, q: asyncio.Queue):
        self.subscribers.discard(q)

    def broadcast(self, event: Dict[str, Any]):
        # Tag with session_id for client multi-session routing
        event["session_id"] = self.session_id
        self.event_history.append(event)
        for q in list(self.subscribers):
            try:
                q.put_nowait(event)
            except Exception:
                pass


class QueryManager:
    """Singleton managing active detached query executions across sessions."""

    _instance = None
    _active_queries: Dict[str, ActiveSessionQuery] = {}
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QueryManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def _make_key(cls, user_id: str, session_id: str) -> str:
        return f"{user_id}:{session_id}"

    @classmethod
    async def get_active_query(cls, user_id: str, session_id: str) -> Optional[ActiveSessionQuery]:
        key = cls._make_key(user_id, session_id)
        return cls._active_queries.get(key)

    @classmethod
    async def is_session_generating(cls, user_id: str, session_id: str) -> bool:
        key = cls._make_key(user_id, session_id)
        q = cls._active_queries.get(key)
        return q is not None and not q.is_completed

    @classmethod
    async def start_query(
        cls,
        user_id: str,
        session_id: str,
        query: str,
        collection_scope: List[str],
        message_attachments: Optional[List[Dict[str, Any]]] = None,
        app_graph: Any = None,
    ) -> ActiveSessionQuery:
        """Launches a shielded background generation task for the session."""
        key = cls._make_key(user_id, session_id)

        # Cancel any previous in-flight task for this exact session
        if key in cls._active_queries:
            prev = cls._active_queries[key]
            if prev.task and not prev.task.done():
                prev.task.cancel()

        active_query = ActiveSessionQuery(user_id=user_id, session_id=session_id, query=query)
        cls._active_queries[key] = active_query

        # Launch detached background task
        task = asyncio.create_task(
            cls._run_query_workflow(
                active_query=active_query,
                collection_scope=collection_scope,
                message_attachments=message_attachments,
                app_graph=app_graph,
            )
        )
        active_query.task = task

        # Cleanup when done after 30 seconds
        def _cleanup(_):
            active_query.is_completed = True
            asyncio.create_task(cls._delayed_evict(key, delay=30))

        task.add_done_callback(_cleanup)
        return active_query

    @classmethod
    async def _delayed_evict(cls, key: str, delay: int = 30):
        await asyncio.sleep(delay)
        cls._active_queries.pop(key, None)

    @classmethod
    async def _run_query_workflow(
        cls,
        active_query: ActiveSessionQuery,
        collection_scope: List[str],
        message_attachments: Optional[List[Dict[str, Any]]],
        app_graph: Any,
    ):
        """Executes the StateGraph inside an asyncio.shield block."""
        user_id = active_query.user_id
        session_id = active_query.session_id
        query = active_query.query

        full_clean_answer = ""
        full_final_answer = ""
        full_citations = ""
        last_stage = None
        has_tool_run = False
        tools_active_count = 0
        pre_tool_tokens = []
        token_buffer = ""
        is_buffering = True
        suppress_stream = False
        streamed_any_tokens = False

        config = {
            "configurable": {
                "thread_id": session_id,
                "session_id": session_id,
                "user_id": user_id,
            },
            "metadata": {
                "session_id": session_id,
                "user_id": user_id,
                "thread_id": session_id,
                "query": query[:120],
            },
            "run_name": f"Chat Turn [{session_id[:8]}]: {query[:35]}",
            "tags": [
                "docuvortex",
                "rag",
                f"session:{session_id}",
                f"user:{user_id}",
            ],
        }

        graph = app_graph if app_graph is not None else rag_graph
        graph_input = {
            "question": query,
            "collection_names": collection_scope,
            "session_id": session_id,
            "user_id": user_id,
            "message_attachments": message_attachments,
            "source_selected": bool(collection_scope),
        }

        try:
            # Shield execution from cancellations caused by client disconnects
            async def _execute():
                nonlocal full_clean_answer, full_final_answer, full_citations, last_stage, has_tool_run, tools_active_count
                nonlocal token_buffer, is_buffering, suppress_stream, streamed_any_tokens

                target_graph = graph
                iterator = target_graph.astream_events(graph_input, config=config, version="v2").__aiter__()

                async def _safe_stream_events():
                    nonlocal target_graph, iterator
                    try:
                        first = await iterator.__anext__()
                        yield first
                    except StopAsyncIteration:
                        return
                    except Exception as err:
                        err_msg = str(err).lower()
                        if target_graph is not rag_graph and (
                            "closed the connection unexpectedly" in err_msg
                            or "operationalerror" in err_msg
                            or "connection" in err_msg
                            or "checkpointer" in err_msg
                            or "consuming input failed" in err_msg
                            or "bad" in err_msg
                        ):
                            logging.warning(
                                "Checkpointer connection dropped (%s). Falling back to direct rag_graph", err
                            )
                            async for ev in rag_graph.astream_events(graph_input, config=config, version="v2"):
                                yield ev
                            return
                        else:
                            raise

                    async for ev in iterator:
                        yield ev

                async for event in _safe_stream_events():
                    kind = event.get("event", "")
                    name = event.get("name", "")

                    # 1. Status event
                    if kind == "on_chain_start" and name in _STAGE_LABELS:
                        if not streamed_any_tokens:
                            stage, message = _STAGE_LABELS[name]
                            if stage != last_stage:
                                last_stage = stage
                                active_query.broadcast({
                                    "type": "status",
                                    "stage": stage,
                                    "message": message,
                                })

                    # 2. Tool Start event
                    elif kind == "on_tool_start":
                        has_tool_run = True
                        tools_active_count += 1
                        pre_tool_tokens.clear()

                        tool_name = name or event.get("metadata", {}).get("langgraph_node", "tool")
                        tool_input = event.get("data", {}).get("input", {})
                        badge_type, msg = TOOL_DISPLAY_MAP.get(tool_name, ("tool", f"🔧 Using {tool_name}..."))

                        if isinstance(tool_input, dict):
                            if tool_name in ("get_stock_price", "stock_price_tool"):
                                sym = tool_input.get("symbol") or tool_input.get("ticker") or tool_input.get("company")
                                msg = f"📈 Fetching stock price for {str(sym).strip().upper()}..." if sym else "📈 Fetching stock price..."
                            elif tool_name in ("get_weather",):
                                loc = tool_input.get("location") or tool_input.get("city")
                                msg = f"🌤️ Checking weather in {str(loc).strip().title()}..." if loc else "🌤️ Checking weather..."
                            elif tool_name in ("search_arxiv",):
                                q = str(tool_input.get("query") or "")
                                msg = f"📚 Searching arXiv for '{(q[:28] + '...') if len(q) > 28 else q}'..." if q else "📚 Searching academic papers..."
                            elif tool_name in ("calculator", "calculator_tool"):
                                op = str(tool_input.get("operation") or "")
                                n1 = tool_input.get("first_num", "")
                                n2 = tool_input.get("second_num", "")
                                msg = f"🧮 Calculating {n1} {op} {n2}..." if (n1 != "" and n2 != "") else "🧮 Calculating..."
                            elif tool_name in ("search_tool", "duckduckgo_search"):
                                q = str(tool_input.get("query") or "")
                                msg = f"🔍 Searching web for '{(q[:28] + '...') if len(q) > 28 else q}'..." if q else "🔍 Searching the web..."
                        elif isinstance(tool_input, str) and tool_input.strip():
                            clean_str = tool_input.strip()
                            if tool_name in ("get_stock_price", "stock_price_tool"):
                                msg = f"📈 Fetching stock price for {clean_str.upper()}..."
                            elif tool_name == "get_weather":
                                msg = f"🌤️ Checking weather in {clean_str.title()}..."
                            elif tool_name in ("search_tool", "duckduckgo_search"):
                                msg = f"🔍 Searching web for '{(clean_str[:28] + '...') if len(clean_str) > 28 else clean_str}'..."

                        active_query.broadcast({
                            "type": "tool_status",
                            "tool": tool_name,
                            "badge_type": badge_type,
                            "message": msg,
                        })

                    # 3. Tool End event
                    elif kind == "on_tool_end":
                        tools_active_count = max(0, tools_active_count - 1)
                        tool_name = name or "tool"
                        active_query.broadcast({
                            "type": "tool_end",
                            "tool": tool_name,
                        })

                    # 4. Real-time token streaming from generation / chitchat agents
                    elif kind == "on_chat_model_stream":
                        node = event.get("metadata", {}).get("langgraph_node")
                        if node in ("generate", "chitchat", "chitchat_agent"):
                            chunk = event.get("data", {}).get("chunk")
                            tok = getattr(chunk, "text", "") or getattr(chunk, "content", "")
                            if tok:
                                if is_buffering:
                                    token_buffer += tok
                                    if "DATA_NOT_FOUND" in token_buffer or '{"type":' in token_buffer:
                                        suppress_stream = True
                                    elif len(token_buffer) >= 20 or "\n" in token_buffer:
                                        is_buffering = False
                                        if not suppress_stream:
                                            streamed_any_tokens = True
                                            active_query.broadcast({
                                                "type": "token",
                                                "content": token_buffer,
                                            })
                                else:
                                    if not suppress_stream:
                                        streamed_any_tokens = True
                                        active_query.broadcast({
                                            "type": "token",
                                            "content": tok,
                                        })

                    # 5. Final output extraction
                    elif kind == "on_chain_end":
                        output = event.get("data", {}).get("output", {})
                        if isinstance(output, dict):
                            if output.get("clean_answer"):
                                full_clean_answer = output.get("clean_answer", full_clean_answer)
                            if output.get("final_answer"):
                                full_final_answer = output.get("final_answer", full_final_answer)
                            if output.get("citations"):
                                full_citations = output.get("citations", full_citations)

            await asyncio.shield(_execute())

            # Flush any unsuppressed tokens remaining in the initial buffer
            if is_buffering and token_buffer and not suppress_stream:
                streamed_any_tokens = True
                active_query.broadcast({
                    "type": "token",
                    "content": token_buffer,
                })

            if not full_final_answer:
                full_final_answer = "I couldn't find information about that in the uploaded document(s)."

            # Check for clarification format
            is_clarification = False
            clean_ans = full_final_answer.strip()
            if clean_ans.startswith("{") and clean_ans.endswith("}"):
                try:
                    c_data = json.loads(clean_ans)
                    if isinstance(c_data, dict) and c_data.get("type") == "clarification":
                        is_clarification = True
                        active_query.broadcast({
                            "type": "clarification",
                            "message": c_data.get("message", "Could you clarify your question?"),
                            "options": c_data.get("options", []),
                        })
                except Exception:
                    pass

            if not is_clarification and not streamed_any_tokens:
                # Stream token playback for pure answer body (without citations to avoid duplicate display)
                playback_text = full_clean_answer or full_final_answer
                playback_text = re.split(r"(?i)\n\s*(sources?|citations?)\s*:\s*\n", playback_text)[0].strip()

                chunks = re.findall(r"\S+|\s+", playback_text)
                total_chunks = len(chunks)
                if total_chunks > 0:
                    delay = 0.008 if total_chunks <= 150 else 0.003
                    step = 1 if total_chunks <= 150 else 2
                    for i in range(0, total_chunks, step):
                        chunk_slice = "".join(chunks[i : i + step])
                        active_query.broadcast({
                            "type": "token",
                            "content": chunk_slice,
                        })
                        await asyncio.sleep(delay)

            if full_citations and full_final_answer != "I couldn't find information about that in the uploaded document(s).":
                active_query.broadcast({
                    "type": "citations",
                    "citations": full_citations,
                })

            active_query.broadcast({
                "type": "done",
                "status": "completed",
                "final_answer": full_final_answer,
                "citations": full_citations,
            })
            active_query.final_answer = full_final_answer

            logging.info(
                "<<< [LIVE QUERY COMPLETED] session: %s | user: %s\nFinal Answer:\n%s%s",
                session_id,
                user_id,
                full_final_answer,
                f"\nCitations: {full_citations}" if full_citations else "",
            )

        except asyncio.CancelledError:
            logging.info("Query task cancelled for session %s", session_id)
            active_query.broadcast({
                "type": "error",
                "error_code": "cancelled",
                "message": "Query generation was cancelled.",
            })
            raise
        except LLMRateLimitError as exc:
            logging.warning("Rate limit hit during generation: %s", exc)
            active_query.broadcast({
                "type": "error",
                "error_code": f"llm_rate_limit_{exc.kind}",
                "message": exc.message,
            })
        except EmbeddingUnavailableError as exc:
            logging.warning("Embedding service unavailable: %s", exc)
            active_query.broadcast({
                "type": "error",
                "error_code": "embedding_unavailable",
                "message": exc.message,
            })
        except CollectionEmbeddingMismatchError as exc:
            logging.warning("Collection embedding mismatch: %s", exc)
            active_query.broadcast({
                "type": "error",
                "error_code": "collection_needs_reindex",
                "message": exc.message,
            })
        except Exception as exc:
            logging.exception("Exception in background query workflow for session %s", session_id)
            err_str = str(exc).lower()
            if "embedding" in err_str and "unavailable" in err_str:
                msg = "The embedding model is unavailable. You can switch the embedding model in Settings."
                code = "embedding_unavailable"
            elif "vector dimension" in err_str or "dimension error" in err_str or "mismatch" in err_str:
                msg = "This document was indexed with a different embedding model. Please switch your embedding model in Settings or re-upload the document."
                code = "collection_needs_reindex"
            else:
                msg = "I encountered an error generating the response. Please try again."
                code = "generation_failed"

            active_query.broadcast({
                "type": "error",
                "error_code": code,
                "message": msg,
            })
        finally:
            active_query.is_completed = True

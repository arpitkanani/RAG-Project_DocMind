"""Query endpoint — streams structured SSE events from the DocuVortex
agentic RAG StateGraph."""

import asyncio
import json
import re
import sys

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from src.auth import get_current_user
from src.components.memory_manager import MemoryManager
from src.exception import CollectionNotFoundError, CustomException, KnowledgeBaseEmptyError
from src.graph.builder import rag_graph
from src.logger import logging
from src.utils.helpers import (
    aresolve_session_scope,
    build_error_response,
    extract_text,
    normalize_collection_scope,
)
from src.schemas import QueryRequest
from src.utils.rate_limiter import LLMRateLimitError

router = APIRouter(tags=["Query"])

# ── Stage labels for SSE status events ──────────────────────────────────────
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
    "finalize": ("finalizing", "Finalizing answer..."),
}


@router.post("/query")
async def query(
    request: QueryRequest,
    raw_request: Request,
    user_id: str = Depends(get_current_user),
):
    try:
        session_id = request.session_id or "default"
        logging.info(
            "Query received: %s... | session: %s | user: %s",
            request.query[:50],
            session_id,
            user_id,
        )

        # Resolve authorized collections for this user and session
        try:
            collection_scope = await aresolve_session_scope(
                session_id,
                normalize_collection_scope(request),
                user_id,
            )
        except (KnowledgeBaseEmptyError, CollectionNotFoundError, Exception):
            # If no collections exist, invalid collection, or vectorstore offline, allow conversational chitchat through graph
            collection_scope = []

        async def event_generator():
            full_final_answer = ""
            last_stage = None
            streamed_any_token = False

            # Stream suppression flags for intermediate tool execution
            tools_active_count = 0
            has_tool_run = False
            pre_tool_tokens = []  # Buffer tokens emitted prior to any tool call

            config = {
                "configurable": {
                    "thread_id": session_id,
                    "session_id": session_id,
                    "user_id": user_id,
                }
            }

            try:
                # Stream events from the StateGraph (with automatic direct fallback if checkpointer connection dropped)
                graph = getattr(raw_request.app.state, "rag_graph", rag_graph)
                graph_input = {
                    "question": request.query,
                    "collection_names": collection_scope,
                    "session_id": session_id,
                    "user_id": user_id,
                    "message_attachments": request.message_attachments,
                    "source_selected": bool(collection_scope),
                }

                async def _safe_stream_events():
                    target_graph = graph
                    iterator = target_graph.astream_events(graph_input, config=config, version="v2").__aiter__()
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
                                "Checkpointer graph connection dropped (%s). Seamlessly falling back to direct rag_graph",
                                err,
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

                    # ── Status events: emit when entering a new graph node ──
                    if kind == "on_chain_start" and name in _STAGE_LABELS:
                        stage, message = _STAGE_LABELS[name]
                        if stage != last_stage:
                            last_stage = stage
                            payload = json.dumps({
                                "type": "status",
                                "stage": stage,
                                "message": message,
                            })
                            yield f"data: {payload}\n\n"

                    # ── Tool Status Event: Mimicking Streamlit st.status ──
                    elif kind == "on_tool_start":
                        has_tool_run = True
                        tools_active_count += 1
                        pre_tool_tokens.clear()  # Discard any intermediate preamble tokens streamed/buffered prior to tool call

                        tool_name = name or event.get("metadata", {}).get("langgraph_node", "tool")
                        tool_input = event.get("data", {}).get("input", {})
                        tool_display_map = {
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
                        badge_type, msg = tool_display_map.get(tool_name, ("tool", f"🔧 Using {tool_name}..."))

                        # Enrich message dynamically with input details
                        if isinstance(tool_input, dict):
                            if tool_name in ("get_stock_price", "stock_price_tool"):
                                sym = tool_input.get("symbol") or tool_input.get("ticker") or tool_input.get("company")
                                if sym:
                                    sym_str = str(sym).strip().upper()
                                    msg = f"📈 Fetching stock price for {sym_str}..."
                                else:
                                    msg = "📈 Fetching stock price..."
                            elif tool_name in ("get_weather",):
                                loc = tool_input.get("location") or tool_input.get("city")
                                if loc:
                                    loc_str = str(loc).strip().title()
                                    msg = f"🌤️ Checking weather in {loc_str}..."
                                else:
                                    msg = "🌤️ Checking weather..."
                            elif tool_name in ("search_arxiv",):
                                q = str(tool_input.get("query") or "")
                                q_trunc = (q[:28] + "...") if len(q) > 28 else q
                                msg = f"📚 Searching arXiv for '{q_trunc}'..." if q else "📚 Searching academic papers..."
                            elif tool_name in ("calculator", "calculator_tool"):
                                op = str(tool_input.get("operation") or "")
                                n1 = tool_input.get("first_num", "")
                                n2 = tool_input.get("second_num", "")
                                if n1 != "" and n2 != "":
                                    msg = f"🧮 Calculating {n1} {op} {n2}..."
                                else:
                                    msg = "🧮 Calculating..."
                            elif tool_name in ("search_tool", "duckduckgo_search"):
                                q = str(tool_input.get("query") or "")
                                q_trunc = (q[:28] + "...") if len(q) > 28 else q
                                msg = f"🔍 Searching web for '{q_trunc}'..." if q else "🔍 Searching the web..."
                        elif isinstance(tool_input, str) and tool_input.strip():
                            clean_str = tool_input.strip()
                            if tool_name in ("get_stock_price", "stock_price_tool"):
                                msg = f"📈 Fetching stock price for {clean_str.upper()}..."
                            elif tool_name == "get_weather":
                                msg = f"🌤️ Checking weather in {clean_str.title()}..."
                            elif tool_name in ("search_tool", "duckduckgo_search"):
                                q_trunc = (clean_str[:28] + "...") if len(clean_str) > 28 else clean_str
                                msg = f"🔍 Searching web for '{q_trunc}'..."
                            elif tool_name == "search_arxiv":
                                q_trunc = (clean_str[:28] + "...") if len(clean_str) > 28 else clean_str
                                msg = f"📚 Searching arXiv for '{q_trunc}'..."

                        tool_payload = json.dumps({
                            "type": "tool_status",
                            "tool": tool_name,
                            "badge_type": badge_type,
                            "message": msg,
                        })
                        yield f"data: {tool_payload}\n\n"

                    elif kind == "on_tool_end":
                        tools_active_count = max(0, tools_active_count - 1)
                        tool_name = name or "tool"
                        end_payload = json.dumps({
                            "type": "tool_end",
                            "tool": tool_name,
                        })
                        yield f"data: {end_payload}\n\n"

                    # ── Capture final answer from finalize or direct nodes ──
                    elif kind == "on_chain_end":
                        output = event.get("data", {}).get("output", {})
                        if isinstance(output, dict) and output.get("final_answer"):
                            full_final_answer = output.get("final_answer", full_final_answer)

                if not full_final_answer:
                    full_final_answer = "I couldn't find information about that in the uploaded document(s)."

                # Check if this is a structured clarification response
                is_clarification = False
                try:
                    parsed = json.loads(full_final_answer)
                    if isinstance(parsed, dict) and parsed.get("type") == "clarification":
                        is_clarification = True
                        clarify_payload = json.dumps({
                            "type": "clarification",
                            "message": parsed.get("message", ""),
                            "options": parsed.get("options", []),
                        })
                        yield f"data: {clarify_payload}\n\n"
                        full_final_answer = ""  # Clear so done event doesn't carry raw JSON
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass

                # Stream the completely structured answer token-by-token
                if not is_clarification and full_final_answer:
                    # Break into tokens/words to create smooth typewriter streaming of the finalized answer
                    chunks = re.findall(r"\S+|\s+", full_final_answer)
                    if chunks:
                        total_chunks = len(chunks)
                        if total_chunks <= 40:
                            step = 1
                            delay = 0.015
                        elif total_chunks <= 120:
                            step = 2
                            delay = 0.012
                        else:
                            step = max(2, total_chunks // 60)
                            delay = 0.010

                        for i in range(0, total_chunks, step):
                            chunk_slice = "".join(chunks[i : i + step])
                            payload = json.dumps({"type": "token", "content": chunk_slice})
                            yield f"data: {payload}\n\n"
                            await asyncio.sleep(delay)

                # Emit done event
                done_payload = json.dumps({
                    "type": "done",
                    "status": "completed",
                    "final_answer": full_final_answer,
                    "session_id": session_id,
                })
                yield f"data: {done_payload}\n\n"

            except LLMRateLimitError as exc:
                logging.warning("LLM rate limit during streaming: %s", exc)
                err_payload = json.dumps({
                    "type": "error",
                    "error_code": f"llm_rate_limit_{exc.kind}",
                    "message": exc.message,
                })
                yield f"data: {err_payload}\n\n"
            except Exception as exc:
                logging.exception("Error during graph streaming")
                err_payload = json.dumps({
                    "type": "error",
                    "error_code": "generation_failed",
                    "message": "I encountered an error generating the response. Please try again.",
                })
                yield f"data: {err_payload}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    except LLMRateLimitError as exc:
        logging.warning("LLM rate limit hit during query | kind: %s", exc.kind)
        return build_error_response(
            status_code=429,
            error_code=f"llm_rate_limit_{exc.kind}",
            message=exc.message,
        )
    except CollectionNotFoundError as exc:
        logging.exception("Query failed because collection is missing")
        missing = getattr(exc, "missing_collections", [])
        return build_error_response(
            status_code=404,
            error_code="collection_not_found",
            message="This document was removed. Please upload a document to continue.",
            extra={"missing_collections": missing},
        )
    except KnowledgeBaseEmptyError:
        logging.exception("Query failed because the knowledge base is empty")
        return build_error_response(
            status_code=404,
            error_code="knowledge_base_empty",
            message="Please upload a document or add a YouTube video first.",
        )
    except CustomException as exc:
        logging.exception("Query failed with application error")
        return build_error_response(
            status_code=400,
            error_code="query_failed",
            message="I couldn't complete that request right now. Please try again.",
        )
    except Exception as exc:
        logging.exception("Query failed")
        return build_error_response(
            status_code=500,
            error_code="server_error",
            message="Server is down. Please try again.",
        )

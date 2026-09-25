"""Query endpoint — streams structured SSE events from the DocuVortex
agentic RAG StateGraph."""

import json
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
    normalize_collection_scope,
)
from src.schemas import QueryRequest
from src.utils.rate_limiter import LLMRateLimitError

router = APIRouter(tags=["Query"])

# ── Stage labels for SSE status events ──────────────────────────────────────
_STAGE_LABELS = {
    "load_context": ("classifying", "Analyzing your question..."),
    "classify_intent": ("classifying", "Analyzing your question..."),
    "clarify_question": ("classifying", "Checking question clarity..."),
    "chitchat": ("thinking", "Thinking..."),
    "run_react_agent": ("thinking", "Thinking..."),
    "agent": ("thinking", "Thinking..."),
    "tools": ("tool", "Executing tool..."),
    "retrieve_qa": ("retrieving", "Retrieving documents..."),
    "retrieve_summary": ("retrieving", "Retrieving document context..."),
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
        except (KnowledgeBaseEmptyError, CollectionNotFoundError):
            # If no collections exist or invalid collection, allow conversational chitchat through graph
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
                # Stream events from the StateGraph
                async for event in rag_graph.astream_events(
                    {
                        "question": request.query,
                        "collection_names": collection_scope,
                        "session_id": session_id,
                        "user_id": user_id,
                        "message_attachments": request.message_attachments,
                        "source_selected": bool(collection_scope),
                    },
                    config=config,
                    version="v2",
                ):
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
                            if tool_name in ("get_stock_price", "stock_price_tool") and tool_input.get("symbol"):
                                sym = str(tool_input["symbol"]).strip().upper()
                                msg = f"📈 Fetching stock price for {sym}..."
                            elif tool_name in ("get_weather",) and tool_input.get("location"):
                                loc = str(tool_input["location"]).strip()
                                msg = f"🌤️ Checking weather in {loc}..."
                            elif tool_name in ("search_arxiv",) and tool_input.get("query"):
                                q = str(tool_input["query"])
                                q_trunc = (q[:28] + "...") if len(q) > 28 else q
                                msg = f"📚 Searching arXiv for '{q_trunc}'..."
                            elif tool_name in ("calculator", "calculator_tool") and tool_input.get("operation"):
                                op = str(tool_input.get("operation"))
                                n1 = tool_input.get("first_num", "")
                                n2 = tool_input.get("second_num", "")
                                msg = f"🧮 Calculating {n1} {op} {n2}..."
                            elif tool_name in ("search_tool", "duckduckgo_search") and tool_input.get("query"):
                                q = str(tool_input["query"])
                                q_trunc = (q[:28] + "...") if len(q) > 28 else q
                                msg = f"🔍 Searching web for '{q_trunc}'..."
                        elif isinstance(tool_input, str):
                            if tool_name in ("search_tool", "duckduckgo_search"):
                                q_trunc = (tool_input[:28] + "...") if len(tool_input) > 28 else tool_input
                                msg = f"🔍 Searching web for '{q_trunc}'..."
                            elif tool_name == "get_weather":
                                loc = tool_input.strip()
                                msg = f"🌤️ Checking weather in {loc}..."
                            elif tool_name == "search_arxiv":
                                q_trunc = (tool_input[:28] + "...") if len(tool_input) > 28 else tool_input
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

                    # ── Token streaming from the LLM generation ──
                    elif kind == "on_chat_model_stream":
                        node_name = event.get("metadata", {}).get("langgraph_node", "")
                        tags = event.get("tags", [])

                        # Prevent stream leakage from memory summarizer, grading, intent classification, etc.
                        # Allow generate, chitchat, run_react_agent, agent, chitchat_agent, and structure_answer
                        allowed_stream_nodes = {
                            "generate",
                            "chitchat",
                            "run_react_agent",
                            "agent",
                            "chitchat_agent",
                            "structure_answer",
                        }
                        if "memory_summary" in tags or node_name not in allowed_stream_nodes:
                            continue

                        # Suppress token streaming while a tool is currently executing
                        if tools_active_count > 0:
                            continue

                        chunk = event.get("data", {}).get("chunk")
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            # Skip tool call argument chunks
                            if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                                continue
                            content = chunk.content
                            if isinstance(content, list):
                                # Gemini-style content blocks or text blocks
                                content = "".join(
                                    block.get("text", "") if isinstance(block, dict) else str(block)
                                    for block in content
                                )
                            if content:
                                # If this is a ReAct agent node (e.g. agent/run_react_agent/chitchat)
                                if node_name in {"run_react_agent", "agent", "chitchat", "chitchat_agent"}:
                                    if not has_tool_run:
                                        # Before any tool is called, buffer tokens so thoughts/preambles aren't leaked
                                        # if a tool is about to be triggered
                                        pre_tool_tokens.append(content)
                                        # If buffer gets reasonably large without a tool call, this is pure chitchat
                                        if len(pre_tool_tokens) > 12:
                                            while pre_tool_tokens:
                                                buffered = pre_tool_tokens.pop(0)
                                                payload = json.dumps({"type": "token", "content": buffered})
                                                yield f"data: {payload}\n\n"
                                        continue
                                    else:
                                        # Post-tool final synthesis: stream directly
                                        payload = json.dumps({"type": "token", "content": content})
                                        yield f"data: {payload}\n\n"
                                else:
                                    # Standard RAG generation node (generate, structure_answer)
                                    streamed_any_token = True
                                    payload = json.dumps({"type": "token", "content": content})
                                    yield f"data: {payload}\n\n"

                    # ── Capture final answer from finalize or direct nodes ──
                    elif kind == "on_chain_end":
                        output = event.get("data", {}).get("output", {})
                        if isinstance(output, dict) and output.get("final_answer"):
                            full_final_answer = output.get("final_answer", full_final_answer)

                # Flush any remaining buffered chitchat tokens if no tools were ever called
                if not has_tool_run and pre_tool_tokens:
                    while pre_tool_tokens:
                        buffered = pre_tool_tokens.pop(0)
                        streamed_any_token = True
                        payload = json.dumps({"type": "token", "content": buffered})
                        yield f"data: {payload}\n\n"

                # If no tokens were streamed (e.g. direct clarify or fallback response), emit full_final_answer as token
                if not streamed_any_token and full_final_answer:
                    payload = json.dumps({"type": "token", "content": full_final_answer})
                    yield f"data: {payload}\n\n"

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

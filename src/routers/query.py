"""Query endpoint — streams structured SSE events from the DocuVortex
agentic RAG StateGraph."""

import json
import sys

from fastapi import APIRouter, Depends
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
    "chitchat": ("structuring", "Structuring answer..."),
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

                    # ── Token streaming from the LLM generation ──
                    elif kind == "on_chat_model_stream":
                        node_name = event.get("metadata", {}).get("langgraph_node", "")
                        tags = event.get("tags", [])

                        # Prevent stream leakage from memory summarizer, grading, intent classification, etc.
                        if "memory_summary" in tags or node_name not in ["generate", "chitchat"]:
                            continue

                        chunk = event.get("data", {}).get("chunk")
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            # Skip tool call argument chunks
                            if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                                continue
                            content = chunk.content
                            if isinstance(content, list):
                                # Gemini-style content blocks
                                content = "".join(
                                    block.get("text", "") if isinstance(block, dict) else str(block)
                                    for block in content
                                )
                            if content:
                                payload = json.dumps({"type": "token", "content": content})
                                yield f"data: {payload}\n\n"

                    # ── Capture final answer from finalize or direct nodes ──
                    elif kind == "on_chain_end":
                        output = event.get("data", {}).get("output", {})
                        if isinstance(output, dict) and output.get("final_answer"):
                            full_final_answer = output.get("final_answer", full_final_answer)

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

import json
import sys

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.auth import get_current_user
from src.components.memory_manager import MemoryManager
from src.exception import CollectionNotFoundError, CustomException, KnowledgeBaseEmptyError
from src.graph.agent import astream_agent_response
from src.logger import logging
from src.routing_helpers import (
    aresolve_session_scope,
    build_error_response,
    normalize_collection_scope,
)
from src.schemas import QueryRequest
from src.utils.rate_limiter import LLMRateLimitError

router = APIRouter(tags=["Query"])


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
        collection_scope = await aresolve_session_scope(
            session_id,
            normalize_collection_scope(request),
            user_id,
        )

        memory = MemoryManager(session_id=session_id, user_id=user_id)
        chat_history = await memory.aget_history()

        # Persist user query to Postgres
        await memory.asave_message(
            "human",
            request.query,
            attachments=request.message_attachments,
        )

        async def event_generator():
            full_final_answer = ""
            try:
                async for event in astream_agent_response(
                    query=request.query,
                    chat_history=chat_history,
                    collection_names=collection_scope,
                    session_id=session_id,
                    user_id=user_id,
                ):
                    event_type = event.get("type")

                    if event_type == "token":
                        payload = json.dumps({"type": "token", "content": event["content"]})
                        yield f"data: {payload}\n\n"

                    elif event_type == "tool_start":
                        payload = json.dumps({
                            "type": "tool_start",
                            "name": event["name"],
                            "input": event.get("input", {}),
                        })
                        yield f"data: {payload}\n\n"

                    elif event_type == "tool_end":
                        payload = json.dumps({
                            "type": "tool_end",
                            "name": event["name"],
                        })
                        yield f"data: {payload}\n\n"

                    elif event_type == "done":
                        full_final_answer = event.get("final_answer", "")

                # Persist the finalized AI response to PostgreSQL
                if full_final_answer:
                    await memory.asave_message("ai", full_final_answer)

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
                logging.exception("Error during agent response streaming")
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

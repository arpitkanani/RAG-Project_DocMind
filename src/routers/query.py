# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Depends
# pyrefly: ignore [missing-import]
from fastapi.responses import StreamingResponse

from src.auth import get_current_user
from src.exception import CollectionNotFoundError, CustomException, KnowledgeBaseEmptyError
from src.logger import logging
from src.pipelines.qa_pipeline import QAPipeline
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
        logging.info("Query received: %s...", request.query[:50])

        collection_scope = await aresolve_session_scope(
            request.session_id or "default",
            normalize_collection_scope(request),
            user_id,
        )
        pipeline = QAPipeline(
            collection_names=collection_scope,
            session_id=request.session_id or "default",
            user_id=user_id,
        )

        return StreamingResponse(
            pipeline.astream(
                request.query,
                message_attachments=request.message_attachments,
            ),
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

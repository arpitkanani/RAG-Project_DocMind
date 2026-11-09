"""Query endpoint — streams structured SSE events from the DocuVortex
agentic RAG StateGraph."""

import asyncio
import json
import re
import sys

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from src.auth import get_current_user
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
from src.utils.helpers import (
    aresolve_session_scope,
    build_error_response,
    extract_text,
    normalize_collection_scope,
)
from src.schemas import QueryRequest
from src.utils.query_manager import QueryManager
from src.utils.rate_limiter import LLMRateLimitError

router = APIRouter(tags=["Query"])


@router.post("/query")
async def query(
    request: QueryRequest,
    raw_request: Request,
    user_id: str = Depends(get_current_user),
):
    try:
        session_id = request.session_id or "default"
        logging.info(
            ">>> [LIVE USER QUERY] user: %s | session: %s | collections: %s\nQuery: %s",
            user_id,
            session_id,
            request.collection_name or request.collection_names or "None",
            request.query,
        )

        # Check active embedding model specs
        active_provider = Embedder.get_active_provider(user_id=user_id)
        active_specs = Embedder.get_model_specs(active_provider)

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

        # Validate that no requested collections have an embedding model / dimension mismatch
        if collection_scope:
            mismatched = []
            memory = MemoryManager(session_id=session_id, user_id=user_id)
            attachments = await memory.aget_attachments()
            att_map = {a["collection"]: a for a in attachments if "collection" in a}

            for col in collection_scope:
                att = att_map.get(col)
                indexed_dim = att.get("vector_dim") if att else None
                # If vector_dim not recorded in attachments table, inspect Qdrant directly
                if not indexed_dim:
                    indexed_dim = VectorStore(collection_name=col).get_collection_vector_dimension(col)

                if indexed_dim and indexed_dim != active_specs["dimension"]:
                    doc_name = att.get("name", col) if att else col
                    indexed_model = att.get("embedding_model") if (att and att.get("embedding_model")) else ("BAAI/bge-small-en-v1.5" if indexed_dim == 384 else "text-embedding-3-small")
                    mismatched.append({
                        "collection": col,
                        "name": doc_name,
                        "indexed_model": indexed_model,
                        "indexed_dim": indexed_dim,
                        "current_model": active_specs["model"],
                        "current_dim": active_specs["dimension"],
                    })

            if mismatched:
                logging.warning("Blocked query due to embedding model mismatch for collections: %s", mismatched)
                first_m = mismatched[0]
                if len(mismatched) == 1:
                    detail_msg = (
                        f"The document '{first_m['name']}' was indexed using '{first_m['indexed_model']}' ({first_m['indexed_dim']}d), "
                        f"but your currently active model is '{first_m['current_model']}' ({first_m['current_dim']}d). "
                        f"Please switch to '{first_m['indexed_model']}' from Settings or re-upload the document."
                    )
                else:
                    m_names = ", ".join(f"'{m['name']}'" for m in mismatched)
                    detail_msg = (
                        f"The documents {m_names} were indexed using '{first_m['indexed_model']}' ({first_m['indexed_dim']}d), "
                        f"but your currently active model is '{first_m['current_model']}' ({first_m['current_dim']}d). "
                        f"Please switch to '{first_m['indexed_model']}' from Settings or re-upload the documents."
                    )
                return build_error_response(
                    status_code=400,
                    error_code="collection_needs_reindex",
                    message=detail_msg,
                    extra={"mismatched_collections": mismatched},
                )

        async def event_generator():
            graph = getattr(raw_request.app.state, "rag_graph", rag_graph)
            active_q = await QueryManager.start_query(
                user_id=user_id,
                session_id=session_id,
                query=request.query,
                collection_scope=collection_scope,
                message_attachments=request.message_attachments,
                app_graph=graph,
            )
            queue = active_q.add_subscriber()
            try:
                while True:
                    event = await queue.get()
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("type") in ("done", "error"):
                        break
            except asyncio.CancelledError:
                logging.info(
                    "SSE client disconnected from session %s; background task continues shielded",
                    session_id,
                )
            finally:
                active_q.remove_subscriber(queue)

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
    except EmbeddingUnavailableError as exc:
        logging.warning("Embedding service unavailable: %s", exc)
        return build_error_response(
            status_code=503,
            error_code="embedding_unavailable",
            message=exc.message,
        )
    except CollectionEmbeddingMismatchError as exc:
        logging.warning("Collection embedding mismatch: %s", exc)
        return build_error_response(
            status_code=400,
            error_code="collection_needs_reindex",
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
        err_str = str(exc)
        if "vector dimension" in err_str.lower() or "dimension error" in err_str.lower() or "wrong input" in err_str.lower():
            return build_error_response(
                status_code=400,
                error_code="collection_needs_reindex",
                message="This document was indexed with a different embedding model. Please switch your embedding model in Settings or re-upload the document.",
            )
        return build_error_response(
            status_code=400,
            error_code="query_failed",
            message="I couldn't complete that request right now. Please try again.",
        )
    except Exception as exc:
        logging.exception("Query failed")
        err_str = str(exc)
        if "vector dimension" in err_str.lower() or "dimension error" in err_str.lower() or "wrong input" in err_str.lower():
            return build_error_response(
                status_code=400,
                error_code="collection_needs_reindex",
                message="This document was indexed with a different embedding model. Please switch your embedding model in Settings or re-upload the document.",
            )
        return build_error_response(
            status_code=500,
            error_code="server_error",
            message="Server is down. Please try again.",
        )

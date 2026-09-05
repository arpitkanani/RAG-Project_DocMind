import asyncio
import json
import os
import sys
from typing import AsyncGenerator, Dict, List, Optional
import uuid

from langchain_core.messages import AIMessageChunk

from src.chains.qa_chain import (
    FALLBACK_ANSWER,
    QA_PROMPT,
    _build_llm,
    build_citations,
    build_source_only_citations,
    is_summary_request,
    sanitize_answer,
)
from src.components.memory_manager import MemoryManager
from src.exception import (
    CollectionNotFoundError,
    CustomException,
    KnowledgeBaseEmptyError,
)
from src.graph import (
    fallback_node,
    finalize_node,
    generate_node,
    load_context_node,
    rag_graph,
    refine_node,
    retrieve_qa_node,
    retrieve_summary_node,
)
from src.logger import logging
from src.utils.rate_limiter import (
    LLMRateLimitError,
    llm_rate_limiter,
    raise_as_rate_limit_error,
)


class QAPipeline:
    """Runs every query through the LangGraph RAG orchestration workflow."""

    def __init__(
        self,
        collection_names: List[str] | None = None,
        session_id: str = "default",
        user_id: str | None = None,
    ):
        try:
            if not user_id:
                raise CustomException(ValueError("QAPipeline requires a user_id"), sys)
            self.collection_names = [name for name in (collection_names or []) if name]
            self.session_id = session_id
            self.user_id = user_id
        except Exception as e:
            raise CustomException(e, sys)

    def run(
        self,
        query: str,
        message_attachments: Optional[List[dict]] = None,
    ) -> dict:
        return asyncio.run(self.arun(query, message_attachments))

    async def arun(
        self,
        query: str,
        message_attachments: Optional[List[dict]] = None,
    ) -> dict:
        """Asynchronously invokes the LangGraph state graph."""
        try:
            logging.info("Processing query via LangGraph async: %s...", query[:50])

            initial_state = {
                "question": query,
                "collection_names": self.collection_names,
                "session_id": self.session_id,
                "user_id": self.user_id,
                "message_attachments": message_attachments,
            }

            query_type = "summary" if is_summary_request(query) else "qa"
            config = {
                "run_name": f"rag_{query_type}_{self.session_id[:8]}",
                "tags": [
                    query_type,
                    f"user_{self.user_id}",
                ],
                "metadata": {
                    "user_id": self.user_id,
                    "session_id": self.session_id,
                    "query_type": query_type,
                    "collection_scope": self.collection_names or [],
                },
                "configurable": {
                    "thread_id": self.session_id,
                },
            }

            result_state = await rag_graph.ainvoke(initial_state, config=config)

            return {
                "answer": result_state.get("final_answer", FALLBACK_ANSWER),
                "collection_scope": self.collection_names or "all",
                "query": query,
                "session_id": self.session_id,
            }
        except (CollectionNotFoundError, KnowledgeBaseEmptyError, LLMRateLimitError):
            raise
        except Exception as e:
            raise CustomException(e, sys)

    async def astream(
        self,
        query: str,
        message_attachments: Optional[List[dict]] = None,
    ) -> AsyncGenerator[str, None]:
        """Asynchronously streams token-by-token SSE events to the frontend."""
        thread_id = f"{self.user_id}:{self.session_id}:{uuid.uuid4()}"
        initial_state = {
            "question": query,
            "collection_names": self.collection_names,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "message_attachments": message_attachments,
        }

        try:
            logging.info("Starting SSE stream for query: %s...", query[:50])

            memory = MemoryManager(session_id=self.session_id, user_id=self.user_id)
            chat_history = await memory.aget_history()
            await memory.asave_message("human", query, attachments=message_attachments)

            is_summary = is_summary_request(query)
            docs = []
            refined_context = ""

            if is_summary:
                sum_res = await retrieve_summary_node(initial_state)
                docs = sum_res.get("docs", [])
                refined_context = sum_res.get("refined_context", "")
            else:
                qa_res = await retrieve_qa_node(initial_state)
                docs = qa_res.get("docs", [])
                refined_context = qa_res.get("refined_context", "")

            if not docs and not refined_context.strip():
                yield f"data: {json.dumps({'type': 'token', 'content': FALLBACK_ANSWER})}\n\n"
                await memory.asave_message("ai", FALLBACK_ANSWER)
                yield f"data: {json.dumps({'type': 'done', 'session_id': self.session_id, 'collection_scope': self.collection_names or 'all', 'final_answer': FALLBACK_ANSWER})}\n\n"
                return

            llm = _build_llm()
            prompt_value = QA_PROMPT.format_prompt(
                sources=refined_context,
                question=query,
                chat_history=chat_history,
            )

            await llm_rate_limiter.aacquire()

            accumulated_tokens: List[str] = []

            llm_config = {
                "run_name": f"rag_{'summary' if is_summary else 'qa'}_{self.session_id[:8]}",
                "tags": [
                    "summary" if is_summary else "qa",
                    f"user_{self.user_id}",
                ],
                "metadata": {
                    "user_id": self.user_id,
                    "session_id": self.session_id,
                    "query_type": "summary" if is_summary else "qa",
                    "collection_scope": self.collection_names or [],
                },
                "configurable": {
                    "thread_id": self.session_id,
                },
            }

            try:
                async for chunk in llm.astream(prompt_value.to_messages(), config=llm_config):
                    content = chunk.content if isinstance(chunk, AIMessageChunk) else str(chunk)
                    if content:
                        accumulated_tokens.append(content)
                        yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
            except Exception as e:
                raise_as_rate_limit_error(e)

            raw_answer = "".join(accumulated_tokens)
            final_answer = sanitize_answer(raw_answer)

            if is_summary:
                citations = build_source_only_citations(docs)
            else:
                citations = build_citations(docs)

            if citations:
                yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"

            persisted_answer = final_answer
            if persisted_answer != FALLBACK_ANSWER and citations:
                persisted_answer = f"{persisted_answer}\n\n{citations}"

            await memory.asave_message("ai", persisted_answer)

            yield f"data: {json.dumps({'type': 'done', 'session_id': self.session_id, 'collection_scope': self.collection_names or 'all', 'final_answer': persisted_answer})}\n\n"

        except LLMRateLimitError as exc:
            logging.warning("LLM rate limit in stream | kind: %s", exc.kind)
            err_data = {
                "type": "error",
                "error_code": f"llm_rate_limit_{exc.kind}",
                "message": exc.message,
            }
            yield f"data: {json.dumps(err_data)}\n\n"
        except CollectionNotFoundError as exc:
            logging.exception("Collection not found in stream")
            missing = getattr(exc, "missing_collections", [])
            err_data = {
                "type": "error",
                "error_code": "collection_not_found",
                "message": "This document was removed. Please upload a document to continue.",
                "missing_collections": missing,
            }
            yield f"data: {json.dumps(err_data)}\n\n"
        except KnowledgeBaseEmptyError:
            logging.exception("Knowledge base empty in stream")
            err_data = {
                "type": "error",
                "error_code": "knowledge_base_empty",
                "message": "Please upload a document or add a YouTube video first.",
            }
            yield f"data: {json.dumps(err_data)}\n\n"
        except Exception as exc:
            logging.exception("Unexpected error in stream")
            err_data = {
                "type": "error",
                "error_code": "server_error",
                "message": "Server encountered an error. Please try again.",
            }
            yield f"data: {json.dumps(err_data)}\n\n"
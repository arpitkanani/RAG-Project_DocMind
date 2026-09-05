import sys
from typing import Any, Dict

from langchain_core.output_parsers import StrOutputParser

from src.chains.qa_chain import (
    FALLBACK_ANSWER,
    QA_PROMPT,
    _build_llm,
    build_citations,
    build_source_only_citations,
    sanitize_answer,
)
from src.components.memory_manager import MemoryManager
from src.exception import CustomException
from src.graph.state import RAGState
from src.logger import logging
from src.utils.rate_limiter import (
    LLMRateLimitError,
    llm_rate_limiter,
    raise_as_rate_limit_error,
)


def fallback_node(state: RAGState) -> Dict[str, Any]:
    return {
        "raw_answer": FALLBACK_ANSWER,
        "final_answer": FALLBACK_ANSWER,
        "citations": "",
    }


async def generate_node(state: RAGState) -> Dict[str, Any]:
    """Generates the grounded answer using the refined context and chat history."""
    try:
        question = state["question"]
        chat_history = state.get("chat_history", [])
        refined_context = state.get("refined_context") or ""

        if not refined_context.strip():
            return {"raw_answer": FALLBACK_ANSWER}

        llm = _build_llm()
        parser = StrOutputParser()
        chain = (QA_PROMPT | llm | parser).with_config(run_name="qa_generation")

        await llm_rate_limiter.aacquire()

        try:
            raw_answer = await chain.ainvoke(
                {
                    "sources": refined_context,
                    "question": question,
                    "chat_history": chat_history,
                }
            )
        except Exception as e:
            raise_as_rate_limit_error(e)

        return {"raw_answer": raw_answer}
    except LLMRateLimitError:
        raise
    except Exception as e:
        raise CustomException(e, sys)


async def finalize_node(state: RAGState) -> Dict[str, Any]:
    """Sanitizes raw generation, builds citations, and asynchronously saves to PostgreSQL."""
    try:
        raw_answer = state.get("raw_answer", "")
        docs = state.get("docs", [])
        is_summary = state.get("is_summary", False)
        session_id = state.get("session_id", "default")
        user_id = state["user_id"]

        if raw_answer == FALLBACK_ANSWER or not raw_answer:
            final_answer = FALLBACK_ANSWER
            citations = ""
        else:
            final_answer = sanitize_answer(raw_answer)

            if is_summary:
                citations = build_source_only_citations(docs)
            else:
                citations = build_citations(docs)

            if final_answer != FALLBACK_ANSWER and citations:
                final_answer = f"{final_answer}\n\n{citations}"

        memory = MemoryManager(session_id=session_id, user_id=user_id)
        await memory.asave_message("ai", final_answer)

        logging.info("Answer finalized and persisted for session %s", session_id)
        return {
            "final_answer": final_answer,
            "citations": citations,
        }
    except Exception as e:
        raise CustomException(e, sys)

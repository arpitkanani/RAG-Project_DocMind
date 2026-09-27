import sys
from typing import Any, Dict

from langchain_core.output_parsers import StrOutputParser #type: ignore
from langsmith import traceable

from src.chains.qa_chain import (
    FALLBACK_ANSWER,
    QA_PROMPT,
    _build_llm,
    build_citations,
    build_source_only_citations,
    format_docs,
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


@traceable(name="RAG_Generator")
async def generate_node(state: RAGState) -> Dict[str, Any]:
    """Generates the grounded answer from retrieved docs and chat history."""
    try:
        question = state["question"]
        chat_history = state.get("chat_history", [])
        docs = state.get("docs", [])

        # Format docs into the source block the prompt expects.
        # retrieve_qa_node and retrieve_summary_node have already merged
        # same-location chunks, so we just call format_docs here.
        sources = format_docs(docs)

        if not sources.strip() or sources == "No grounded source passages are available.":
            return {"raw_answer": FALLBACK_ANSWER}

        # Prevent chat history poisoning / refusal cascades / memory summary leaking:
        # 1. Do NOT pass SystemMessages into QA_PROMPT's chat_history (they confuse the LLM into meta-narration).
        # 2. Strip previous fallbacks, DATA_NOT_FOUND tokens, and conversation summaries.
        clean_chat_history = []
        for msg in chat_history:
            if getattr(msg, "type", "") == "system":
                continue
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                content_lower = content.lower()
                if (
                    FALLBACK_ANSWER in content
                    or "DATA_NOT_FOUND" in content
                    or "couldn't find information" in content_lower
                    or "could not find relevant information" in content_lower
                    or "summary of earlier parts of this conversation" in content_lower
                    or "no further action or decision was made" in content_lower
                ):
                    continue
            clean_chat_history.append(msg)

        llm = _build_llm()
        parser = StrOutputParser()
        chain = (QA_PROMPT | llm | parser).with_config(run_name="RAG_Generation_Chain")

        await llm_rate_limiter.aacquire()

        try:
            from src.utils.helpers import extract_text

            raw_answer = await chain.ainvoke(
                {
                    "sources": sources,
                    "question": question,
                    "chat_history": clean_chat_history,
                }
            )
            raw_answer = extract_text(raw_answer)
            logging.info("generate_node: raw_answer received (len=%d): %r", len(raw_answer), raw_answer[:120])
        except Exception as e:
            raise_as_rate_limit_error(e)

        return {"raw_answer": raw_answer}
    except LLMRateLimitError:
        raise
    except Exception as e:
        raise CustomException(e, sys)


@traceable(name="Finalize_Response")
async def finalize_node(state: RAGState) -> Dict[str, Any]:
    """Sanitizes raw generation, builds citations, and asynchronously saves to PostgreSQL."""
    try:
        import json
        from src.utils.helpers import extract_text

        raw_answer = extract_text(state.get("raw_answer", ""))
        docs = state.get("docs", [])
        is_summary = state.get("is_summary", False)
        session_id = state.get("session_id", "default")
        user_id = state["user_id"]
        intent = state.get("intent", "")

        # Guard: Chitchat, conversational, and clarification responses must NEVER be sanitized with document refusal rules
        if intent in ("chitchat", "conversational", "clarify"):
            final_answer = raw_answer or extract_text(state.get("generation", ""))
            if not final_answer:
                final_answer = "I'm here to help! Feel free to ask a question or upload a document."
            citations = ""
            db_save_text = final_answer
            if intent == "clarify":
                try:
                    c_data = json.loads(final_answer)
                    if isinstance(c_data, dict) and c_data.get("type") == "clarification":
                        opts = "\n- ".join(c_data.get("options", []))
                        db_save_text = f"{c_data.get('message', 'Please clarify your question:')}\n- {opts}"
                except Exception:
                    pass
            memory = MemoryManager(session_id=session_id, user_id=user_id)
            await memory.asave_message("ai", db_save_text)
            logging.info("Chitchat/conversational answer finalized for session %s", session_id)
            return {
                "final_answer": final_answer,
                "citations": citations,
            }

        if raw_answer == FALLBACK_ANSWER or not raw_answer:
            final_answer = FALLBACK_ANSWER
            citations = ""
        else:
            final_answer = sanitize_answer(raw_answer)
            citations = ""

            if is_summary:
                citations = build_source_only_citations(docs)
            else:
                citations = build_citations(docs, final_answer)

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

"""Consolidated graph nodes for the DocuVortex agentic RAG workflow.

Nodes:
    classify_intent   – Detect conversational vs retrieval intent
    chitchat          – Lightweight conversational response (Gemini 3.6 Flash)
    grade_documents   – Evaluate retrieved chunk relevance (Gemini 3.6 Flash)
    fallback_response – Polite message when docs are irrelevant
"""

import sys
from typing import Any, Dict

from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
from langsmith import traceable  # type: ignore

from src.chains.qa_chain import FALLBACK_ANSWER
from src.exception import CustomException
from src.graph.state import RAGState
from src.logger import logging


# ─── Gemini 3.6 Flash — shared lightweight model for classification/grading ───
_gemini_flash = None


def _get_gemini_flash() -> ChatGoogleGenerativeAI:
    """Lazy-initialize and cache the Gemini 3.6 Flash model."""
    global _gemini_flash
    if _gemini_flash is None:
        _gemini_flash = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            temperature=0,
            max_output_tokens=50,
        )
    return _gemini_flash


FALLBACK_MESSAGE = (
    "I couldn't find relevant information in your uploaded documents to answer "
    "this question. Please try rephrasing your question or upload additional "
    "documents that may contain the answer."
)


# ══════════════════════════════════════════════════════════════════════════════
# Node: classify_intent
# ══════════════════════════════════════════════════════════════════════════════

@traceable(name="classify_intent")
async def classify_intent(state: RAGState) -> Dict[str, Any]:
    """Detect whether to perform document retrieval or general conversational AI.

    Logic:
        - If a document source is selected/uploaded (source_selected or collection_names),
          route to 'retrieve' / 'retrieval'.
        - If no document is selected/uploaded, route to 'chitchat' (conversational AI).
    """
    has_source = bool(state.get("source_selected") or state.get("collection_names"))

    if has_source:
        logging.info("→ classify_intent: source present, routing to retrieval")
        return {"intent": "retrieval"}

    logging.info("→ classify_intent: no source selected, routing to chitchat")
    return {"intent": "chitchat"}


# ══════════════════════════════════════════════════════════════════════════════
# Node: chitchat
# ══════════════════════════════════════════════════════════════════════════════

@traceable(name="chitchat")
async def chitchat(state: RAGState) -> Dict[str, Any]:
    """Respond as a general-purpose AI using parametric knowledge and conversation history."""
    question = state["question"]
    chat_history = state.get("chat_history", [])

    try:
        from src.chains.qa_chain import _build_llm
        from src.utils.rate_limiter import llm_rate_limiter
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.output_parsers import StrOutputParser

        llm = _build_llm()

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are DocuVortex, an intelligent, helpful, and friendly AI assistant. "
                "No specific document has been selected for this query. "
                "Answer the user's question directly, accurately, and helpfully using your general knowledge. "
                "Engage conversationally, clearly, and concisely without making references to missing document context."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        chain = prompt | llm | StrOutputParser()

        await llm_rate_limiter.aacquire()
        answer = await chain.ainvoke({
            "chat_history": chat_history,
            "question": question,
        })

        answer = (answer or "").strip()
        logging.info("→ chitchat: responded with %d chars", len(answer))
        return {
            "raw_answer": answer,
            "final_answer": answer,
            "citations": "",
        }
    except Exception as e:
        logging.warning("chitchat failed (%s), returning generic greeting", e)
        default_msg = "Hello! I'm DocuVortex, your AI assistant. How can I help you today?"
        return {
            "raw_answer": default_msg,
            "final_answer": default_msg,
            "citations": "",
        }


# ══════════════════════════════════════════════════════════════════════════════
# Node: grade_documents
# ══════════════════════════════════════════════════════════════════════════════

@traceable(name="grade_documents")
async def grade_documents(state: RAGState) -> Dict[str, Any]:
    """Evaluate whether retrieved chunks contain relevant information.

    Strategy to avoid Gemini 429 errors and context size issues:
        - Pass condensed summaries (first 200 chars per chunk + keywords)
        - Cap at 5 chunks maximum
        - max_output_tokens=10 (just 'yes' or 'no')
        - On ANY failure, gracefully degrade: skip grading, assume relevant
    """
    docs = state.get("docs", [])
    question = state["question"]

    if not docs:
        logging.info("→ grade_documents: no docs retrieved, marking irrelevant")
        return {"grade": "irrelevant"}

    try:
        # Condense: first 200 chars per chunk, max 5 chunks
        condensed = "\n".join(
            f"[Chunk {i+1}]: {doc.page_content[:200]}..."
            for i, doc in enumerate(docs[:5])
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            temperature=0,
            max_output_tokens=10,
        )

        result = await llm.ainvoke(
            "You are a relevance grader. Given a user question and document context, "
            "determine if the context contains information relevant to answering the question.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{condensed}\n\n"
            "Reply with ONLY one word: 'yes' if relevant, 'no' if irrelevant."
        )

        answer = result.content.strip().lower()
        grade = "relevant" if "yes" in answer else "irrelevant"
        logging.info("→ grade_documents: %s (raw=%r, %d chunks evaluated)", grade, answer, min(len(docs), 5))
        return {"grade": grade}

    except Exception as e:
        # Graceful degradation: skip grading on any error (429, context too large, etc.)
        logging.warning(
            "→ grade_documents: grading failed (%s), skipping — routing to generate", e
        )
        return {"grade": "relevant"}


# ══════════════════════════════════════════════════════════════════════════════
# Node: fallback_response
# ══════════════════════════════════════════════════════════════════════════════

@traceable(name="fallback_response")
async def fallback_response(state: RAGState) -> Dict[str, Any]:
    """Return a polite message when documents don't contain relevant information."""
    logging.info("→ fallback_response: no relevant docs found")
    return {
        "raw_answer": FALLBACK_ANSWER,
        "final_answer": FALLBACK_ANSWER,
        "citations": "",
    }

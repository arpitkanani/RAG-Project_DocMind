import asyncio
import os
import re
import sys
from typing import Any, AsyncGenerator, Dict, List, Optional, TypedDict
import uuid

import yaml
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END

from src.chains.qa_chain import (
    FALLBACK_ANSWER,
    NOT_FOUND_TOKEN,
    QA_PROMPT,
    _build_llm,
    _format_locator,
    _format_section,
    _format_source_label,
    build_citations,
    build_source_only_citations,
    format_docs,
    is_summary_request,
    merge_same_location_docs,
    sanitize_answer,
)
from src.components.memory_manager import MemoryManager
from src.components.retriever import Retriever
from src.exception import (
    CollectionNotFoundError,
    CustomException,
    KnowledgeBaseEmptyError,
)
from src.logger import logging
from src.utils.rate_limiter import (
    LLMRateLimitError,
    llm_rate_limiter,
    raise_as_rate_limit_error,
)

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)


# =====================================================================
#  STATE DEFINITION
# =====================================================================

class RAGState(TypedDict, total=False):
    # Request inputs
    question: str
    collection_names: List[str]
    session_id: str
    user_id: str
    message_attachments: Optional[List[dict]]

    # Internal routing & context
    is_summary: bool
    chat_history: List[dict]
    docs: List[Document]

    # CRAG Refine Fields
    strips: List[str]
    kept_strips: List[str]
    refined_context: str

    # Output & Persistence
    raw_answer: str
    final_answer: str
    citations: str

    # Error tracking
    error_code: Optional[str]
    error_message: Optional[str]


# =====================================================================
#  REFINE (DECOMPOSE -> FILTER JUDGE -> RECOMPOSE)
# =====================================================================

class KeepOrDrop(BaseModel):
    """Schema for strict sentence relevance judging."""
    keep: bool = Field(description="Return true only if the sentence directly helps answer the question")


def decompose_to_sentences(text: str) -> List[str]:
    """Split text into individual sentences and filter out short fragments."""
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


filter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict relevance filter.\n"
            "Return keep=true only if the sentence directly helps answer the question.\n"
            "Use ONLY the sentence. Output JSON only.",
        ),
        ("human", "Question: {question}\n\nSentence:\n{sentence}"),
    ]
)


def _build_filter_chain():
    """Build structured output filter chain using configured LLM."""
    llm = _build_llm()
    return filter_prompt | llm.with_structured_output(KeepOrDrop)


# =====================================================================
#  GRAPH NODES
# =====================================================================

def load_context_node(state: RAGState) -> Dict[str, Any]:
    """Loads Postgres chat history and persists the human query."""
    try:
        session_id = state.get("session_id", "default")
        user_id = state["user_id"]
        question = state["question"]
        attachments = state.get("message_attachments")

        memory = MemoryManager(session_id=session_id, user_id=user_id)
        chat_history = memory.get_history()

        # Persist human message
        memory.save_message("human", question, attachments=attachments)

        is_summary = is_summary_request(question)
        return {
            "chat_history": chat_history,
            "is_summary": is_summary,
        }
    except Exception as e:
        raise CustomException(e, sys)


def retrieve_qa_node(state: RAGState) -> Dict[str, Any]:
    """Executes QA vector search and lexical reranking."""
    try:
        question = state["question"]
        collection_names = state.get("collection_names")

        retriever = Retriever(collection_names=collection_names)
        ranked_docs = retriever.retrieve_ranked(question)
        docs = [doc for doc, _, _ in ranked_docs]
        docs = merge_same_location_docs(docs)

        # Logging for evaluation
        logging.info("context: question=%r | %d chunk(s) retrieved", question, len(docs))
        for i, doc in enumerate(docs, start=1):
            logging.info("context[%d]: %s", i, doc.page_content)

        return {"docs": docs}
    except (CollectionNotFoundError, KnowledgeBaseEmptyError):
        raise
    except Exception as e:
        raise CustomException(e, sys)


def retrieve_summary_node(state: RAGState) -> Dict[str, Any]:
    """Scrolls and retrieves full document context for summary requests."""
    try:
        collection_names = state.get("collection_names")
        retriever = Retriever(collection_names=collection_names)
        docs = retriever.get_full_context(
            max_chars=config["retriever"].get("summary_max_chars", 6000)
        )
        docs = merge_same_location_docs(docs)

        logging.info("summary context: %d chunk(s) retrieved", len(docs))
        return {
            "docs": docs,
            "refined_context": format_docs(docs),
        }
    except (CollectionNotFoundError, KnowledgeBaseEmptyError):
        raise
    except Exception as e:
        raise CustomException(e, sys)


async def refine_node(state: RAGState) -> Dict[str, Any]:
    """
    CRAG Refine step:
    1. Combine retrieved docs into context
    2. Decompose into sentence strips
    3. Filter: LLM judge evaluates each sentence (in bounded parallel batches)
    4. Recompose: Glue kept strips back together into refined_context
    """
    q = state["question"]
    docs = state.get("docs", [])
    if not docs:
        return {
            "strips": [],
            "kept_strips": [],
            "refined_context": "",
        }

    # 1) Combine retrieved docs into one context string
    context = "\n\n".join(d.page_content for d in docs).strip()

    # 1) DECOMPOSITION: context -> sentence strips
    strips = decompose_to_sentences(context)
    if not strips:
        return {
            "strips": [],
            "kept_strips": [],
            "refined_context": context,
        }

    # 2) FILTER: keep only relevant strips via judge LLM
    filter_chain = _build_filter_chain()
    kept: List[str] = []

    try:
        # Evaluate strips with proactive rate limiting
        # Use abatch for fast concurrent evaluation
        inputs = [{"question": q, "sentence": s} for s in strips]
        
        # Batch size guard to avoid hitting provider burst limits
        batch_size = 10
        for i in range(0, len(inputs), batch_size):
            llm_rate_limiter.acquire()
            batch_inputs = inputs[i : i + batch_size]
            results = await filter_chain.abatch(batch_inputs)
            for s, res in zip(strips[i : i + batch_size], results):
                if res and getattr(res, "keep", False):
                    kept.append(s)
    except Exception as e:
        logging.warning("Refine filtering encountered error, falling back to full strips: %s", str(e))
        kept = strips

    # 3) RECOMPOSE: glue kept strips back together (internal knowledge)
    refined_context = "\n".join(kept).strip()

    return {
        "strips": strips,
        "kept_strips": kept,
        "refined_context": refined_context,
    }


def fallback_node(state: RAGState) -> Dict[str, Any]:
    """Produces the standard fallback answer when no context was found."""
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

        # QA prompt expects {sources, question, chat_history}
        chain = QA_PROMPT | llm | parser

        llm_rate_limiter.acquire()

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


def finalize_node(state: RAGState) -> Dict[str, Any]:
    """Sanitizes raw generation, builds clean citations, and saves to PostgreSQL."""
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

        # Persist AI message to Postgres
        memory = MemoryManager(session_id=session_id, user_id=user_id)
        memory.save_message("ai", final_answer)

        logging.info("Answer finalized and persisted for session %s", session_id)
        return {
            "final_answer": final_answer,
            "citations": citations,
        }
    except Exception as e:
        raise CustomException(e, sys)


# =====================================================================
#  ROUTERS & CONDITIONAL EDGES
# =====================================================================

def route_query_mode(state: RAGState) -> str:
    """Route between QA retrieval mode and Summary retrieval mode."""
    return "retrieve_summary" if state.get("is_summary", False) else "retrieve_qa"


def check_context_exists(state: RAGState) -> str:
    """Check if any valid context was retrieved / kept."""
    docs = state.get("docs", [])
    refined_context = state.get("refined_context", "")
    if not docs and not refined_context.strip():
        return "fallback"
    return "generate"


# =====================================================================
#  GRAPH BUILDER
# =====================================================================

def build_rag_graph():
    """Compiles the stateless LangGraph StateGraph workflow."""
    builder = StateGraph(RAGState)

    # Register Nodes
    builder.add_node("load_context", load_context_node)
    builder.add_node("retrieve_qa", retrieve_qa_node)
    builder.add_node("retrieve_summary", retrieve_summary_node)
    builder.add_node("refine", refine_node)
    builder.add_node("fallback", fallback_node)
    builder.add_node("generate", generate_node)
    builder.add_node("finalize", finalize_node)

    # Register Edges
    builder.add_edge(START, "load_context")
    builder.add_conditional_edges(
        "load_context",
        route_query_mode,
        {
            "retrieve_qa": "retrieve_qa",
            "retrieve_summary": "retrieve_summary",
        },
    )

    builder.add_edge("retrieve_qa", "refine")

    builder.add_conditional_edges(
        "refine",
        check_context_exists,
        {
            "fallback": "fallback",
            "generate": "generate",
        },
    )

    builder.add_conditional_edges(
        "retrieve_summary",
        check_context_exists,
        {
            "fallback": "fallback",
            "generate": "generate",
        },
    )

    builder.add_edge("fallback", "finalize")
    builder.add_edge("generate", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


# Process-wide compiled graph singleton (graph structure is stateless; state is passed per invocation)
rag_graph = build_rag_graph()

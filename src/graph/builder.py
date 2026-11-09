"""DocuVortex LangGraph StateGraph — Agentic RAG workflow with intent
classification, grounded retrieval, and synthesis.

Graph shape:
    START → load_context → classify_intent
       ├─(conversational)→ chitchat → finalize → END
       └─(retrieval)→ route_query_mode
            ├─(qa)→ retrieve_qa ─┬─(empty)→ fallback_response → finalize → END
            │                    └─(docs exist)→ generate → finalize → END
            └─(summary)→ retrieve_summary ─┬─(empty)→ fallback_response → finalize → END
                                           └─(docs exist)→ generate → finalize → END
"""

from langgraph.graph import END, START, StateGraph

from src.graph.nodes_agentic import (
    chitchat,
    clarify_question,
    classify_intent,
    fallback_response,
    grade_documents,
)
from src.graph.nodes.generate import finalize_node, generate_node
from src.graph.nodes.retrieve import (
    load_context_node,
    retrieve_qa_node,
    retrieve_summary_node,
)
from src.graph.state import RAGState


# ─── Conditional edge functions ───────────────────────────────────────────────

def route_after_intent(state: RAGState) -> str:
    """Route based on classify_intent result, query clarity, and summary detection."""
    intent = state.get("intent", "retrieval")
    if intent in ("chitchat", "conversational"):
        return "chitchat"
    if intent == "clarify":
        return "clarify_question"
    # Retrieval intent — check if summary or QA
    if state.get("is_summary", False):
        return "retrieve_summary"
    return "retrieve_qa"


def check_docs_exist_qa(state: RAGState) -> str:
    """Route QA retrieval directly to generate when docs exist, or fallback when empty."""
    docs = state.get("docs", [])
    if not docs:
        return "fallback_response"
    return "generate"


def check_docs_exist_summary(state: RAGState) -> str:
    """Summary retrieval fetches full context; route directly to generate if docs exist."""
    docs = state.get("docs", [])
    if not docs:
        return "fallback_response"
    return "generate"


def route_grade(state: RAGState) -> str:
    """Route based on document grading result."""
    grade = state.get("grade", "relevant")
    if grade == "irrelevant":
        return "fallback_response"
    return "generate"


# ─── Graph builder ────────────────────────────────────────────────────────────

def build_rag_graph(checkpointer=None):
    """Compiles the asynchronous LangGraph StateGraph workflow."""
    builder = StateGraph(RAGState)

    # Register all nodes
    builder.add_node("load_context", load_context_node)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("clarify_question", clarify_question)
    builder.add_node("chitchat", chitchat)
    builder.add_node("retrieve_qa", retrieve_qa_node)
    builder.add_node("retrieve_summary", retrieve_summary_node)
    builder.add_node("grade_documents", grade_documents)
    builder.add_node("generate", generate_node)
    builder.add_node("fallback_response", fallback_response)
    builder.add_node("finalize", finalize_node)

    # ── Edges ──

    # START → load_context → classify_intent
    builder.add_edge(START, "load_context")
    builder.add_edge("load_context", "classify_intent")

    # Intent routing: conversational → chitchat, clarify → clarify_question, retrieval → retrieve_qa/summary
    builder.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "chitchat": "chitchat",
            "clarify_question": "clarify_question",
            "retrieve_qa": "retrieve_qa",
            "retrieve_summary": "retrieve_summary",
        },
    )

    # Clarification feedback → finalize → END
    builder.add_edge("clarify_question", "finalize")

    # Chitchat → finalize → END
    builder.add_edge("chitchat", "finalize")

    # After retrieval:
    # QA routes directly to generate (grade_documents permanently bypassed for speed & reliability)
    builder.add_conditional_edges(
        "retrieve_qa",
        check_docs_exist_qa,
        {
            "generate": "generate",
            "fallback_response": "fallback_response",
        },
    )

    # Summary routes directly to generate
    builder.add_conditional_edges(
        "retrieve_summary",
        check_docs_exist_summary,
        {
            "generate": "generate",
            "fallback_response": "fallback_response",
        },
    )

    # Terminal edges → finalize → END
    builder.add_edge("generate", "finalize")
    builder.add_edge("fallback_response", "finalize")
    builder.add_edge("finalize", END)

    if checkpointer:
        return builder.compile(checkpointer=checkpointer)
    return builder.compile()


rag_graph = build_rag_graph()

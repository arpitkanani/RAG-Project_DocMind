"""DocuVortex LangGraph StateGraph — Agentic RAG workflow with intent
classification, document grading, and grounded generation.

Graph shape:
    START → load_context → classify_intent
       ├─(conversational)→ chitchat → finalize → END
       └─(retrieval)→ route_query_mode
            ├─(qa)→ retrieve_qa ─┐
            └─(summary)→ retrieve_summary ─┤
                                           ├─[check_docs]→ grade_documents
                                           │                  ├─(relevant)→ generate → finalize → END
                                           │                  └─(irrelevant)→ fallback_response → finalize → END
                                           └─(empty)→ fallback_response → finalize → END
"""

from langgraph.graph import END, START, StateGraph

from src.graph.nodes_agentic import (
    chitchat,
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
    """Route based on classify_intent result and summary detection."""
    intent = state.get("intent", "retrieval")
    if intent in ("chitchat", "conversational"):
        return "chitchat"
    # Retrieval intent — check if summary or QA
    if state.get("is_summary", False):
        return "retrieve_summary"
    return "retrieve_qa"


def check_docs_exist(state: RAGState) -> str:
    """Route to grade_documents or fallback when no docs retrieved."""
    docs = state.get("docs", [])
    if not docs:
        return "fallback_response"
    return "grade_documents"


def route_grade(state: RAGState) -> str:
    """Route based on document grading result."""
    grade = state.get("grade", "relevant")
    if grade == "irrelevant":
        return "fallback_response"
    return "generate"


# ─── Graph builder ────────────────────────────────────────────────────────────

def build_rag_graph():
    """Compiles the asynchronous LangGraph StateGraph workflow."""
    builder = StateGraph(RAGState)

    # Register all nodes
    builder.add_node("load_context", load_context_node)
    builder.add_node("classify_intent", classify_intent)
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

    # Intent routing: conversational → chitchat, retrieval → retrieve_qa/summary
    builder.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "chitchat": "chitchat",
            "retrieve_qa": "retrieve_qa",
            "retrieve_summary": "retrieve_summary",
        },
    )

    # Chitchat → finalize → END
    builder.add_edge("chitchat", "finalize")

    # After retrieval, check if docs exist before grading
    builder.add_conditional_edges(
        "retrieve_qa",
        check_docs_exist,
        {
            "grade_documents": "grade_documents",
            "fallback_response": "fallback_response",
        },
    )

    builder.add_conditional_edges(
        "retrieve_summary",
        check_docs_exist,
        {
            "grade_documents": "grade_documents",
            "fallback_response": "fallback_response",
        },
    )

    # Grade routing: relevant → generate, irrelevant → fallback
    builder.add_conditional_edges(
        "grade_documents",
        route_grade,
        {
            "generate": "generate",
            "fallback_response": "fallback_response",
        },
    )

    # Terminal edges → finalize → END
    builder.add_edge("generate", "finalize")
    builder.add_edge("fallback_response", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


rag_graph = build_rag_graph()

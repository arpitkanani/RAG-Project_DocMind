from langgraph.graph import StateGraph, START, END

from src.graph.nodes.generate import fallback_node, finalize_node, generate_node
from src.graph.nodes.refine import refine_node
from src.graph.nodes.retrieve import (
    load_context_node,
    retrieve_qa_node,
    retrieve_summary_node,
)
from src.graph.state import RAGState


def route_query_mode(state: RAGState) -> str:
    return "retrieve_summary" if state.get("is_summary", False) else "retrieve_qa"


def check_context_exists(state: RAGState) -> str:
    docs = state.get("docs", [])
    refined_context = state.get("refined_context", "")
    if not docs and not refined_context.strip():
        return "fallback"
    return "generate"


def build_rag_graph():
    """Compiles the asynchronous LangGraph StateGraph workflow."""
    builder = StateGraph(RAGState)

    builder.add_node("load_context", load_context_node)
    builder.add_node("retrieve_qa", retrieve_qa_node)
    builder.add_node("retrieve_summary", retrieve_summary_node)
    builder.add_node("refine", refine_node)
    builder.add_node("fallback", fallback_node)
    builder.add_node("generate", generate_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "load_context")
    builder.add_conditional_edges(
        "load_context",
        route_query_mode,
        {
            "retrieve_qa": "retrieve_qa",
            "retrieve_summary": "retrieve_summary",
        },
    )

    builder.add_conditional_edges(
        "retrieve_qa",
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


rag_graph = build_rag_graph()

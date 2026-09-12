from langgraph.graph import END, START, StateGraph

from src.graph.nodes.generate import fallback_node, finalize_node, generate_node
from src.graph.nodes.retrieve import (
    load_context_node,
    retrieve_qa_node,
    retrieve_summary_node,
)
from src.graph.state import RAGState


def route_query_mode(state: RAGState) -> str:
    return "retrieve_summary" if state.get("is_summary", False) else "retrieve_qa"


def check_docs_exist(state: RAGState) -> str:
    """Route to fallback when no docs were retrieved."""
    docs = state.get("docs", [])
    if not docs:
        return "fallback"
    return "generate"


def build_rag_graph():
    """Compiles the asynchronous LangGraph StateGraph workflow.

    Graph shape:
        START → load_context
                    ├─(summary)→ retrieve_summary ─┐
                    └─(qa)────→ retrieve_qa ────────┤
                                                    ├─(docs)→ generate → finalize → END
                                                    └─(empty)→ fallback → finalize → END
    """
    builder = StateGraph(RAGState)

    builder.add_node("load_context", load_context_node)
    builder.add_node("retrieve_qa", retrieve_qa_node)
    builder.add_node("retrieve_summary", retrieve_summary_node)
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
        check_docs_exist,
        {
            "fallback": "fallback",
            "generate": "generate",
        },
    )

    builder.add_conditional_edges(
        "retrieve_summary",
        check_docs_exist,
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

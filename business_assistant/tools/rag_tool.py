import time
from typing import Optional
from langchain_core.tools import tool

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

from src.components.retriever import Retriever
from src.exception import CollectionNotFoundError, KnowledgeBaseEmptyError
from src.logger import logging, log_latency


def get_brochure_rag_tool(bot_id: str):
    """
    Creates an LLM-callable RAG tool bound to the specific deployed bot's
    brochures vector collection (biz_{bot_id}).
    Reuses existing Retriever from src.components.retriever without modifying core RAG code.
    Instruments vector search with @traceable and logs retrieval latency to logs/latency.log.
    """
    collection_name = f"biz_{bot_id}"

    @tool
    @traceable(name="BrochureRAG.search", run_type="retriever")
    async def search_business_brochures(query: str) -> str:
        """Search the business brochures, product manuals, and catalogs for verified facts, pricing, usage instructions, and benefits.

        Args:
            query: The specific question or topic to search for in the brochures.
        """
        logging.info("[Brochure RAG Tool] Query: %r | Collection: %s", query, collection_name)
        t_rag_start = time.perf_counter()
        try:
            retriever = Retriever(collection_names=[collection_name])
            ranked_docs = await retriever.retrieve_ranked(query)
            rag_duration_ms = (time.perf_counter() - t_rag_start) * 1000.0

            log_latency(
                "BROCHURE_RAG",
                f"Bot: '{bot_id}' | Collection: '{collection_name}' | Query: {query[:80]!r} | "
                f"Retrieved: {len(ranked_docs) if ranked_docs else 0} chunks | Latency: {rag_duration_ms:.1f}ms",
            )

            if not ranked_docs:
                return "No matching information found in the uploaded brochures."

            excerpts = []
            for doc, score, col in ranked_docs[:5]:
                content = doc.page_content.strip()
                page = doc.metadata.get("page", "unknown")
                source = doc.metadata.get("source", "brochure.pdf")
                excerpts.append(f"[{source}, Page {page}]:\n{content}")

            return "\n\n---\n\n".join(excerpts)

        except (CollectionNotFoundError, KnowledgeBaseEmptyError) as e:
            rag_duration_ms = (time.perf_counter() - t_rag_start) * 1000.0
            log_latency("BROCHURE_RAG", f"Bot: '{bot_id}' | Collection not found/empty in {rag_duration_ms:.1f}ms: {e}")
            logging.warning("[Brochure RAG Tool] Collection not found or empty: %s", e)
            return "No brochures available for this business."
        except Exception as e:
            rag_duration_ms = (time.perf_counter() - t_rag_start) * 1000.0
            log_latency("BROCHURE_RAG", f"Bot: '{bot_id}' | Retrieval error in {rag_duration_ms:.1f}ms: {e}")
            err_str = str(e).lower()
            if any(k in err_str for k in ("embedding", "rate limit", "429", "unavailable", "quota")):
                logging.warning("[Brochure RAG Tool] Embedding lookup temporarily unavailable: %s", e)
                return "Note: Real-time brochure lookup is currently rate-limited; please answer based on the verified business blueprint and core policies."
            logging.exception("[Brochure RAG Tool] Unexpected retrieval error: %s", e)
            return "Note: Real-time brochure lookup is temporarily unavailable; please answer based on the verified business blueprint."

    return search_business_brochures

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
from src.utils.perf import capture_rag_perf, log_perf_kv, mask_pii

_RAG_RESULT_CACHE: dict[tuple[str, str], tuple[float, str]] = {}
RAG_CACHE_TTL_SEC = 120.0


def get_brochure_rag_tool(bot_id: str):
    """
    Creates an LLM-callable RAG tool bound to the specific deployed bot's
    brochures vector collection (biz_{bot_id}).
    Reuses existing Retriever from src.components.retriever without modifying core RAG code.
    Instruments vector search with @traceable, granular sub-step telemetry, and PII masking.
    """
    collection_name = f"biz_{bot_id}"

    @tool
    @traceable(name="BrochureRAG.search", run_type="retriever")
    async def search_business_brochures(query: str) -> str:
        """Search the business brochures, product manuals, and catalogs for verified facts, pricing, usage instructions, and benefits.

        Args:
            query: The specific question or topic to search for in the brochures.
        """
        masked_q = mask_pii(query)
        cache_key = (bot_id, " ".join(query.strip().lower().split()))
        cached_entry = _RAG_RESULT_CACHE.get(cache_key)
        if cached_entry and (time.time() - cached_entry[0]) < RAG_CACHE_TTL_SEC:
            logging.info("[Brochure RAG Tool] Serving cached context (<1ms) for %r", masked_q[:60])
            return cached_entry[1]

        logging.info("[Brochure RAG Tool] Query: %r | Collection: %s", masked_q, collection_name)
        with capture_rag_perf() as rag_perf:
            t_rag_start = time.perf_counter()
            try:
                retriever = Retriever(collection_names=[collection_name])
                ranked_docs = await retriever.retrieve_ranked(query)
                rag_duration_ms = (time.perf_counter() - t_rag_start) * 1000.0

                t_merge0 = time.perf_counter()
                if not ranked_docs:
                    res_text = "No matching information found in the uploaded brochures."
                else:
                    excerpts = []
                    for doc, score, col in ranked_docs[:5]:
                        content = doc.page_content.strip()
                        page = doc.metadata.get("page", "unknown")
                        source = doc.metadata.get("source", "brochure.pdf")
                        excerpts.append(f"[{source}, Page {page}]:\n{content}")
                    res_text = "\n\n---\n\n".join(excerpts)
                merge_ms = (time.perf_counter() - t_merge0) * 1000.0
                _RAG_RESULT_CACHE[cache_key] = (time.time(), res_text)

                log_latency(
                    "BROCHURE_RAG",
                    f"Bot: '{bot_id}' | Collection: '{collection_name}' | Query: {masked_q[:80]!r} | "
                    f"Retrieved: {len(ranked_docs) if ranked_docs else 0} chunks | Latency: {rag_duration_ms:.1f}ms",
                )

                log_perf_kv(
                    "rag",
                    bot=bot_id,
                    query_masked=masked_q,
                    exists_check_ms=rag_perf.get("collection_exists_check_ms", 0.0),
                    variants_build_ms=rag_perf.get("query_variants_build_ms", 0.0),
                    emb_calls=rag_perf.get("emb_calls", 0),
                    emb_ms_each=rag_perf.get("emb_ms", []),
                    emb_provider=rag_perf.get("emb_provider", "unknown"),
                    emb_model=rag_perf.get("emb_model", "unknown"),
                    emb_batched=rag_perf.get("emb_batched", 0),
                    qdrant_calls=rag_perf.get("qdrant_calls", 0),
                    qdrant_ms_each=rag_perf.get("qdrant_ms", []),
                    rerank_ms=rag_perf.get("rerank_ms", 0.0),
                    merge_and_format_ms=round(merge_ms, 1),
                    total_rag_ms=round(rag_duration_ms, 1),
                    cold=rag_perf.get("cold", 0),
                )

                return res_text

            except (CollectionNotFoundError, KnowledgeBaseEmptyError) as e:
                rag_duration_ms = (time.perf_counter() - t_rag_start) * 1000.0
                log_latency("BROCHURE_RAG", f"Bot: '{bot_id}' | Collection not found/empty in {rag_duration_ms:.1f}ms: {e}")
                log_perf_kv(
                    "rag",
                    bot=bot_id,
                    query_masked=masked_q,
                    exists_check_ms=rag_perf.get("collection_exists_check_ms", 0.0),
                    total_rag_ms=round(rag_duration_ms, 1),
                    error=str(e)[:60],
                    cold=rag_perf.get("cold", 0),
                )
                logging.warning("[Brochure RAG Tool] Collection not found or empty: %s", e)
                return "No brochures available for this business."
            except Exception as e:
                rag_duration_ms = (time.perf_counter() - t_rag_start) * 1000.0
                log_latency("BROCHURE_RAG", f"Bot: '{bot_id}' | Retrieval error in {rag_duration_ms:.1f}ms: {e}")
                log_perf_kv(
                    "rag",
                    bot=bot_id,
                    query_masked=masked_q,
                    exists_check_ms=rag_perf.get("collection_exists_check_ms", 0.0),
                    total_rag_ms=round(rag_duration_ms, 1),
                    error=str(e)[:60],
                    cold=rag_perf.get("cold", 0),
                )
                err_str = str(e).lower()
                if any(k in err_str for k in ("embedding", "rate limit", "429", "unavailable", "quota")):
                    logging.warning("[Brochure RAG Tool] Embedding lookup temporarily unavailable: %s", e)
                    return "Note: Real-time brochure lookup is currently rate-limited; please answer based on the verified business blueprint and core policies."
                logging.exception("[Brochure RAG Tool] Unexpected retrieval error: %s", e)
                return "Note: Real-time brochure lookup is temporarily unavailable; please answer based on the verified business blueprint."

    return search_business_brochures

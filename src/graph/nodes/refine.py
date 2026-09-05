import contextvars
from typing import Any, Dict, List

from src.graph.state import RAGState
from src.graph.utils import (
    _build_filter_chain,
    decompose_to_sentences,
    recompose_sentences,
)
from src.logger import logging
from src.utils.rate_limiter import llm_rate_limiter


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

    context = "\n\n".join(d.page_content for d in docs).strip()
    strips = decompose_to_sentences(context)
    if not strips:
        return {
            "strips": [],
            "kept_strips": [],
            "refined_context": context,
        }

    filter_chain = _build_filter_chain()
    kept: List[str] = []

    # Capture current context to propagate LangSmith trace parent down to judge tasks
    ctx = contextvars.copy_context()

    try:
        inputs = [{"question": q, "sentence": s} for s in strips]
        batch_size = 10
        for i in range(0, len(inputs), batch_size):
            await llm_rate_limiter.aacquire()
            batch_inputs = inputs[i : i + batch_size]
            results = await ctx.run(filter_chain.abatch, batch_inputs)
            for s, res in zip(strips[i : i + batch_size], results):
                if res and getattr(res, "keep", False):
                    kept.append(s)
    except Exception as e:
        logging.warning("Refine filtering encountered error, falling back to full strips: %s", str(e))
        kept = strips

    refined_context = recompose_sentences(kept)
    return {
        "strips": strips,
        "kept_strips": kept,
        "refined_context": refined_context,
    }

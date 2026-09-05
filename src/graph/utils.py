import re
from typing import List

from langchain_core.prompts import ChatPromptTemplate

from src.chains.qa_chain import _build_llm
from src.schemas import KeepOrDrop


# --- Sentence Decomposition & Recomposition Helpers ---

def decompose_to_sentences(text: str) -> List[str]:
    """Split text into individual sentences and filter out short fragments."""
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def recompose_sentences(kept_strips: List[str]) -> str:
    """Recompose kept sentence strips into a coherent refined context string."""
    return "\n".join(kept_strips).strip()


# --- Relevance Judge Prompt & Filter Chain ---

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
    return (filter_prompt | llm.with_structured_output(KeepOrDrop)).with_config(
        run_name="judge_filter"
    )

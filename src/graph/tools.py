import sys
from typing import Any, List, Optional

from langchain_core.tools import tool #type: ignore
from langsmith import traceable #type: ignore
import yaml

from src.chains.qa_chain import (
    build_citations,
    build_source_only_citations,
    format_docs,
    merge_same_location_docs,
)
from src.components.retriever import Retriever
from src.exception import CollectionNotFoundError, CustomException, KnowledgeBaseEmptyError
from src.logger import logging

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)


@tool
@traceable(run_type="tool", name="rag_query")
async def rag_query(
    query: str,
    collection_names: Optional[List[str]] = None,
    session_id: str = "default",
    user_id: str = "default",
) -> str:
    """Search the uploaded documents and knowledge base for relevant facts.

    Args:
        query: Specific question to look up in the documents.
        collection_names: Collections to scope the search to.
        session_id: Current user session ID.
        user_id: Current user ID.

    Returns:
        Retrieved document excerpts with source citations.
    """
    try:
        logging.info(
            "rag_query tool | query: %r | scope: %s | session: %s",
            query, collection_names, session_id
        )

        retriever = Retriever(collection_names=collection_names)
        ranked_docs = await retriever.retrieve_ranked(query)
        docs = [doc for doc, _, _ in ranked_docs]
        docs = merge_same_location_docs(docs)

        if not docs:
            return "No relevant information found in the documents for this query."

        formatted = format_docs(docs)
        citations = build_citations(docs)

        logging.info("rag_query retrieved %d chunks", len(docs))

        # Return formatted content with citations
        return f"{formatted}\n\n{citations}" if citations else formatted

    except (CollectionNotFoundError, KnowledgeBaseEmptyError) as e:
        logging.warning("rag_query collection error: %s", e)
        return "The document collection is empty or was not found."
    except Exception as e:
        logging.exception("rag_query tool failed")
        return f"Error querying documents: {str(e)}"


@tool
@traceable(run_type="tool", name="summarize_document")
async def summarize_document(
    collection_names: Optional[List[str]] = None,
    session_id: str = "default",
    user_id: str = "default",
) -> str:
    """Retrieve full document context for summarization.

    Args:
        collection_names: Collections to summarize.
        session_id: Current user session ID.
        user_id: Current user ID.

    Returns:
        Full document text with source citations.
    """
    try:
        logging.info(
            "summarize_document tool | scope: %s | session: %s",
            collection_names, session_id
        )

        retriever = Retriever(collection_names=collection_names)
        max_chars = config.get("retriever", {}).get("summary_max_chars", 6000)
        docs = await retriever.get_full_context(max_chars=max_chars)
        docs = merge_same_location_docs(docs)

        if not docs:
            return "No documents found to summarize."

        formatted = format_docs(docs)
        citations = build_source_only_citations(docs)

        logging.info("summarize_document retrieved %d chunks", len(docs))
        return f"{formatted}\n\n{citations}" if citations else formatted

    except (CollectionNotFoundError, KnowledgeBaseEmptyError) as e:
        logging.warning("summarize_document collection error: %s", e)
        return "The document collection is empty or was not found."
    except Exception as e:
        logging.exception("summarize_document tool failed")
        return f"Error retrieving document context: {str(e)}"


AVAILABLE_TOOLS = [rag_query, summarize_document]

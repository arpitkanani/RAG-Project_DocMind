import sys
from typing import Any, Dict

import yaml

from langsmith import traceable

from src.chains.qa_chain import (
    is_summary_request,
    merge_same_location_docs,
)
from src.components.memory_manager import MemoryManager
from src.components.retriever import Retriever
from src.exception import (
    CollectionNotFoundError,
    CustomException,
    KnowledgeBaseEmptyError,
)
from src.graph.state import RAGState
from src.logger import logging

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)


@traceable(name="Load_Context")
async def load_context_node(state: RAGState) -> Dict[str, Any]:
    """Asynchronously loads Postgres chat history and persists the human query."""
    try:
        session_id = state.get("session_id", "default")
        user_id = state["user_id"]
        question = state["question"]
        attachments = state.get("message_attachments")

        memory = MemoryManager(session_id=session_id, user_id=user_id)
        chat_history = await memory.aget_history()
        await memory.asave_message("human", question, attachments=attachments)

        is_summary = is_summary_request(question)
        return {
            "chat_history": chat_history,
            "is_summary": is_summary,
        }
    except Exception as e:
        raise CustomException(e, sys)


@traceable(name="Retrieve_QA")
async def retrieve_qa_node(state: RAGState) -> Dict[str, Any]:
    """Asynchronously executes QA vector search and lexical reranking."""
    try:
        question = state["question"]
        collection_names = state.get("collection_names")

        retriever = Retriever(collection_names=collection_names)
        ranked_docs = await retriever.retrieve_ranked(question)
        docs = [doc for doc, _, _ in ranked_docs]
        docs = merge_same_location_docs(docs)

        logging.info("context: question=%r | %d chunk(s) retrieved", question, len(docs))
        for i, doc in enumerate(docs, start=1):
            logging.info("context[%d]: %s", i, doc.page_content)

        return {
            "docs": docs,
        }
    except (CollectionNotFoundError, KnowledgeBaseEmptyError):
        raise
    except Exception as e:
        raise CustomException(e, sys)


@traceable(name="Retrieve_Summary")
async def retrieve_summary_node(state: RAGState) -> Dict[str, Any]:
    """Asynchronously retrieves full document context for summary requests."""
    try:
        collection_names = state.get("collection_names")
        retriever = Retriever(collection_names=collection_names)
        docs = await retriever.get_full_context(
            max_chars=config["retriever"].get("summary_max_chars", 6000)
        )
        docs = merge_same_location_docs(docs)

        logging.info("summary context: %d chunk(s) retrieved", len(docs))
        return {
            "docs": docs,
        }
    except (CollectionNotFoundError, KnowledgeBaseEmptyError):
        raise
    except Exception as e:
        raise CustomException(e, sys)

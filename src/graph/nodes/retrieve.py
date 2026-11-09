import sys
from typing import Any, Dict

import yaml

from langchain_core.runnables import RunnableConfig

from src.chains.qa_chain import (
    is_summary_request,
    merge_same_location_docs,
)
from src.components.memory_manager import MemoryManager
from src.components.retriever import Retriever
from src.exception import (
    CollectionEmbeddingMismatchError,
    CollectionNotFoundError,
    CustomException,
    KnowledgeBaseEmptyError,
)
from src.graph.state import RAGState
from src.logger import logging

with open("config/config.yaml") as f:
    APP_CONFIG = yaml.safe_load(f) or {}
config = APP_CONFIG


async def load_context_node(state: RAGState, config: RunnableConfig = None) -> Dict[str, Any]:
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


async def retrieve_qa_node(state: RAGState, config: RunnableConfig = None) -> Dict[str, Any]:
    """Asynchronously executes QA vector search and lexical reranking."""
    collection_names = state.get("collection_names")
    try:
        question = state["question"]
        if not collection_names:
            logging.info("retrieve_qa_node: no collections attached to session, returning empty docs")
            return {"docs": []}

        retriever = Retriever(collection_names=collection_names)
        ranked_docs = await retriever.retrieve_ranked(question, config=config)
        docs = [doc for doc, _, _ in ranked_docs]
        docs = merge_same_location_docs(docs)

        logging.info("context: question=%r | %d chunk(s) retrieved", question, len(docs))
        for i, doc in enumerate(docs, start=1):
            logging.debug("context[%d]: %s", i, doc.page_content)

        return {
            "docs": docs,
        }
    except CollectionEmbeddingMismatchError:
        raise
    except (CollectionNotFoundError, KnowledgeBaseEmptyError) as e:
        logging.warning("Knowledge base empty or collection missing in retrieve_qa_node: %s", e)
        return {"docs": []}
    except Exception as e:
        err_str = str(e).lower()
        if "vector dimension" in err_str or "dimension error" in err_str or "wrong input" in err_str:
            raise CollectionEmbeddingMismatchError(
                collection_name=collection_names[0] if collection_names else "document",
                expected_model="active model",
                actual_model="previous model",
                expected_dim=1536,
                actual_dim=384,
            )
        raise CustomException(e, sys)


async def retrieve_summary_node(state: RAGState, run_config: RunnableConfig = None) -> Dict[str, Any]:
    """Asynchronously retrieves full document context for summary requests."""
    collection_names = state.get("collection_names")
    try:
        if not collection_names:
            logging.info("retrieve_summary_node: no collections attached to session, returning empty docs")
            return {"docs": []}

        summary_limit = APP_CONFIG.get("retriever", {}).get("summary_max_chars", 6000)
        retriever = Retriever(collection_names=collection_names)
        docs = await retriever.get_full_context(
            max_chars=summary_limit
        )
        docs = merge_same_location_docs(docs)

        logging.info("summary context: %d chunk(s) retrieved", len(docs))
        return {
            "docs": docs,
        }
    except CollectionEmbeddingMismatchError:
        raise
    except (CollectionNotFoundError, KnowledgeBaseEmptyError) as e:
        logging.warning("Knowledge base empty or collection missing in retrieve_summary_node: %s", e)
        return {"docs": []}
    except Exception as e:
        err_str = str(e).lower()
        if "vector dimension" in err_str or "dimension error" in err_str or "wrong input" in err_str:
            raise CollectionEmbeddingMismatchError(
                collection_name=collection_names[0] if collection_names else "document",
                expected_model="active model",
                actual_model="previous model",
                expected_dim=1536,
                actual_dim=384,
            )
        raise CustomException(e, sys)

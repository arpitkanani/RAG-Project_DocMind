import sys
from typing import List

from src.chains.qa_chain import get_answer
from src.exception import (
    CollectionNotFoundError,
    CustomException,
    KnowledgeBaseEmptyError,
)
from src.logger import logging


class QAPipeline:
    """Runs every query through the RAG QA chain."""

    def __init__(
        self,
        collection_names: List[str] | None = None,
        session_id: str = "default",
    ):
        try:
            self.collection_names = [name for name in (collection_names or []) if name]
            self.session_id = session_id
            logging.info(
                "Pipeline ready | collections: %s | session: %s",
                self.collection_names or "all",
                session_id,
            )
        except Exception as e:
            raise CustomException(e, sys)

    def run(self, query: str) -> dict:
        try:
            logging.info("Processing query: %s...", query[:50])
            answer = get_answer(
                query,
                collection_names=self.collection_names,
                session_id=self.session_id,
            )
            return {
                "answer": answer,
                "collection_scope": self.collection_names or "all",
                "query": query,
                "session_id": self.session_id,
            }
        except (CollectionNotFoundError, KnowledgeBaseEmptyError):
            raise
        except Exception as e:
            raise CustomException(e, sys)

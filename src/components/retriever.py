import sys
import yaml
from langchain_core.documents import Document
from src.components.vector_store import VectorStore
from src.logger import logging
from src.exception import CustomException
from typing import List, Tuple

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)


class Retriever:
    """Handles retrieval strategies for DocMind RAG pipeline."""

    def __init__(self, collection_name: str = None):
        try:
            logging.info("Initializing Retriever")
            self.vs = VectorStore(collection_name=collection_name)
            self.search_type     = config["retriever"]["search_type"]
            self.k               = config["retriever"]["k"]
            self.score_threshold = config["retriever"]["score_threshold"]
            logging.info("Retriever initialized successfully")
        except Exception as e:
            raise CustomException(e, sys)

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve most relevant chunks for a query."""
        try:
            logging.info(f"Retrieving for: {query[:50]}...")
            retriever = self.get_retriever_object()
            docs = retriever.invoke(query)
            logging.info(f"Retrieved {len(docs)} chunks")
            return docs
        except Exception as e:
            raise CustomException(e, sys)

    def retrieve_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """Retrieve chunks with their similarity scores."""
        try:
            logging.info(f"Retrieving with scores: {query[:50]}...")
            db = self.vs.get_vectordb()
            results = db.similarity_search_with_score(query, k=self.k)
            logging.info(f"Retrieved {len(results)} chunks with scores")
            return results
        except Exception as e:
            raise CustomException(e, sys)

    def get_retriever_object(self):
        """Return LangChain retriever object for use in chains."""
        try:
            db = self.vs.get_vectordb()
            retriever = db.as_retriever(
                search_type=self.search_type,
                search_kwargs={
                    "k": self.k,
                    "score_threshold": self.score_threshold
                }
            )
            logging.info(
                f"Retriever object ready | "
                f"type: {self.search_type} | "
                f"threshold: {self.score_threshold}"
            )
            return retriever
        except Exception as e:
            raise CustomException(e, sys)
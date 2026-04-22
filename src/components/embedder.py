import sys
import yaml
from langchain_ollama import OllamaEmbeddings
from src.logger import logging
from src.exception import CustomException

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)


class Embedder:
    """Manages embedding model for DocMind project."""

    def __init__(self):
        try:
            logging.info("Initializing Embedder")
            self._initialize_model()
            logging.info(f"Embedder ready: {config['embedding']['model']}")
        except Exception as e:
            raise CustomException(e, sys)

    def _initialize_model(self):
        """Load embedding model from config."""
        try:
            self.embedding_model = OllamaEmbeddings(
                model=config["embedding"]["model"]
            )
            logging.info("Embedding model initialized")
        except Exception as e:
            raise CustomException(e, sys)

    def get_embedding_model(self) -> OllamaEmbeddings:
        """Return raw embedding model for VectorStore and Retriever."""
        return self.embedding_model

    def generate_embedding(self, text: str) -> list:
        """Generate embedding for single query text."""
        try:
            logging.info(f"Generating embedding: {text[:50]}...")
            vector = self.embedding_model.embed_query(text)
            logging.info(f"Embedding done: {len(vector)} dims")
            return vector
        except Exception as e:
            raise CustomException(e, sys)

    def generate_embeddings(self, texts: list) -> list:
        """Generate embeddings for multiple texts in batch."""
        try:
            logging.info(f"Batch embedding: {len(texts)} texts")
            vectors = self.embedding_model.embed_documents(texts)
            logging.info(f"Batch done: {len(vectors)} vectors")
            return vectors
        except Exception as e:
            raise CustomException(e, sys)
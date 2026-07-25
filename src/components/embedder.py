import os
import sys
import threading

import yaml

from src.logger import logging
from src.exception import CustomException

# Must be set BEFORE importing anything from huggingface_hub /
# sentence_transformers / langchain_community.embeddings, so no network
# calls are attempted at all. The model is already downloaded and cached
# locally from earlier runs, so there's no reason to ever hit
# huggingface.co again just for a routine "check for updates" request.
#
# This is what was causing the ~45s startup/refresh/delete lag: every
# VectorStore() -> Embedder() instantiation was trying (and failing,
# since huggingface.co isn't reachable from this environment) a HEAD
# request with 5 retries and exponential backoff, for SEVERAL files in a
# row, before eventually falling back to the local cache anyway.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)


class Embedder:
    """
    Manages the embedding model for DocMind, running LOCALLY via
    sentence-transformers (BAAI/bge-small-en-v1.5 or whatever
    config.yaml's embedding.model specifies).

    SINGLETON: the actual model is loaded into memory exactly ONCE per
    process (class-level cache, thread-safe via double-checked locking),
    and every subsequent Embedder() instance reuses that same loaded
    model. Without this, every VectorStore() call anywhere in the app --
    which happens on nearly every request: uploads, queries, session
    listing, session deletion, attachment removal -- would reload the
    full model from disk into memory from scratch every time, which is
    slow and was the direct cause of per-request lag across the app,
    not just on upload.
    """

    _model = None
    _lock = threading.Lock()

    def __init__(self):
        try:
            logging.info("Initializing Embedder")
            self._initialize_model()
            logging.info(f"Embedder ready: {config['embedding']['model']}")
        except Exception as e:
            raise CustomException(e, sys)

    def _initialize_model(self):
        try:
            if Embedder._model is not None:
                self.document_embedding_model = Embedder._model
                self.query_embedding_model = Embedder._model
                return

            with Embedder._lock:
                # Double-checked locking -- another thread may have
                # finished loading the model while we were waiting for
                # the lock, so re-check before loading again.
                if Embedder._model is None:
                    model_name = config["embedding"]["model"]
                    device = config["embedding"].get("device", "cpu")

                    logging.info(
                        "Loading embedding model into memory (first use only, "
                        "this process): %s",
                        model_name,
                    )
                    query_encode_kwargs = {"normalize_embeddings": True}
                    if "bge" in model_name.lower() and "bge-m3" not in model_name.lower():
                        query_encode_kwargs["prompt"] = ( # type: ignore
                            "Represent this question for searching relevant passages: "
                        )

                    Embedder._model = HuggingFaceEmbeddings(
                        model_name=model_name,
                        model_kwargs={"device": device},
                        encode_kwargs={"normalize_embeddings": True},
                        query_encode_kwargs=query_encode_kwargs,
                    )
                    logging.info(
                        "Embedding model loaded and cached for the rest of "
                        "this process's lifetime"
                    )

            self.document_embedding_model = Embedder._model
            self.query_embedding_model = Embedder._model
        except Exception as e:
            raise CustomException(e, sys)

    def get_embedding_model(self, for_query: bool = False) -> Embeddings:
        """
        for_query=False (default) -> used when storing chunks
        for_query=True            -> used when searching (BGE query prefix applied internally)
        """
        return self.query_embedding_model if for_query else self.document_embedding_model

    def generate_embedding(self, text: str) -> list:
        try:
            logging.info(f"Generating query embedding: {text[:50]}...")
            vector = self.query_embedding_model.embed_query(text)
            logging.info(f"Embedding done: {len(vector)} dims")
            return vector
        except Exception as e:
            raise CustomException(e, sys)

    def generate_embeddings(self, texts: list) -> list:
        try:
            logging.info(f"Batch embedding: {len(texts)} texts")
            vectors = self.document_embedding_model.embed_documents(texts)
            logging.info(f"Batch done: {len(vectors)} vectors")
            return vectors
        except Exception as e:
            raise CustomException(e, sys)
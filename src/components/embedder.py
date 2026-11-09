import os
import re
import sys
import threading
from typing import Optional

import yaml

from src.logger import logging
from src.exception import CustomException, EmbeddingUnavailableError
from langchain_core.embeddings import Embeddings

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)


def _normalize_azure_endpoint(endpoint: str) -> str:
    """Strips path segments like /openai/v1 so Azure clients get the clean base URL."""
    ep = (endpoint or "").strip().rstrip("/")
    ep = re.sub(r"/openai(?:/v\d+)?(?:/.*)?/?$", "", ep, flags=re.IGNORECASE)
    ep = re.sub(r"/responses/?$", "", ep, flags=re.IGNORECASE)
    return ep


class ResilientAzureEmbeddings(Embeddings):
    """Wraps AzureOpenAIEmbeddings to enforce explicit failure and raise EmbeddingUnavailableError
    after retries rather than silently switching models at runtime."""

    def __init__(self, inner_client):
        self._inner = inner_client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._inner.embed_documents(texts)
        except Exception as e:
            logging.error("Azure OpenAI embed_documents failed: %s", e)
            raise EmbeddingUnavailableError("The embedding model is unavailable. You can switch the embedding model in Settings.") from e

    def embed_query(self, text: str) -> list[float]:
        try:
            return self._inner.embed_query(text)
        except Exception as e:
            logging.error("Azure OpenAI embed_query failed: %s", e)
            raise EmbeddingUnavailableError("The embedding model is unavailable. You can switch the embedding model in Settings.") from e


class Embedder:
    """
    Manages embedding model providers for DocuVortex.
    Enforces the same embedding model per collection. NO silent runtime fallback.
    Supported providers:
      - 'azure': Azure OpenAI text-embedding-3-small (1536 dims, default)
      - 'bge': BAAI/bge-small-en-v1.5 (384 dims, local HuggingFace, lazy-loaded)
    """

    _azure_model = None
    _bge_model = None
    _azure_lock = threading.Lock()
    _bge_lock = threading.Lock()

    def __init__(self, provider: Optional[str] = None, user_id: Optional[str] = None):
        try:
            self.provider = (provider or Embedder.get_active_provider(user_id=user_id)).lower()
            self._model = self._initialize_provider(self.provider)
            self.document_embedding_model = self._model
            self.query_embedding_model = self._model
        except EmbeddingUnavailableError:
            raise
        except Exception as e:
            raise CustomException(e, sys)

    @classmethod
    def get_active_provider(cls, user_id: Optional[str] = None) -> str:
        """Determines the active embedding provider, checking user_settings in DB first, then config/env."""
        if user_id:
            try:
                from src.database.db import get_db_cursor
                with get_db_cursor(commit=False) as cur:
                    cur.execute("SELECT active_embedding_provider FROM user_settings WHERE user_id = %s LIMIT 1;", (user_id,))
                    row = cur.fetchone()
                    if row and row.get("active_embedding_provider"):
                        return row["active_embedding_provider"].strip().lower()
            except Exception as e:
                logging.debug("Could not read user_settings provider: %s", e)

        return os.getenv(
            "EMBEDDING_PROVIDER",
            config.get("embedding", {}).get("provider", "azure")
        ).strip().lower()

    @classmethod
    def get_model_specs(cls, provider: Optional[str] = None) -> dict:
        """Returns the canonical model name and dimension for a given provider."""
        p = (provider or "azure").strip().lower()
        if p == "bge":
            return {
                "provider": "bge",
                "model": config.get("embedding", {}).get("fallback_model", "BAAI/bge-small-en-v1.5"),
                "dimension": config.get("embedding", {}).get("fallback_dimension", 384),
            }
        return {
            "provider": "azure",
            "model": config.get("embedding", {}).get("model", "text-embedding-3-small"),
            "dimension": config.get("embedding", {}).get("dimension", 1536),
        }

    @classmethod
    def is_bge_available(cls) -> bool:
        """Checks whether the local HuggingFace environment / torch / transformers are functional."""
        try:
            import importlib.util
            has_torch = importlib.util.find_spec("torch") is not None
            has_hf = importlib.util.find_spec("sentence_transformers") is not None or importlib.util.find_spec("transformers") is not None
            return bool(has_torch and has_hf)
        except Exception:
            return False

    def _initialize_provider(self, provider: str) -> Embeddings:
        if provider == "bge":
            return self._get_bge_singleton()
        return self._get_azure_singleton()

    @classmethod
    def _get_azure_singleton(cls) -> Embeddings:
        if cls._azure_model is not None:
            return cls._azure_model

        with cls._azure_lock:
            if cls._azure_model is None:
                azure_key = (
                    os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY")
                    or os.getenv("AZURE_OPENAI_API_KEY")
                )
                raw_azure_endpoint = (
                    os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
                    or os.getenv("AZURE_OPENAI_ENDPOINT")
                )
                azure_deployment = (
                    os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")
                    or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
                    or config.get("embedding", {}).get("model", "text-embedding-3-small")
                ).strip("\"'")
                api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01").strip("\"'")

                if not azure_key or not raw_azure_endpoint:
                    raise EmbeddingUnavailableError("Azure OpenAI embedding credentials are not configured.")

                from langchain_openai import AzureOpenAIEmbeddings
                azure_endpoint = _normalize_azure_endpoint(raw_azure_endpoint)
                logging.info(
                    "Initializing Azure OpenAI Embeddings | Deployment: %s | Endpoint: %s",
                    azure_deployment,
                    azure_endpoint,
                )
                inner = AzureOpenAIEmbeddings(
                    azure_endpoint=azure_endpoint,
                    api_key=azure_key,
                    azure_deployment=azure_deployment,
                    api_version=api_version,
                    max_retries=config.get("embedding", {}).get("providers", {}).get("azure", {}).get("max_retries", 3),
                )
                cls._azure_model = ResilientAzureEmbeddings(inner)
        return cls._azure_model

    @classmethod
    def _get_bge_singleton(cls) -> Embeddings:
        """Loads HuggingFace BGE embeddings lazily ONLY when requested. Never loaded at Azure startup."""
        if cls._bge_model is not None:
            return cls._bge_model

        with cls._bge_lock:
            if cls._bge_model is None:
                # Lazy import prevents PyTorch/Transformers overhead for Azure-only deployments
                os.environ.setdefault("USE_TF", "0")
                os.environ.setdefault("USE_TORCH", "1")
                os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
                os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
                sys.modules["tensorflow"] = None

                from langchain_huggingface import HuggingFaceEmbeddings
                model_name = config.get("embedding", {}).get("fallback_model", "BAAI/bge-small-en-v1.5")
                device = config.get("embedding", {}).get("device", "cpu")

                logging.info("Lazy-loading HuggingFace local embedding model: %s on %s (loaded once)", model_name, device)
                query_encode_kwargs = {"normalize_embeddings": True}
                if "bge" in model_name.lower() and "bge-m3" not in model_name.lower():
                    query_encode_kwargs["prompt"] = (
                        "Represent this question for searching relevant passages: "
                    )

                cls._bge_model = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": device},
                    encode_kwargs={"normalize_embeddings": True},
                    query_encode_kwargs=query_encode_kwargs,
                )
                logging.info("HuggingFace BGE model cached in memory for process lifetime")
        return cls._bge_model

    def get_embedding_model(self, for_query: bool = False) -> Embeddings:
        return self.query_embedding_model if for_query else self.document_embedding_model

    def generate_embedding(self, text: str) -> list:
        try:
            return self.query_embedding_model.embed_query(text)
        except EmbeddingUnavailableError:
            raise
        except Exception as e:
            raise CustomException(e, sys)

    def generate_embeddings(self, texts: list) -> list:
        try:
            return self.document_embedding_model.embed_documents(texts)
        except EmbeddingUnavailableError:
            raise
        except Exception as e:
            raise CustomException(e, sys)

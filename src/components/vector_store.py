import asyncio
import os
import socket
import sys
import threading
import urllib.parse
from typing import Any, Iterable, List, Optional

import yaml
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient

from src.components.embedder import Embedder
from src.exception import (
    CollectionEmbeddingMismatchError,
    CollectionNotFoundError,
    CustomException,
    EmbeddingUnavailableError,
    KnowledgeBaseEmptyError,
)
from src.logger import logging

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)


def _is_remote_qdrant_up(url: str, api_key: str | None = None, timeout: float = 5.0) -> bool:
    """Check whether remote Qdrant is listening and reachable (supports local HTTP and Cloud HTTPS)."""
    if not url:
        return False
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme == "https":
            client = QdrantClient(url=url, api_key=api_key, timeout=timeout, check_compatibility=False)
            client.get_collections()
            return True
        else:
            host = parsed.hostname or "localhost"
            port = parsed.port or 6333
            with socket.create_connection((host, port), timeout=timeout):
                return True
    except Exception as e:
        logging.warning("Remote Qdrant check failed for %s: %s", url, e)
        return False


_local_client = None
_local_client_lock = threading.Lock()


def _get_local_client(storage_path: str = "data/qdrant_storage") -> QdrantClient:
    global _local_client
    if _local_client is None:
        with _local_client_lock:
            if _local_client is None:
                abs_path = os.path.abspath(storage_path)
                os.makedirs(abs_path, exist_ok=True)
                _local_client = QdrantClient(path=abs_path)
                logging.info("Initialized local disk QdrantClient at: %s", abs_path)
    return _local_client


class VectorStore:
    """Manages Qdrant vector storage for DocuVortex with sync and async operations,
    enforcing strict vector dimensions and collection model compatibility."""

    def __init__(self, collection_name: str | None = None, user_id: str | None = None, provider: str | None = None):
        try:
            self.user_id = user_id
            self.qdrant_url = os.environ.get(
                "QDRANT_URL", config.get("vectorstore", {}).get("url", "http://localhost:6333")
            )
            self.qdrant_api_key = os.environ.get("QDRANT_API_KEY")
            self.default_collection = config.get("vectorstore", {}).get("collection_name", "docmind")
            self.collection_name = collection_name or self.default_collection
            self.is_remote = _is_remote_qdrant_up(self.qdrant_url, api_key=self.qdrant_api_key)

            embedder = Embedder(provider=provider, user_id=user_id)
            self.embedder = embedder
            self.embedding_model = embedder.get_embedding_model(for_query=True)
            self.document_embedding_model = embedder.get_embedding_model(for_query=False)
            self.active_specs = Embedder.get_model_specs(embedder.provider)
        except EmbeddingUnavailableError:
            raise
        except Exception as e:
            raise CustomException(e, sys)

    def _client(self) -> QdrantClient:
        if self.is_remote:
            try:
                return QdrantClient(
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key,
                    timeout=35.0,
                    check_compatibility=False,
                )
            except Exception as e:
                logging.warning("Remote Qdrant connection to %s failed: %s. Using local disk fallback.", self.qdrant_url, e)
                self.is_remote = False
        return _get_local_client()

    def _async_client(self) -> AsyncQdrantClient:
        return AsyncQdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            timeout=35.0,
            check_compatibility=False,
        )

    def get_collection_vector_dimension(self, collection_name: str) -> Optional[int]:
        """Inspects Qdrant collection configuration to get the exact vector dimension."""
        client = self._client()
        try:
            info = client.get_collection(collection_name=collection_name)
            params = info.config.params.vectors
            if hasattr(params, "size"):
                return params.size
            if isinstance(params, dict) and "size" in params:
                return params["size"]
            if hasattr(params, "values"):
                # Named vectors case
                for v in params.values():
                    if hasattr(v, "size"):
                        return v.size
            return None
        except Exception as e:
            logging.debug("Could not inspect collection %s dimension: %s", collection_name, e)
            return None

    def verify_collection_compatibility(self, collection_name: str):
        """Ensures that the collection's vector dimension matches the active embedding model.
        Fails fast if there is a mismatch so incompatible searches never reach Qdrant."""
        actual_dim = self.get_collection_vector_dimension(collection_name)
        expected_dim = self.active_specs["dimension"]
        expected_model = self.active_specs["model"]

        if actual_dim is not None and actual_dim != expected_dim:
            actual_model = "BAAI/bge-small-en-v1.5" if actual_dim == 384 else "text-embedding-3-small"
            raise CollectionEmbeddingMismatchError(
                collection_name=collection_name,
                expected_model=expected_model,
                actual_model=actual_model,
                expected_dim=expected_dim,
                actual_dim=actual_dim,
            )

    def _initialize_vectordb(self) -> QdrantVectorStore:
        try:
            all_collections = self.list_collections()
            if not all_collections:
                raise KnowledgeBaseEmptyError("No collections found. Upload a document first.")
            if self.collection_name not in all_collections:
                raise CollectionNotFoundError([self.collection_name])

            # Fail fast if dimension does not match active embedding model
            self.verify_collection_compatibility(self.collection_name)

            client = self._client()
            return QdrantVectorStore(
                client=client,
                collection_name=self.collection_name,
                embedding=self.embedding_model,
            )
        except (CollectionNotFoundError, KnowledgeBaseEmptyError, CollectionEmbeddingMismatchError):
            raise
        except Exception as e:
            raise CustomException(e, sys)

    def add_documents(
        self,
        chunks: Iterable[Document],
        batch_size: int = 250,
        progress_callback=None,
        on_retry=None,
    ) -> QdrantVectorStore:
        try:
            logging.info("Starting batch ingestion | collection: %s (remote=%s)", self.collection_name, self.is_remote)
            total_added = 0
            current_batch = []
            db = None

            def _save_batch(batch: List[Document]) -> QdrantVectorStore:
                client = self._client()
                try:
                    exists = client.collection_exists(self.collection_name)
                except Exception:
                    try:
                        colls = [c.name for c in client.get_collections().collections]
                        exists = self.collection_name in colls
                    except Exception:
                        exists = False

                expected_dim = self.active_specs["dimension"]

                if not exists:
                    from qdrant_client.http import models as rest
                    try:
                        client.create_collection(
                            collection_name=self.collection_name,
                            vectors_config=rest.VectorParams(size=expected_dim, distance=rest.Distance.COSINE),
                        )
                        logging.info("Created collection %s with vector dimension %d", self.collection_name, expected_dim)
                    except Exception as coll_err:
                        logging.warning("Collection creation notice: %s", coll_err)
                else:
                    # Enforce that existing collection matches our active dimension before inserting
                    self.verify_collection_compatibility(self.collection_name)

                store = QdrantVectorStore(
                    client=client,
                    collection_name=self.collection_name,
                    embedding=self.document_embedding_model,
                )
                store.add_documents(batch)
                return store

            def _store_batch_with_fallback(batch: List[Document]) -> QdrantVectorStore:
                if self.is_remote:
                    try:
                        return _save_batch(batch)
                    except Exception as rem_err:
                        logging.warning("Remote Qdrant batch failed: %s. Falling back to local storage.", rem_err)
                        self.is_remote = False
                return _save_batch(batch)

            for chunk in chunks:
                chunk.metadata["doc_index"] = total_added + len(current_batch)
                chunk.metadata["content_length"] = len(chunk.page_content)
                chunk.metadata["collection_name"] = self.collection_name
                chunk.metadata["embedding_model"] = self.active_specs["model"]
                chunk.metadata["vector_dim"] = self.active_specs["dimension"]
                current_batch.append(chunk)

                if len(current_batch) >= batch_size:
                    db = _store_batch_with_fallback(current_batch)
                    total_added += len(current_batch)
                    current_batch = []
                    if progress_callback:
                        progress_callback(total_added)

            if current_batch:
                db = _store_batch_with_fallback(current_batch)
                total_added += len(current_batch)
                if progress_callback:
                    progress_callback(total_added)

            logging.info("Successfully added %d chunks", total_added)
            return db
        except (CollectionEmbeddingMismatchError, EmbeddingUnavailableError):
            raise
        except Exception as e:
            raise CustomException(e, sys)

    def get_all_documents(self) -> List[Document]:
        return asyncio.run(self.aget_all_documents())

    async def aget_all_documents(self) -> List[Document]:
        try:
            if self.is_remote:
                client = self._async_client()
                try:
                    all_collections = await self.alist_collections()
                    if not all_collections:
                        return []

                    collections_to_search = (
                        [self.collection_name]
                        if self.collection_name != self.default_collection and self.collection_name in all_collections
                        else all_collections
                    )

                    docs: List[Document] = []
                    for collection_name in collections_to_search:
                        offset = None
                        while True:
                            points, offset = await client.scroll(
                                collection_name=collection_name,
                                limit=256,
                                offset=offset,
                                with_payload=True,
                                with_vectors=False,
                            )
                            for point in points:
                                payload = point.payload or {}
                                text = payload.get("page_content", "")
                                metadata = payload.get("metadata", {}) or {}
                                if text and text.strip():
                                    docs.append(Document(page_content=text, metadata=metadata))
                            if offset is None:
                                break

                    return docs
                except Exception as rem_err:
                    logging.warning("Remote aget_all_documents failed: %s. Trying local fallback.", rem_err)
                    self.is_remote = False
                finally:
                    try:
                        await client.close()
                    except Exception:
                        pass

            client = self._client()
            def _sync_scroll():
                all_colls = [c.name for c in client.get_collections().collections]
                if not all_colls:
                    return []
                collections_to_search = (
                    [self.collection_name]
                    if self.collection_name != self.default_collection and self.collection_name in all_colls
                    else all_colls
                )
                local_docs = []
                for c_name in collections_to_search:
                    offset = None
                    while True:
                        points, offset = client.scroll(
                            collection_name=c_name,
                            limit=256,
                            offset=offset,
                            with_payload=True,
                            with_vectors=False,
                        )
                        for point in points:
                            payload = point.payload or {}
                            text = payload.get("page_content", "")
                            metadata = payload.get("metadata", {}) or {}
                            if text and text.strip():
                                local_docs.append(Document(page_content=text, metadata=metadata))
                        if offset is None:
                            break
                return local_docs
            return await asyncio.to_thread(_sync_scroll)
        except Exception as e:
            raise CustomException(e, sys)

    def get_vectordb(self) -> QdrantVectorStore:
        return self._initialize_vectordb()

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        return self._initialize_vectordb().similarity_search(query, k=k)

    def list_collections(self) -> List[str]:
        client = self._client()
        try:
            return [c.name for c in client.get_collections().collections]
        except Exception as e:
            logging.warning("Could not list collections via client: %s", e)
            try:
                local_client = _get_local_client()
                return [c.name for c in local_client.get_collections().collections]
            except Exception:
                return []

    async def alist_collections(self) -> List[str]:
        if self.is_remote:
            client = self._async_client()
            try:
                colls = await client.get_collections()
                return [c.name for c in colls.collections]
            except Exception as e:
                logging.warning("Could not reach remote Qdrant on %s: %s", self.qdrant_url, e)
                try:
                    local_client = _get_local_client()
                    colls = await asyncio.to_thread(local_client.get_collections)
                    return [c.name for c in colls.collections]
                except Exception:
                    return []
            finally:
                try:
                    await client.close()
                except Exception:
                    pass
        else:
            try:
                local_client = self._client()
                colls = await asyncio.to_thread(local_client.get_collections)
                return [c.name for c in colls.collections]
            except Exception as e:
                logging.warning("Could not reach local Qdrant: %s", e)
                return []

    def delete_collection(self) -> bool:
        client = self._client()
        try:
            collections = [c.name for c in client.get_collections().collections]
            if self.collection_name not in collections:
                return False
            client.delete_collection(self.collection_name)
            from src.components.retriever import invalidate_cached_db
            invalidate_cached_db(self.collection_name)
            return True
        except Exception as e:
            logging.warning("Failed to delete collection %s: %s", self.collection_name, e)
            return False

    async def adelete_collection(self) -> bool:
        if self.is_remote:
            client = self._async_client()
            try:
                colls = await client.get_collections()
                collections = [c.name for c in colls.collections]
                if self.collection_name not in collections:
                    return False
                await client.delete_collection(self.collection_name)
                from src.components.retriever import invalidate_cached_db
                invalidate_cached_db(self.collection_name)
                return True
            except Exception as e:
                logging.warning("Failed to delete remote collection %s: %s", self.collection_name, e)
                return False
            finally:
                try:
                    await client.close()
                except Exception:
                    pass
        else:
            try:
                client = self._client()
                colls = await asyncio.to_thread(client.get_collections)
                collections = [c.name for c in colls.collections]
                if self.collection_name not in collections:
                    return False
                await asyncio.to_thread(client.delete_collection, self.collection_name)
                from src.components.retriever import invalidate_cached_db
                invalidate_cached_db(self.collection_name)
                return True
            except Exception as e:
                logging.warning("Failed to delete local collection %s: %s", self.collection_name, e)
                return False

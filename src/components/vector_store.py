import asyncio
import os
import socket
import sys
import threading
import urllib.parse
from typing import Any, Iterable, List

import yaml
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient

from src.components.embedder import Embedder
from src.exception import (
    CollectionNotFoundError,
    CustomException,
    KnowledgeBaseEmptyError,
)
from src.logger import logging

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)


def _is_remote_qdrant_up(url: str, timeout: float = 0.5) -> bool:
    """Quick socket check to verify whether remote Qdrant port is listening."""
    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 6333
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
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
    automatically falling back to local embedded disk storage if remote Qdrant is unreachable."""

    def __init__(self, collection_name: str | None = None):
        try:
            # Prefer QDRANT_URL env var (set by docker-compose) over config.yaml
            self.qdrant_url = os.environ.get(
                "QDRANT_URL", config["vectorstore"]["url"]
            )
            self.default_collection = config["vectorstore"]["collection_name"]
            self.collection_name = collection_name or self.default_collection
            self.is_remote = _is_remote_qdrant_up(self.qdrant_url)
            embedder = Embedder()
            self.embedding_model = embedder.get_embedding_model(for_query=True)
            self.document_embedding_model = embedder.get_embedding_model(for_query=False)
        except Exception as e:
            raise CustomException(e, sys)

    def _client(self) -> QdrantClient:
        if self.is_remote:
            try:
                return QdrantClient(url=self.qdrant_url, timeout=5)
            except Exception as e:
                logging.warning("Remote Qdrant connection to %s failed: %s. Using local disk fallback.", self.qdrant_url, e)
                self.is_remote = False
        return _get_local_client()

    def _async_client(self) -> AsyncQdrantClient:
        return AsyncQdrantClient(url=self.qdrant_url, timeout=5)

    def _initialize_vectordb(self) -> QdrantVectorStore:
        try:
            all_collections = self.list_collections()
            if not all_collections:
                raise KnowledgeBaseEmptyError("No collections found. Upload a document first.")
            if self.collection_name not in all_collections:
                raise CollectionNotFoundError([self.collection_name])

            client = self._client()
            return QdrantVectorStore(
                client=client,
                collection_name=self.collection_name,
                embedding=self.embedding_model,
            )
        except (CollectionNotFoundError, KnowledgeBaseEmptyError):
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

                if not exists:
                    from qdrant_client.http import models as rest
                    try:
                        sample_vec = self.document_embedding_model.embed_query("doc")
                        dim = len(sample_vec)
                    except Exception:
                        dim = 384
                    try:
                        client.create_collection(
                            collection_name=self.collection_name,
                            vectors_config=rest.VectorParams(size=dim, distance=rest.Distance.COSINE),
                        )
                    except Exception as coll_err:
                        logging.warning("Collection creation notice: %s", coll_err)

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
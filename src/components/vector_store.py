import asyncio
import os
import sys
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


class VectorStore:
    """Manages Qdrant vector storage for DocMind with sync and async operations."""

    def __init__(self, collection_name: str | None = None):
        try:
            # Prefer QDRANT_URL env var (set by docker-compose) over config.yaml
            self.qdrant_url = os.environ.get(
                "QDRANT_URL", config["vectorstore"]["url"]
            )
            self.default_collection = config["vectorstore"]["collection_name"]
            self.collection_name = collection_name or self.default_collection
            embedder = Embedder()
            self.embedding_model = embedder.get_embedding_model(for_query=True)
            self.document_embedding_model = embedder.get_embedding_model(for_query=False)
        except Exception as e:
            raise CustomException(e, sys)

    def _client(self) -> QdrantClient:
        return QdrantClient(url=self.qdrant_url)

    def _async_client(self) -> AsyncQdrantClient:
        return AsyncQdrantClient(url=self.qdrant_url)

    def _initialize_vectordb(self) -> QdrantVectorStore:
        try:
            all_collections = self.list_collections()
            if not all_collections:
                raise KnowledgeBaseEmptyError("No collections found. Upload a document first.")
            if self.collection_name not in all_collections:
                raise CollectionNotFoundError([self.collection_name])

            return QdrantVectorStore.from_existing_collection(
                embedding=self.embedding_model,
                collection_name=self.collection_name,
                url=self.qdrant_url,
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
            logging.info("Starting batch ingestion | collection: %s", self.collection_name)
            total_added = 0
            current_batch = []
            db = None

            for chunk in chunks:
                chunk.metadata["doc_index"] = total_added + len(current_batch)
                chunk.metadata["content_length"] = len(chunk.page_content)
                chunk.metadata["collection_name"] = self.collection_name
                current_batch.append(chunk)

                if len(current_batch) >= batch_size:
                    db = QdrantVectorStore.from_documents(
                        documents=current_batch,
                        embedding=self.document_embedding_model,
                        url=self.qdrant_url,
                        collection_name=self.collection_name,
                    )
                    total_added += len(current_batch)
                    current_batch = []
                    if progress_callback:
                        progress_callback(total_added)

            if current_batch:
                db = QdrantVectorStore.from_documents(
                    documents=current_batch,
                    embedding=self.document_embedding_model,
                    url=self.qdrant_url,
                    collection_name=self.collection_name,
                )
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
            finally:
                await client.close()
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
        finally:
            client.close()

    async def alist_collections(self) -> List[str]:
        client = self._async_client()
        try:
            colls = await client.get_collections()
            return [c.name for c in colls.collections]
        finally:
            await client.close()

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
        finally:
            client.close()

    async def adelete_collection(self) -> bool:
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
        finally:
            await client.close()
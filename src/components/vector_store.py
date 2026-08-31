import sys
from typing import List, Iterable

import yaml
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient

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
    """Manages Qdrant vector storage for DocMind.

    Embeddings are now generated LOCALLY (BGE via sentence-transformers),
    so the rate-limit retry loop that used to wrap add_documents() for
    Gemini's 429 errors has been removed -- there is no external quota
    to hit anymore, and no cleanup-and-retry cycle is needed.

    NOTE: the on_retry parameter is still accepted (as a no-op) purely
    so existing callers -- ingestion_pipeline.py, and app.py's
    _run_upload_job / _run_youtube_job background functions -- don't
    need their call signatures changed. It's simply never invoked now.
    """

    def __init__(self, collection_name: str | None = None):
        try:
            logging.info("Initializing VectorStore")
            self.qdrant_url = config["vectorstore"]["url"]
            self.default_collection = config["vectorstore"]["collection_name"]
            self.collection_name = collection_name or self.default_collection
            embedder = Embedder()
            self.embedding_model = embedder.get_embedding_model(for_query=True)
            self.document_embedding_model = embedder.get_embedding_model(for_query=False)
            logging.info(
                "VectorStore ready | collection: %s | url: %s",
                self.collection_name,
                self.qdrant_url,
            )
        except Exception as e:
            raise CustomException(e, sys)

    def _client(self) -> QdrantClient:
        """Short-lived client — always closed after use."""
        return QdrantClient(url=self.qdrant_url)

    def _initialize_vectordb(self) -> QdrantVectorStore:
        """Load an existing Qdrant collection."""
        try:
            all_collections = self.list_collections()
            if not all_collections:
                raise KnowledgeBaseEmptyError(
                    "No collections found. Upload a document first."
                )

            if self.collection_name not in all_collections:
                raise CollectionNotFoundError([self.collection_name])

            db = QdrantVectorStore.from_existing_collection(
                embedding=self.embedding_model,
                collection_name=self.collection_name,
                url=self.qdrant_url,
            )
            logging.info("Qdrant collection loaded | collection: %s", self.collection_name)
            return db
        except (CollectionNotFoundError, KnowledgeBaseEmptyError):
            raise
        except Exception as e:
            raise CustomException(e, sys)

    def add_documents(
        self,
        chunks: Iterable[Document],
        batch_size: int = 250,
        progress_callback = None,
        on_retry=None,
    ) -> QdrantVectorStore:
        """Embed and store document chunks in Qdrant in batches.

        on_retry is unused now (kept only for signature compatibility --
        see class docstring). Local embeddings never hit an external
        rate limit, so this is now a single straightforward call.
        """
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
        """Fetch all stored documents for this collection, or all collections."""
        try:
            client = self._client()
            try:
                all_collections = self.list_collections()
                if not all_collections:
                    return []

                if self.collection_name != self.default_collection:
                    collections_to_search = (
                        [self.collection_name]
                        if self.collection_name in all_collections
                        else []
                    )
                else:
                    collections_to_search = all_collections

                docs: List[Document] = []
                for collection_name in collections_to_search:
                    offset = None
                    while True:
                        points, offset = client.scroll(
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

                logging.info("Retrieved %d documents from vector store", len(docs))
                return docs
            finally:
                client.close()
        except Exception as e:
            raise CustomException(e, sys)

    def get_vectordb(self) -> QdrantVectorStore:
        """Return raw Qdrant vector store object for Retriever."""
        try:
            return self._initialize_vectordb()
        except (CollectionNotFoundError, KnowledgeBaseEmptyError):
            raise
        except Exception as e:
            raise CustomException(e, sys)

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Basic similarity search for testing."""
        try:
            db = self._initialize_vectordb()
            results = db.similarity_search(query, k=k)
            logging.info("Search returned %d results", len(results))
            return results
        except (CollectionNotFoundError, KnowledgeBaseEmptyError):
            raise
        except Exception as e:
            raise CustomException(e, sys)

    def exists(self) -> bool:
        """Check if the configured collection exists."""
        try:
            collections = self.list_collections()
            if self.collection_name == self.default_collection:
                result = bool(collections)
            else:
                result = self.collection_name in collections
            logging.info("Collection '%s' exists: %s", self.collection_name, result)
            return result
        except Exception as e:
            raise CustomException(e, sys)

    def list_collections(self) -> List[str]:
        """Return all collection names in the vector store."""
        try:
            client = self._client()
            try:
                names = [c.name for c in client.get_collections().collections]
                logging.info("Collections found: %s", names)
                return names
            finally:
                client.close()
        except Exception as e:
            raise CustomException(e, sys)

    def delete_collection(self) -> bool:
        """Delete only the configured collection and return whether it existed."""
        try:
            client = self._client()
            try:
                collections = [c.name for c in client.get_collections().collections]
                if self.collection_name not in collections:
                    logging.info("Collection not found: %s", self.collection_name)
                    return False
                client.delete_collection(self.collection_name)
                logging.info("Collection deleted: %s", self.collection_name)
                from src.components.retriever import invalidate_cached_db
                invalidate_cached_db(self.collection_name)
                return True
            finally:
                client.close()
        except Exception as e:
            raise CustomException(e, sys)

    def delete_all_collections(self) -> list[str]:
        """Delete every collection from the vector store."""
        try:
            client = self._client()
            try:
                names = [c.name for c in client.get_collections().collections]
                from src.components.retriever import invalidate_cached_db
                for name in names:
                    client.delete_collection(name)
                    invalidate_cached_db(name)
                    logging.info("Collection deleted during reset: %s", name)
                return names
            finally:
                client.close()
        except Exception as e:
            raise CustomException(e, sys)
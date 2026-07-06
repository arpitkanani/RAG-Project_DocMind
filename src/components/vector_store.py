import os
import sys
from typing import List

import yaml
from langchain_chroma import Chroma
from langchain_core.documents import Document

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
    """Manages ChromaDB vector storage for DocMind."""

    def __init__(self, collection_name: str | None = None):
        try:
            logging.info("Initializing VectorStore")
            self.persist_dir = config["vectorstore"]["persist_directory"]
            self.default_collection = config["vectorstore"]["collection_name"]
            self.collection_name = collection_name or self.default_collection
            embedder = Embedder()
            self.embedding_model = embedder.get_embedding_model()
            os.makedirs(self.persist_dir, exist_ok=True)
            logging.info("VectorStore ready | collection: %s", self.collection_name)
        except Exception as e:
            raise CustomException(e, sys)

    def _initialize_vectordb(self) -> Chroma:
        """Load an existing Chroma collection from disk."""
        try:
            all_collections = self.list_collections()
            if not all_collections:
                raise KnowledgeBaseEmptyError(
                    "No collections found. Upload a document first."
                )

            if self.collection_name == self.default_collection:
                collection_name = all_collections[0]
            elif self.collection_name in all_collections:
                collection_name = self.collection_name
            else:
                raise CollectionNotFoundError([self.collection_name])

            db = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embedding_model,
                collection_name=collection_name,
            )

            logging.info(
                "ChromaDB loaded | collection: %s | vectors: %s",
                collection_name,
                db._collection.count(),
            )
            return db
        except (CollectionNotFoundError, KnowledgeBaseEmptyError):
            raise
        except Exception as e:
            raise CustomException(e, sys)

    def add_documents(self, chunks: List[Document]) -> Chroma:
        """Embed and store document chunks in ChromaDB."""
        try:
            logging.info(
                "Adding %d chunks | collection: %s",
                len(chunks),
                self.collection_name,
            )

            for index, chunk in enumerate(chunks):
                chunk.metadata["doc_index"] = index
                chunk.metadata["content_length"] = len(chunk.page_content)
                chunk.metadata["collection_name"] = self.collection_name

            db = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                persist_directory=self.persist_dir,
                collection_name=self.collection_name,
            )
            logging.info("Successfully added %d chunks", len(chunks))
            return db
        except Exception as e:
            raise CustomException(e, sys)

    def get_all_documents(self) -> List[Document]:
        """Fetch all stored documents for a collection or all collections."""
        try:
            import chromadb

            client = chromadb.PersistentClient(path=self.persist_dir)
            all_collections = client.list_collections()
            if not all_collections:
                return []

            if self.collection_name != self.default_collection:
                collections_to_search = [self.collection_name]
            else:
                collections_to_search = [collection.name for collection in all_collections]

            docs: List[Document] = []
            for collection_name in collections_to_search:
                if collection_name not in [c.name for c in all_collections]:
                    continue

                db = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=self.embedding_model,
                    collection_name=collection_name,
                )
                results = db.get()

                for text, metadata in zip(results["documents"], results["metadatas"]):
                    if text and text.strip():
                        docs.append(Document(page_content=text, metadata=metadata))

            logging.info("Retrieved %d documents from vector store", len(docs))
            return docs
        except Exception as e:
            raise CustomException(e, sys)

    def get_vectordb(self) -> Chroma:
        """Return raw ChromaDB object for Retriever."""
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
        """Return all collection names in vectorstore."""
        try:
            import chromadb

            client = chromadb.PersistentClient(path=self.persist_dir)
            names = [collection.name for collection in client.list_collections()]
            logging.info("Collections found: %s", names)
            return names
        except Exception as e:
            raise CustomException(e, sys)

    def delete_collection(self) -> bool:
        """Delete only the configured collection and return whether it existed."""
        try:
            import chromadb
            import gc
            import time

            gc.collect()
            client = chromadb.PersistentClient(path=self.persist_dir)
            collections = [collection.name for collection in client.list_collections()]

            if self.collection_name not in collections:
                logging.info("Collection not found: %s", self.collection_name)
                return False

            client.delete_collection(self.collection_name)
            logging.info("Collection deleted: %s", self.collection_name)

            del client
            gc.collect()
            time.sleep(0.3)
            return True
        except Exception as e:
            raise CustomException(e, sys)

    def delete_all_collections(self) -> list[str]:
        """Delete every collection from the vector store."""
        try:
            import chromadb
            import gc
            import time

            gc.collect()
            client = chromadb.PersistentClient(path=self.persist_dir)
            names = [collection.name for collection in client.list_collections()]

            for name in names:
                client.delete_collection(name)
                logging.info("Collection deleted during reset: %s", name)

            del client
            gc.collect()
            time.sleep(0.3)
            return names
        except Exception as e:
            raise CustomException(e, sys)

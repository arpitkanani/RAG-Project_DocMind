import sys
import os
import yaml
import shutil
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.components.embedder import Embedder
from src.logger import logging
from src.exception import CustomException
from typing import List

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)


class VectorStore:
    """Manages ChromaDB vector storage for DocMind."""

    def __init__(self, collection_name: str = None):
        try:
            logging.info("Initializing VectorStore")
            self.persist_dir = config["vectorstore"]["persist_directory"]
            self.collection_name = (
                collection_name or
                config["vectorstore"]["collection_name"]
            )
            embedder = Embedder()
            self.embedding_model = embedder.get_embedding_model()
            os.makedirs(self.persist_dir, exist_ok=True)
            logging.info(f"VectorStore ready | collection: {self.collection_name}")
        except Exception as e:
            raise CustomException(e, sys)

    def _initialize_vectordb(self) -> Chroma:
        """Load existing ChromaDB from disk."""
        try:
            if not self.exists():
                raise FileNotFoundError(
                    f"No vector store found at {self.persist_dir}. "
                    "Upload a document first."
                )
            db = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embedding_model,
                collection_name=self.collection_name
            )
            logging.info(f"ChromaDB loaded: {db._collection.count()} vectors")
            return db
        except Exception as e:
            raise CustomException(e, sys)

    def add_documents(self, chunks: List[Document]) -> Chroma:
        """Embed and store document chunks in ChromaDB."""
        try:
            logging.info(f"Adding {len(chunks)} chunks | collection: {self.collection_name}")

            # enrich metadata with position and size info
            for i, chunk in enumerate(chunks):
                chunk.metadata["doc_index"] = i
                chunk.metadata["content_length"] = len(chunk.page_content)

            db = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                persist_directory=self.persist_dir,
                collection_name=self.collection_name
            )
            logging.info(f"Successfully added {len(chunks)} chunks")
            return db
        except Exception as e:
            raise CustomException(e, sys)

    def get_all_documents(self) -> List[Document]:
        """Fetch all stored documents for summarization."""
        try:
            db = self._initialize_vectordb()
            results = db.get()
            docs = [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(
                    results["documents"],
                    results["metadatas"]
                )
                if text.strip()
            ]
            logging.info(f"Retrieved {len(docs)} total documents")
            return docs
        except Exception as e:
            raise CustomException(e, sys)

    def get_vectordb(self) -> Chroma:
        """Return raw ChromaDB object for Retriever."""
        try:
            return self._initialize_vectordb()
        except Exception as e:
            raise CustomException(e, sys)

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Basic similarity search for testing."""
        try:
            db = self._initialize_vectordb()
            results = db.similarity_search(query, k=k)
            logging.info(f"Search returned {len(results)} results")
            return results
        except Exception as e:
            raise CustomException(e, sys)

    def exists(self) -> bool:
        """Check if vector store has data on disk."""
        try:
            result = (
                os.path.exists(self.persist_dir) and
                len(os.listdir(self.persist_dir)) > 0
            )
            logging.info(f"VectorStore exists: {result}")
            return result
        except Exception as e:
            raise CustomException(e, sys)

    def delete_collection(self):
        """Delete all vectors and reset storage."""
        try:
            import chromadb
            import gc
            import time

            # close any open ChromaDB connections first
            # gc.collect() forces Python garbage collector
            # to clean up unreferenced ChromaDB objects
            # releasing file locks on Windows 
            gc.collect()

            if os.path.exists(self.persist_dir):
                # try deleting via ChromaDB client first
                # cleaner than deleting folder directly
                try:
                    client = chromadb.PersistentClient(
                        path=self.persist_dir
                    )
                    collections = client.list_collections()
                    for col in collections:
                        client.delete_collection(col.name)
                    del client
                    gc.collect()
                    # small wait for Windows to release lock
                    time.sleep(0.5)
                except Exception:
                    pass

                # now safe to delete folder
                try:
                    shutil.rmtree(self.persist_dir)
                except PermissionError:
                    # if still locked → delete files individually
                    for root, dirs, files in os.walk(
                        self.persist_dir, topdown=False
                    ):
                        for file in files:
                            try:
                                os.remove(os.path.join(root, file))
                            except Exception:
                                pass

            os.makedirs(self.persist_dir, exist_ok=True)
            logging.info("Collection deleted successfully")

        except Exception as e:
            raise CustomException(e, sys)#type:ignore
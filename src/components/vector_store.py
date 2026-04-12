import sys
import os
import yaml
import shutil
from src.logger import logging
from src.exception import CustomException
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.components.embedder import Embedder
from typing import List

with open("D:\Langchain Project\config\config.yaml") as f:
    config = yaml.safe_load(f)


class VectorStore:

    """Manages ChromaDB vector database"""
    def __init__(self):
        try:
            logging.info("Initalizing VectorStore ")

            self.persist_dir=config["vectorstore"]['persist_directory']
            self.collection_name = config["vectorstore"]["collection_name"]

            embedder=Embedder()

            self.embeddings=embedder.get_embedding()

            os.makedirs(self.persist_dir,exist_ok=True)
            logging.info(
                f"VectorStore config ready | "
                f"dir: {self.persist_dir} | "
                f"collection: {self.collection_name}"
            )

        except Exception as e:
            raise CustomException(e, sys)#type:ignore
        
    def store(self,chunks:List[Document],collection_name: str = None) -> Chroma:

        """
        Embed all chunks and store in Chroma
        input:  List[Document] from TextSplitter
        output: Chroma DB object ready for search
        """

        try:
            collection = collection_name or self.collection_name

            logging.info(
                f"Storing {len(chunks)} chunks in ChromaDB"
            )

            db= Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_dir,
                collection_name=collection
            )

            logging.info(f"Stored successfully in collection: {collection}")
            return db
        except Exception as e:
            raise CustomException(e, sys)#type:ignore
        
    def load(self,collection_name: str = None) -> Chroma:

        """
        load Existing Data of ChromaDB from disk

        called EVERY TIME user asks a question
        because we need DB to search against

        raises error if DB doesn't exist yet
        meaning no document uploaded yet
        """
        try:
            collection = collection_name or self.collection_name
            if not self.exists(): 
                raise FileNotFoundError(
                    f"No vector store found at {self.persist_dir}. "
                    f"Please upload and process a document first."
                )

            logging.info(f"Loading collection: {collection}")

            db = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
                collection_name=collection
            )
            count =db._collection.count()
            logging.info(f"ChromaDB loaded | {count} vectors available ")

            return db
        except Exception as e:
            raise CustomException(e,sys)#type:ignore
        
    def load_all(self) -> Chroma:
        """
        Load ALL documents across all collections
        used when user asks question across multiple docs

        ChromaDB trick:
        use same persist_dir but no collection filter
        searches across everything stored 
        """
        try:
            logging.info("Loading all collections for cross-doc search")

            db = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
                # no collection_name → searches ALL 
            )

            count = db._collection.count()
            logging.info(f"All collections loaded: {count} total vectors")
            return db

        except Exception as e:
            raise CustomException(e, sys)#type:ignore


    def exists(self)->bool:
        """ Check if vector store has alredy data or not."""
        try:
            exists=(
                os.path.exists(self.persist_dir) and
                len(os.listdir(self.persist_dir))>0
            )    
            logging.info(f"VectorStore exists: {exists}")
            return exists
        except Exception as e:
            raise CustomException(e,sys)#type:ignore
        
    def clear(self):
        """
        Delete all stored Vectors
        """
        try:
            if os.path.exists(self.persist_dir):
                shutil.rmtree(self.persist_dir)
            
            os.makedirs(self.persist_dir, exist_ok=True)
            logging.info("VectorStore cleared successfully")

        except Exception as e:
            raise CustomException(e, sys)#type:ignore
        

    def get_retriever(self, collection_name: str = None,
                  search_across_all: bool = False):
        """
        Get retriever for specific doc or all docs

        search_across_all=True  → searches all uploaded documents
        search_across_all=False → searches specific collection
        """
        try:
            search_type = config["retriever"]["search_type"]
            k           = config["retriever"]["k"]
            fetch_k     = config["retriever"]["fetch_k"]
            lambda_mult = config["retriever"]["lambda_mult"]

            # decide which DB to load
            if search_across_all:
                db = self.load_all()
            else:
                db = self.load(collection_name)

            retriever = db.as_retriever(
                search_type=search_type,
                search_kwargs={
                    "k": k,
                    "fetch_k": fetch_k,
                    "lambda_mult": lambda_mult
                }
            )

            logging.info(
                f"Retriever ready | "
                f"all_docs: {search_across_all} | "
                f"collection: {collection_name}"
            )
            return retriever

        except Exception as e:
            raise CustomException(e, sys)#type:ignore
        
    def list_collections(self) -> list:
        """
        List all document collections stored

        WHY THIS FUNCTION?
        frontend needs to show user which documents
        are already uploaded and ready to query
        this returns list of collection names 
        """
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_dir)
            collections = client.list_collections()
            names = [col.name for col in collections]
            logging.info(f"Collections found: {names}")
            return names

        except Exception as e:
            raise CustomException(e, sys)#type:ignore

    

        

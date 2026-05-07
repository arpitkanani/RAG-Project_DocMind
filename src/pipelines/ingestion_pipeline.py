import sys
import yaml
import os
import shutil
from src.components.document_loader import DocumentLoader
from src.components.text_splitter import TextSplitter
from src.components.vector_store import VectorStore
from src.utils.file_helper import (validate_file, validate_file_size, delete_file_after_processing,get_filename,save_uploaded_file)
from src.utils.youtube_helper import is_youtube_url,extract_video_id
from src.exception import CustomException
from src.logger import logging

with open("config/config.yaml") as f:
    config=yaml.safe_load(f)

class IngestionPipeline:
    """
    Connects all ingestion components into one clean pipeline.

    source (file path or YouTube URL)
        ↓
    DocumentLoader  → List[Document]
        ↓
    TextSplitter    → List[Document] (chunks)
        ↓
    VectorStore     → stored in ChromaDB

    """
    def __init__(self) :
        try:

            logging.info("Initializing Ingestion Pipeline components...")
            self.loader = DocumentLoader()
            self.splitter = TextSplitter()
            
            logging.info("Ingestion Pipeline ready.")

        except Exception as e:
            logging.error(f"Error initializing Ingestion Pipeline: {e}")
            raise CustomException(e, sys) # type: ignore
        
    def run(self, source:str,
            collection_name:str=None,
            clear_existing:bool = True) -> dict:
        """
        Run full ingestion pipeline on any source.

        dict gives all this info cleanly
        FastAPI can return this as JSON response

        """
        try:
            logging.info(f"starting Ingestion :{source}")

            if collection_name is None:
                collection_name = self._get_collection_name(source)
            logging.info(f"Collection: {collection_name}")

            docs = self.loader.load(source)
            logging.info(f"Loaded {len(docs)} documents")


            chunks = self.splitter.split(docs)
            logging.info(f"Created {len(chunks)} chunks")


            vs = VectorStore(collection_name=collection_name)

            if clear_existing:
                vs.delete_collection()
                logging.info("Existing collection cleared")

            vs.add_documents(chunks)

            if not is_youtube_url(source):
                if os.path.exists(source):
                    #delete_file_after_processing(source)
                    logging.info(f"Source file deleted: {source}")

            result = {
                "success":         True,
                "source":          source,
                "collection_name": collection_name,
                "documents_loaded": len(docs),
                "chunks_stored":   len(chunks),
                "is_youtube":      is_youtube_url(source)
            }

            logging.info(
                f"Ingestion complete | "
                f"chunks: {len(chunks)} | "
                f"collection: {collection_name}"
            )
            return result

        except Exception as e:
            raise CustomException(e, sys) # type: ignore
        
    def run_from_bytes(self, file_bytes: bytes,
                       filename: str) -> dict:
        
        """
        Run ingestion from raw file bytes.
        """
        try:
            logging.info(f"Ingesting from bytes: {filename}")

            # validate file before saving
            if not validate_file(filename):
                return {
                    "success": False,
                    "error":   f"File type not allowed: {filename}"
                }

            if not validate_file_size(file_bytes, filename):
                return {
                    "success": False,
                    "error":   f"File too large. Max {config['upload']['max_file_size_mb']}MB"
                }
            
            file_path = save_uploaded_file(file_bytes, filename)
            logging.info(f"File saved: {file_path}")

            # run normal pipeline on saved file
            result = self.run(file_path)
            return result

        except Exception as e:
            raise CustomException(e, sys)#type:ignore
        
    
    def _get_collection_name(self, source:str) -> str:
        """Generate a collection name based on the source."""
        try:
            if is_youtube_url(source):
                # extract video ID for collection name
                
                video_id = extract_video_id(source)
                return f"youtube_{video_id}"
            filename = os.path.basename(source)
            name     = os.path.splitext(filename)[0]

            name = name.replace(" ", "_").lower()

            logging.info(f"Collection name: {name}")
            return name
        except Exception as e:
            raise CustomException(e, sys) # type: ignore
        



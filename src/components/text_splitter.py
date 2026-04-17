import sys
import os
from dataclasses_json import config
from langchain_text_splitters import RecursiveCharacterTextSplitter
import yaml
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from src.logger import logging
from src.exception import CustomException
from typing import List

config_path="D:\Langchain Project\config\config.yaml"

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

class TextSplitter:
    """
    splits text into sementic Chunks 
    """
    def __init__(self):
        try:
            logging.info("Initializing TextSplitter")

            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                separators=["\n\n", "\n", " ", ""]
            )

            logging.info("TextSplitter initialized successfully")

        except Exception as e:
            raise CustomException(e, sys)#type:ignore
    
    def split(self, docs: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        try:
            logging.info(f"Splitting {len(docs)} documents")
            chunks = self.splitter.split_documents(docs)

            avg_size = sum(
                len(c.page_content) for c in chunks
            ) // len(chunks) if chunks else 0

            logging.info(
                f"Split complete: {len(chunks)} chunks | "
                f"avg size: {avg_size} chars"
            )
            return chunks
        
        except Exception as e:
            logging.error(f"Error in TextSplitter split: {e}")
            raise CustomException(e, sys)#type:ignore
        
    def split_text(self,text:str) -> List[str]:
        """
        splits plain string into sementic chunks 
        and returns list of strings 

        maybe not needed if we can directly split documents but
        just in case we have plain text in youtube transcripts contain text.

        """
        try:
            logging.info(f"splitting plain text into sementic Chunks")

            chunks=self.splitter.split_text(text)
            logging.info(f"Text split into {len(chunks)} chunks")

            return chunks
        except Exception as e:
            logging.error(f"Error in TextSplitter split_text: {e}")
            raise CustomException(e, sys)#type:ignore

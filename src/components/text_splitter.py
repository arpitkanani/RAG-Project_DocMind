import sys
import os
from dataclasses_json import config
import yaml
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from src.logger import logging
from src.exception import CustomException
from typing import List

config_path="D:\Langchain Project\config\config.yaml"

with open(config_path) as f:
    config = yaml.safe_load(f)

class TextSplitter:
    """
    splits text into sementic Chunks 
    """
    def __init__(self):
        try:
            logging.info("initalizing TextSplitter")
            self.embeddings=OllamaEmbeddings(model="nomic-embed-text") #type: ignore
            
            self.chunker=SemanticChunker(
                embeddings=self.embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=80,

            )

        except Exception as e:
            logging.error(f"Error in TextSplitter initialization: {e}")
            raise CustomException(e, sys)#type:ignore
    
    def split(self,docs:List[Document])->List[Document]:
        """
        splits text into sementic Chunks 
        """
        try:
            logging.info(f"splitting text into sementic Chunks")


            chunks=self.chunker.split_documents(docs)
            avg_size=sum(len(c.page_content) for c in chunks)//len(chunks) if chunks else 0
            logging.info(f"Text split into {len(chunks)} chunks with average size {avg_size} characters")

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

            chunks=self.chunker.split_text(text)

            return chunks
        except Exception as e:
            logging.error(f"Error in TextSplitter split_text: {e}")
            raise CustomException(e, sys)#type:ignore

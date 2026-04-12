import sys
import yaml
from langchain_ollama import OllamaEmbeddings
from src.logger import logging
from src.exception import CustomException

with open("D:\Langchain Project\config\config.yaml") as f:
    config = yaml.safe_load(f)

class Embedder:
    """
     Manages embedding model for the entire project
    """

    def __init__(self):
        try:
            logging.info("Initializing Embedder")

            self.embedding_model = OllamaEmbeddings(
                model=config["embedding"]["embed_model"]

            )
            logging.info(
                f"Embedder ready: {config['embedding']['embed_model']}"
            )

        except Exception as e:
            raise CustomException(e, sys)#type:ignore
        
    def get_embedding(self)-> OllamaEmbeddings:
        """
        Return embedding model object

        """
        return self.embedding_model
    
    def embed_text(self,text:str) -> list:

        try:
            logging.info(f"Embedding single text: {text[:50]}...")
            
            vector = self.embedding_model.embed_query(text)

            logging.info(
                f"Embedding complete: {len(vector)} dimensions"
            )
            return vector
        except Exception as e:
            raise CustomException(e, sys)#type:ignore
        
    def embed_documents(self, texts: list) -> list:
        try:

            logging.info(f"Embedding {len(texts)} documents")
            # embed_documents → designed for batch processing
            vectors = self.embedding_model.embed_documents(texts)

            logging.info(
                f"Batch embedding complete: "
                f"{len(vectors)} vectors of "
                f"{len(vectors[0])} dimensions each"
            )
            return vectors

        except Exception as e:
            raise CustomException(e, sys)#type:ignore
        
        

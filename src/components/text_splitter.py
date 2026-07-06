import sys
from typing import List

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.exception import CustomException
from src.logger import logging

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)


class TextSplitter:
    """Split loaded documents into retrieval-sized chunks."""

    def __init__(self):
        try:
            logging.info("Initializing TextSplitter")
            splitter_config = config.get("splitter", {})
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=splitter_config.get("chunk_size", 600),
                chunk_overlap=splitter_config.get("chunk_overlap", 100),
                separators=splitter_config.get(
                    "separators",
                    ["\n\n", "\n", " ", ""],
                ),
            )
            logging.info("TextSplitter initialized successfully")
        except Exception as e:
            raise CustomException(e, sys)

    def split(self, docs: List[Document]) -> List[Document]:
        """Split document objects into chunks."""
        try:
            logging.info("Splitting %s documents", len(docs))
            chunks = self.splitter.split_documents(docs)
            avg_size = (
                sum(len(chunk.page_content) for chunk in chunks) // len(chunks)
                if chunks
                else 0
            )
            logging.info(
                "Split complete | chunks: %s | avg_size: %s chars",
                len(chunks),
                avg_size,
            )
            return chunks
        except Exception as e:
            logging.error("Error in TextSplitter.split: %s", e)
            raise CustomException(e, sys)

    def split_text(self, text: str) -> List[str]:
        """Split raw text into chunks."""
        try:
            logging.info("Splitting plain text into chunks")
            chunks = self.splitter.split_text(text)
            logging.info("Text split into %s chunks", len(chunks))
            return chunks
        except Exception as e:
            logging.error("Error in TextSplitter.split_text: %s", e)
            raise CustomException(e, sys)

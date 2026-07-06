import sys
from typing import List

import pandas as pd
import yaml
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document

from src.exception import CustomException
from src.logger import logging
from src.utils.file_helper import get_file_extension
from src.utils.youtube_helper import get_transcript_segments, is_youtube_url

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)


class DocumentLoader:
    """Load supported files or YouTube transcripts as LangChain documents."""

    def load(self, source: str) -> List[Document]:
        try:
            logging.info("Loading document from source: %s", source)

            if is_youtube_url(source):
                return self._load_youtube(source)

            extension = get_file_extension(source)
            loader_map = {
                ".pdf": self._load_pdf,
                ".txt": self._load_txt,
                ".docx": self._load_docx,
                ".csv": self._load_csv,
                ".md": self._load_md,
                ".xlsx": self._load_xlsx,
            }

            if extension not in loader_map:
                raise ValueError(f"Unsupported file type: {extension}")

            docs = loader_map[extension](source)
            logging.info("Loaded %d documents from %s", len(docs), source)
            return docs
        except Exception as e:
            raise CustomException(e, sys)

    def _load_pdf(self, path: str) -> List[Document]:
        try:
            docs = PyPDFLoader(path).load()
            logging.info("PDF loaded: %d pages", len(docs))
            return docs
        except Exception as e:
            raise CustomException(e, sys)

    def _load_txt(self, path: str) -> List[Document]:
        try:
            for encoding in ("utf-8", "cp1252", "latin-1"):
                try:
                    docs = TextLoader(
                        path,
                        encoding=encoding,
                        autodetect_encoding=True,
                    ).load()
                    logging.info("TXT loaded (%s): %d doc", encoding, len(docs))
                    return docs
                except Exception:
                    logging.warning("Encoding %s failed for %s", encoding, path)

            with open(path, "r", encoding="utf-8", errors="ignore") as file_obj:
                text = file_obj.read()

            return [Document(page_content=text, metadata={"source": path})]
        except Exception as e:
            raise CustomException(e, sys)

    def _load_docx(self, path: str) -> List[Document]:
        try:
            return Docx2txtLoader(path).load()
        except Exception as e:
            raise CustomException(e, sys)

    def _load_csv(self, path: str) -> List[Document]:
        try:
            return CSVLoader(path, encoding="utf-8").load()
        except Exception as e:
            raise CustomException(e, sys)

    def _load_md(self, path: str) -> List[Document]:
        try:
            return UnstructuredMarkdownLoader(path).load()
        except Exception as e:
            raise CustomException(e, sys)

    def _load_xlsx(self, path: str) -> List[Document]:
        try:
            dataframe = pd.read_excel(path)
            documents: List[Document] = []
            for index, row in dataframe.iterrows():
                row_text = "\n".join(f"{column}: {value}" for column, value in row.items())
                documents.append(
                    Document(
                        page_content=row_text,
                        metadata={"source": path, "row": index},
                    )
                )
            logging.info("XLSX loaded: %d row documents", len(documents))
            return documents
        except Exception as e:
            raise CustomException(e, sys)

    def _load_youtube(self, url: str) -> List[Document]:
        try:
            segments = get_transcript_segments(url)
            docs = [
                Document(
                    page_content=str(segment["text"]),
                    metadata={
                        "source": "YouTube transcript",
                        "video_url": url,
                        "type": "youtube",
                        "timestamp": str(segment["timestamp"]),
                        "start_seconds": float(segment["start"]),
                        "duration_seconds": float(segment["duration"]),
                    },
                )
                for segment in segments
            ]
            logging.info("YouTube transcript loaded: %d segments", len(docs))
            return docs
        except Exception as e:
            raise CustomException(e, sys)

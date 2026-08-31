import sys
from typing import List, Iterator
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

    def load(self, source: str) -> Iterator[Document]:
        try:
            logging.info("Loading document from source: %s", source)

            if is_youtube_url(source):
                yield from self._load_youtube(source)
                return

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

            docs_iterator = loader_map[extension](source)
            logging.info("Started streaming documents from %s", source)
            yield from docs_iterator

        except Exception as e:
            raise CustomException(e, sys)

    def _load_pdf(self, path: str) -> Iterator[Document]:
        try:
            # First try PyPDFLoader.load() to guarantee page reading across all pypdf versions
            loader = PyPDFLoader(path)
            if hasattr(loader, "lazy_load"):
                try:
                    pages = list(loader.lazy_load())
                    if pages:
                        for page in pages:
                            yield page
                        logging.info("PDF stream completed: %d pages", len(pages))
                        return
                except Exception as lazy_err:
                    logging.warning("PyPDFLoader lazy_load failed, falling back to load(): %s", lazy_err)

            docs = loader.load()
            logging.info("PDF loaded: %d pages", len(docs))
            for doc in docs:
                yield doc
        except Exception as e:
            raise CustomException(e, sys)

    def _load_txt(self, path: str) -> Iterator[Document]:
        try:
            for encoding in ("utf-8", "cp1252", "latin-1"):
                try:
                    docs_iter = TextLoader(
                        path,
                        encoding=encoding,
                        autodetect_encoding=True,
                    ).lazy_load()
                    yield from docs_iter
                    logging.info("TXT stream completed (%s)", encoding)
                    return

                except Exception:
                    logging.warning("Encoding %s failed for %s", encoding, path)

            with open(path, "r", encoding="utf-8", errors="ignore") as file_obj:
                text = file_obj.read()

            yield Document(page_content=text, metadata={"source": path})
        except Exception as e:
            raise CustomException(e, sys)

    def _load_docx(self, path: str) -> Iterator[Document]:
        try:
            yield from Docx2txtLoader(path).lazy_load()
        except Exception as e:
            raise CustomException(e, sys)

    def _load_csv(self, path: str) -> Iterator[Document]:
        try:
            yield from CSVLoader(path, encoding="utf-8").lazy_load()
        except Exception as e:
            raise CustomException(e, sys)

    def _load_md(self, path: str) -> Iterator[Document]:
        try:
            yield from UnstructuredMarkdownLoader(path).lazy_load()
        except Exception as e:
            raise CustomException(e, sys)

    def _load_xlsx(self, path: str) -> Iterator[Document]:
        try:
            dataframe = pd.read_excel(path)
            for index, row in dataframe.iterrows():
                row_text = "\n".join(f"{column}: {value}" for column, value in row.items())
                yield Document(
                    page_content=row_text,
                    metadata={"source": path, "row": index},
                )
            logging.info("XLSX stream completed")
        except Exception as e:
            raise CustomException(e, sys)

    def _load_youtube(self, url: str) -> Iterator[Document]:
        try:
            segments = get_transcript_segments(url)
            target_size = config.get("splitter", {}).get("chunk_size", 600)
            windows = self._merge_transcript_segments(segments, target_size)

            for window in windows:
                yield Document(
                    page_content=window["text"],
                    metadata={
                        "source": "YouTube transcript",
                        "video_url": url,
                        "type": "youtube",
                        "timestamp": window["timestamp_range"],
                        "start_seconds": window["start_seconds"],
                        "duration_seconds": window["duration_seconds"],
                    },
                )
            logging.info(
                "YouTube transcript stream completed: %d segments merged into %d chunks",
                len(segments),
                len(windows),
            )

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        total = int(seconds)
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"

    @classmethod
    def _merge_transcript_segments(cls, segments: list, target_size: int) -> List[dict]:
        """
        Group consecutive short caption segments into ~target_size character
        windows, so each retrieval unit is a coherent passage (matching the
        size of chunks from every other document type) instead of a single
        5-15 word caption fragment. Each window keeps a timestamp *range*
        covering everything merged into it.
        """
        windows: List[dict] = []
        current_texts: List[str] = []
        current_len = 0
        window_start = None
        window_end = None

        def flush():
            if not current_texts:
                return
            windows.append(
                {
                    "text": " ".join(current_texts).strip(),
                    "start_seconds": window_start,
                    "duration_seconds": max(window_end - window_start, 0.0),
                    "timestamp_range": (
                        f"{cls._format_seconds(window_start)}-{cls._format_seconds(window_end)}"
                    ),
                }
            )

        for segment in segments:
            text = str(segment["text"]).strip()
            if not text:
                continue

            start = float(segment["start"])
            duration = float(segment["duration"])
            end = start + duration

            if window_start is None:
                window_start = start

            if current_len + len(text) > target_size and current_texts:
                flush()
                current_texts, current_len = [], 0
                window_start = start

            current_texts.append(text)
            current_len += len(text) + 1
            window_end = end

        flush()
        return windows
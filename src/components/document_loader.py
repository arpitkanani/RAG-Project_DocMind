import pandas as pd
import sys
import os
from typer.cli import docs
import yaml
from langchain_community.document_loaders import PyPDFLoader,TextLoader,Docx2txtLoader,UnstructuredMarkdownLoader,CSVLoader
from langchain_core.documents import Document
from src.exception import CustomException
from src.logger import logging
from src.utils.file_helper import get_file_extension
from src.utils.youtube_helper import get_transcript,is_youtube_url
from typing import List

with open("config/config.yaml") as f:
    config=yaml.safe_load(f)

class DocumentLoader:
    """
    Handles loading of all supported document types:
    - PDF   → PyPDFLoader
    - TXT   → TextLoader
    - DOCX  → Docx2txtLoader
    - CSV   → CSVLoader
    - MD    → UnstructuredMarkdownLoader
    - XLSX  → pandas based loading
    - YouTube URL → transcript fetching
    
    Single entry point → .load(source)
    Detects type automatically → calls correct loader
    Always returns List[Document] regardless of source type
    """

    def load(self,source:str)->List[Document]:
        """
        Main entry point for all document loading
        
        source = file path OR youtube URL
        
        flow:
            is youtube URL? → fetch transcript
            is file?        → detect extension → use correct loader
        
        always returns List[Document] ← this is critical
        rest of pipeline doesn't care what source was
        it just receives List[Document] and processes it
        """
        try:
            logging.info(f"loading document from source: {source}")

            if is_youtube_url(source):
                return self._load_youtube(source)
            
            ext=get_file_extension(source)
            
            loader_map={
                ".pdf":self._load_pdf,
                ".txt":self._load_txt,
                ".docx":self._load_docx,
                ".csv":self._load_csv,
                ".md":self._load_md,
                ".xlsx":self._load_xlsx
            }

            if ext not in loader_map:
                raise ValueError(f"Unsupported file type: {ext}")
            
            docs=loader_map[ext](source)
            logging.info(
                f"Loaded {len(docs)} documents from {source}"
            )
            return docs
        except Exception as e:
            raise CustomException(e, sys) # type: ignore
        
    def _load_pdf(self,path:str) -> List[Document]:
        """
        Load PDF file
        PyPDFLoader splits by page automatically
        each page = one Document

        metadata added automatically:
        {"source": "file.pdf", "page": 0}
        page number used later for citations
        """
        try:
            loader=PyPDFLoader(path)
            logging.info(f"Pdf Loaded: {loader.load()}")
            return loader.load()
        except Exception as e:
            raise CustomException(e, sys) # type: ignore

    def _load_txt(self,path:str) -> List[Document]:
        """
        Load TXT file
        TextLoader treats whole file as one Document
        no splitting by line or paragraph

        metadata:
        {"source": "file.txt"}
        """
        try:
            loader=TextLoader(path)
            logging.info(f"Txt Loaded: {loader.load()}")
            return loader.load()
        except Exception as e:
            raise CustomException(e, sys) # type: ignore
        
    def _load_docx(self,path:str) -> List[Document]:
        """
        Load DOCX file
        Docx2txtLoader treats whole file as one Document
        no splitting by line or paragraph

        metadata:
        {"source": "file.docx"}
        """
        try:
            loader=Docx2txtLoader(path)
            logging.info(f"Docx Loaded: {loader.load()}")
            return loader.load()
        except Exception as e:
            raise CustomException(e, sys) # type: ignore
        
    def _load_csv(self,path:str) -> List[Document]:
        """
        Load CSV file
        CSVLoader treats whole file as one Document
        no splitting by line or row

        metadata:
        {"source": "file.csv"}
        """
        try:
            loader=CSVLoader(path,encoding="utf-8")
            logging.info(f"Csv Loaded: {loader.load()}")
            return loader.load()
        except Exception as e:
            raise CustomException(e, sys) # type: ignore
        
    def _load_md(self,path:str) -> List[Document]:
        """
        Load Markdown file
        UnstructuredMarkdownLoader treats whole file as one Document
        no splitting by line or paragraph

        metadata:
        {"source": "file.md"}
        """
        try:
            loader=UnstructuredMarkdownLoader(path)
            logging.info(f"Md Loaded: {loader.load()}")
            return loader.load()
        except Exception as e:
            raise CustomException(e, sys) # type: ignore
    
    def _load_xlsx(self,path:str) -> List[Document]:
        """ 
        Load Excel file using pandas
        No direct LangChain loader for xlsx
        so we use pandas to read it manually
        then convert each row to Document ourselves
        metadata:
        {"source": "file.xlsx", "row": 0}
        """
        try:
            df=pd.read_excel(path)
            documents=[]
            for index,row in df.iterrows():
                row_text = "\n".join(
                    f"{col}: {val}"
                    for col, val in row.items()
                )
                doc = Document(
                    page_content=row_text,
                    metadata={
                        "source": path,
                        "row": index  # row number for reference
                    }
                )
                documents.append(doc) # type: ignore
            logging.info(f"Xlsx Loaded: {len(documents)} documents created from {path}")
            return documents
        except Exception as e:
            raise CustomException(e, sys) # type: ignore
        
    def _load_youtube(self,url:str) -> List[Document]:
        """
        Fetch YouTube transcript and wrap as Document
        
        transcript = full video text as one long string
        we wrap it in a single Document with URL as source
        
        metadata stores:
            source → YouTube URL (used for citation)
            type   → "youtube" (so we know where it came from)
        """
        try:
            transcript=get_transcript(url)
            doc=Document(
                page_content=transcript,
                metadata={
                    "source": url,
                    "type": "youtube"
                }
            )
            logging.info(f"YouTube Loaded: Transcript fetched for {url}")
            return [doc] # return as list of one Document
        except Exception as e:
            raise CustomException(e,sys)#type:ignore
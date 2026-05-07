import sys
import yaml
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from src.components.memory_manager import MemoryManager
from src.components.vector_store import VectorStore
from src.utils.intent_detector import detect_summary_type
from typing import List
from src.logger import  logging
from src.exception import CustomException

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)


SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are DocMind, a document summarization assistant.
Summarize ONLY from the provided document context.
Never add information from outside the document.
Write in same language as the document content.
If multiple questions asked, answer each one fully."""),

    ("human", """Document context:
{context}

Summary type requested: {summary_type}

Generate summary based on type:
- If "bullets": provide bullet points covering ALL key points
  Format: • point1 • point2 • point3...
- If "structured": write detailed flowing paragraphs covering
  main topic, key concepts, findings, examples
- If "brief": write only 2-3 sentences covering main point

{summary_type} Summary:""")
])

# MAP — stays separate, different job
MAP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Extract and summarize only key information
from this text chunk. Be concise but preserve all important facts."""),

    ("human", """Text chunk:
{chunk}

Key points from this chunk:""")
])

# REDUCE — stays separate, different inputs
REDUCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are DocMind, a document summarization assistant.
Combine partial summaries into one complete summary.
Do not add information not present in partial summaries.
Write in same language as content.
If multiple questions asked, answer each one fully."""),

    ("human", """Partial summaries:
{partial_summaries}

Combine into one complete {summary_type} summary:""")
])

class SummaryChain:
    """ Handles all summaeization tasks using different prompt templates based on config settings. """

    def __init__(self):
        """
        Initialize LLM, parser and all summary chains.
        Built once, reused for every summarize() call.
        """
        try:
            logging.info("Initializing SummaryChain.")
            self.llm=ChatOllama(
                model=config["llm"]["model"],
                temperature=config["llm"]["temperature"]
            )
            self.parser=StrOutputParser()

            self.direct_threshold=config["summary"]["max_chunk_size_for_direct"]

            self.summary_chain=SUMMARY_PROMPT |self.llm | self.parser
            self.map_chain=MAP_PROMPT | self.llm | self.parser
            self.reduce_chain=REDUCE_PROMPT | self.llm | self.parser

            logging.info("SummaryChain initialized successfully")

        except Exception as e:
            raise CustomException(e, sys)
        
    def _map_step(self,chunks: List[Document]) -> List[str]:
        """MAP — summarize each chunk individually.
        skips empty chunks automatically.
        """  
        try:
            logging.info(f"MAP step: {len(chunks)} chunks")

            mini_summaries=[]

            for i , chunk in enumerate(chunks):
                if not chunk.page_content.strip():
                    continue

                mini=self.map_chain.invoke({"chunk":chunk.page_content})

                mini_summaries.append(mini)
            
            logging.info(f"MAP done: {len(mini_summaries)} summaries")
            return mini_summaries
        except Exception as e:
            raise CustomException(e, sys)
        
    def _reduce_step(self,mini_summaries:List[str],summary_type:str)->str:
        """REDUCE — combine all mini summaries into final.

        summary_type passed so LLM knows
        what format to produce at the end.
        """
        try:
            logging.info(
                f"REDUCE: summaries → {summary_type}"
            )
            combined="\n\n---\n\n".join(mini_summaries)

            result=self.reduce_chain.invoke({
                "partial_summaries": combined,
                "summary_type": summary_type
            })

            return result
        except Exception as e:
            raise CustomException(e, sys)
        
    def summarize(self,summary_type:str="bullets",collection_name:str=None) -> str:
        """
        Main entry point for all summarization.

        collection_name=None  → summarizes ALL uploaded documents
        collection_name="x"   → summarizes only that collection

        small doc → sends all chunks at once to LLM
        large doc → map each chunk → reduce into final
        """
        try:
            vs = VectorStore(collection_name=collection_name)
            chunks = vs.get_all_documents()

            if not chunks:
                return "No document found. Please upload a document first."
            
            if len(chunks)<=self.direct_threshold:
                logging.info("Strategy: direct")
                context="\n\n---\n\n".join(
                    chunk.page_content
                    for chunk in chunks
                    if chunk.page_content.strip()
                )

                return self.summary_chain.invoke({
                    "context": context,
                    "summary_type": summary_type
                })
            else:
                logging.info("Strategy: map-reduce")
                mini_summaries = self._map_step(chunks)
                return self._reduce_step(mini_summaries, summary_type)
            
        except Exception as e:
            raise CustomException(e, sys)
        
def get_summary(query:str,collection_name:str=None)->str:
    """
    Entry point called by pipeline and FastAPI.
    Detects summary type from query then generates.
    """

    try:
                                   
        summary_type=detect_summary_type(query)
        chain=SummaryChain()

        result=chain.summarize(
            summary_type=summary_type,
            collection_name=collection_name
        )

        memory = MemoryManager()
        memory.save_message("human", query)
        memory.save_message("ai", result)

        logging.info("Summary returned successfully")
        return result
    except Exception as e:
        raise CustomException(e, sys)
    
    
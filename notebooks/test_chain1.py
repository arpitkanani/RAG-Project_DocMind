import sys
import os 

sys.path.append(os.getcwd())

from src.components.document_loader import DocumentLoader
from src.components.text_splitter import TextSplitter
from src.components.vector_store import VectorStore
from src.components.memory_manager import MemoryManager
from src.chains.qa_chain import get_answer


loader=DocumentLoader()

splitter=TextSplitter()
vs=VectorStore()
docs=loader.load("test.txt")

chunks=splitter.split(docs)

vs.store(chunks)

print(f"Document loaded and chunks store in VectorDB {len(chunks)}")

MemoryManager().clear()

answer1=get_answer("what is machine learning? ") # type: ignore
print(f"answer: {answer1}")


memory = MemoryManager()
print(f"Messages in memory: {memory.get_message_count()}")
print(f"\nConversation history:")
print(memory.get_history_as_text())



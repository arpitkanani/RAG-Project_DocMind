import sys
import os
sys.path.append(os.getcwd())

from src.components.document_loader import DocumentLoader
from src.components.text_splitter import TextSplitter
from src.components.vector_store import VectorStore
from src.chains.quiz_chain import get_quiz


loader   = DocumentLoader()
splitter = TextSplitter()
vs       = VectorStore()

docs   = loader.load("test.txt")
chunks = splitter.split(docs)

vs.delete_collection()
vs.add_documents(chunks)
print(f"{len(chunks)} chunks stored\n")




result2 = get_quiz("generate 5 questions with long answers like 5 line of answer")
print(result2)





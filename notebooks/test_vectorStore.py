import sys
import os
sys.path.append(os.getcwd())

from src.components.document_loader import DocumentLoader
from src.components.text_splitter import TextSplitter
from src.components.vector_store import VectorStore

loader   = DocumentLoader()
splitter = TextSplitter()
vs       = VectorStore()

print("TEST 1 — STORE CHUNKS")


docs   = loader.load("test.txt")
chunks = splitter.split(docs)

print(f"Chunks to store: {len(chunks)}")

vs.clear()  # fresh start
db = vs.store(chunks)

print(f"Stored successfully!")
print(f"Vectors in DB: {db._collection.count()}")


print(f"VectorStore exists: {vs.exists()}")


print("TEST 3 — LOAD AND SEARCH")


retriever = vs.get_retriever()
results = retriever.invoke("what is machine learning?")

print(f"Query: 'what is machine learning?'")
print(f"Results returned: {len(results)}")
for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(f"Content  : {doc.page_content}")
    print(f"Metadata : {doc.metadata}")


print("ALL VECTORSTORE TESTS COMPLETE!")

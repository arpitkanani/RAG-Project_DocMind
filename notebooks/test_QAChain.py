import sys
import os
sys.path.append(os.getcwd())

from src.components.document_loader import DocumentLoader
from src.components.text_splitter import TextSplitter
from src.components.vector_store import VectorStore
from src.components.retriever import Retriever
from src.components.memory_manager import MemoryManager
from src.chains.qa_chain import get_answer



loader   = DocumentLoader()
splitter = TextSplitter()
vs       = VectorStore()
memory   = MemoryManager()

docs = loader.load("test.txt")
print(f"Documents loaded    : {len(docs)}")
print(f"   Type               : {type(docs[0]).__name__}")
print(f"   Content preview    : {docs[0].page_content[:800]}")
print(f"   Metadata           : {docs[0].metadata}")


chunks = splitter.split(docs)
print(f"Chunks created      : {len(chunks)}")
print(f"   Avg chunk size     : {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
print(f"   First chunk preview: {chunks[0].page_content[:800]}")
print(f"   First chunk meta   : {chunks[0].metadata}")



vs.delete_collection()
db = vs.add_documents(chunks)
print(f"    Chunks stored       : {db._collection.count()}")
print(f"   VectorStore exists : {vs.exists()}")
print(f"   Persist directory  : {vs.persist_dir}")


retriever = Retriever()
test_query = "what is machine learning?"

retrieved_docs = retriever.retrieve(test_query)
print(f"    Query               : '{test_query}'")
print(f"   Chunks retrieved   : {len(retrieved_docs)}")

for i, doc in enumerate(retrieved_docs):
    print(f"\n\n   Result {i+1}:")
    print(f"   Content  : {doc.page_content[0:]}")
    print(f"   Metadata : {doc.metadata}")

# retrieve with scores
print(f"\n  Retrieval with scores:")
scored = retriever.retrieve_with_scores(test_query)
for doc, score in scored:
    print(f"   Score: {score:.4f} | {doc.page_content[0:]}")




memory.clear()
print(f"Memory cleared")
print(f"Messages in memory : {memory.get_message_count()}")

print("-" * 40)

questions = [
    "What is machine learning?",
    "What is Python used for in data science?",
]

for i, question in enumerate(questions):
    print(f"\n   Q{i+1}: {question}")
    answer = get_answer(question)
    print(f"   A{i+1}: {answer}")
    print(f"   {'-' * 50}")



followup = "Can you give me an example of that?"
followup_answer = get_answer(followup)
print(f"\n   Follow-up question: '{followup}'")
print(f"   Answer: {followup_answer}")


out_of_context = "What is the capital of France?"
print(f"   Question: '{out_of_context}'")


oc_answer = get_answer(out_of_context)
print(f"   Answer: {oc_answer}")

print(f"Messages saved      : {memory.get_message_count()}")
print(f"\n   Conversation history:")
print(memory.get_history_as_text())


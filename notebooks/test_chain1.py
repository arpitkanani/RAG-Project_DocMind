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

print("\nSTEP 1 — Document Loading")
print("-" * 40)

docs = loader.load("test.txt")
print(f"Documents loaded    : {len(docs)}")
print(f"   Type               : {type(docs[0]).__name__}")
print(f"   Content preview    : {docs[0].page_content[:100]}")
print(f"   Metadata           : {docs[0].metadata}")

print("\nSTEP 2 — Text Splitting")
print("-" * 40)

chunks = splitter.split(docs)
print(f"Chunks created      : {len(chunks)}")
print(f"   Avg chunk size     : {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
print(f"   First chunk preview: {chunks[0].page_content[:100]}")
print(f"   First chunk meta   : {chunks[0].metadata}")


print("\nSTEP 3 — Vector Store")
print("-" * 40)

vs.delete_collection()
db = vs.add_documents(chunks)
print(f"Chunks stored       : {db._collection.count()}")
print(f"   VectorStore exists : {vs.exists()}")
print(f"   Persist directory  : {vs.persist_dir}")

print("\nSTEP 4 — Retriever")
print("-" * 40)

retriever = Retriever()
test_query = "what is machine learning?"

retrieved_docs = retriever.retrieve(test_query)
print(f"Query               : '{test_query}'")
print(f"   Chunks retrieved   : {len(retrieved_docs)}")

for i, doc in enumerate(retrieved_docs):
    print(f"\n   Result {i+1}:")
    print(f"   Content  : {doc.page_content[:100]}")
    print(f"   Metadata : {doc.metadata}")

# retrieve with scores
print(f"\n  Retrieval with scores:")
scored = retriever.retrieve_with_scores(test_query)
for doc, score in scored:
    print(f"   Score: {score:.4f} | {doc.page_content[:80]}")

print("\nSTEP 5 — Memory")


memory.clear()
print(f"Memory cleared")
print(f"   Messages in memory : {memory.get_message_count()}")

print("\nSTEP 6 — QA Chain")
print("-" * 40)

questions = [
    "What is machine learning?",
    "What are the types of machine learning?",
    "What is Python used for in data science?",
]

for i, question in enumerate(questions):
    print(f"\n   Q{i+1}: {question}")
    answer = get_answer(question)
    print(f"   A{i+1}: {answer}")
    print(f"   {'-' * 50}")

print("\nSTEP 7 — Memory After QA")
print("-" * 40)

print(f"Messages saved      : {memory.get_message_count()}")
print(f"\n   Conversation history:")
print(memory.get_history_as_text())

print("\nSTEP 8 — Follow Up Question (Memory Test)")
print("-" * 40)

followup = "Can you give me an example of that?"
print(f"   Follow up: '{followup}'")
print(f"   (LLM should know 'that' refers to previous answer)")

followup_answer = get_answer(followup)
print(f"   Answer: {followup_answer}")

print("\nSTEP 9 — Out of Context Question")
print("-" * 40)

out_of_context = "What is the capital of France?"
print(f"   Question: '{out_of_context}'")
print(f"   (Should say: not available in document)")

oc_answer = get_answer(out_of_context)
print(f"   Answer: {oc_answer}")




print(f"DocumentLoader  → working")
print(f"TextSplitter    → working")
print(f"VectorStore     → working")
print(f"Retriever       → working")
print(f"MemoryManager   → working")
print(f"QA Chain        → working")
print(f"Full RAG Pipeline → working")
print("=" * 60)
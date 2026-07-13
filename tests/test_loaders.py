import sys
import os

# this line tells Python where to find src/
# because we're running from root directory
sys.path.append(os.getcwd())

from src.components.document_loader import DocumentLoader

loader = DocumentLoader()
print("TEST 1 — TXT LOADER")

txt_docs = loader.load("test.txt")
print(f"Documents loaded: {len(txt_docs)}")
print(f"Type: {type(txt_docs[0])}")
print(f"Content: {txt_docs[0].page_content}")
print(f"Metadata: {txt_docs[0].metadata}")

print("TEST 2 — PDF LOADER")

pdf_docs = loader.load("sample.pdf")
print(f"Pages loaded: {len(pdf_docs)}")
print(f"Page 1 preview: {pdf_docs[0].page_content[:200]}")
print(f"Metadata: {pdf_docs[0].metadata}")

print("TEST 3 — CSV LOADER")

csv_docs = loader.load("test.csv")
print(f"Rows loaded: {len(csv_docs)}")
for i, doc in enumerate(csv_docs):
    print(f"\nRow {i+1}:")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")

print("TEST 4 — YOUTUBE LOADER")

yt_docs = loader.load("https://youtu.be/hIJKZZfB9k8?si=n1MQoGjIco24hrnc")
print(f"Documents: {len(yt_docs)}")
print(f"Transcript preview: {yt_docs[0].page_content[:300]}")
print(f"Metadata: {yt_docs[0].metadata}")

print("\nALL TESTS PASSED!")
import sys
import os

# this line tells Python where to find src/
# because we're running from root directory
sys.path.append(os.getcwd())

from src.components.document_loader import DocumentLoader

loader = DocumentLoader()

# ─────────────────────────────────────────
# TEST 1 — TXT
# ─────────────────────────────────────────
print("=" * 50)
print("TEST 1 — TXT LOADER")
print("=" * 50)
txt_docs = loader.load("test.txt")
print(f"Documents loaded: {len(txt_docs)}")
print(f"Type: {type(txt_docs[0])}")
print(f"Content: {txt_docs[0].page_content}")
print(f"Metadata: {txt_docs[0].metadata}")

# ─────────────────────────────────────────
# TEST 2 — PDF
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("TEST 2 — PDF LOADER")
print("=" * 50)
pdf_docs = loader.load("sample.pdf")
print(f"Pages loaded: {len(pdf_docs)}")
print(f"Page 1 preview: {pdf_docs[0].page_content[:200]}")
print(f"Metadata: {pdf_docs[0].metadata}")

# ─────────────────────────────────────────
# TEST 3 — CSV
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("TEST 3 — CSV LOADER")
print("=" * 50)
csv_docs = loader.load("test.csv")
print(f"Rows loaded: {len(csv_docs)}")
for i, doc in enumerate(csv_docs):
    print(f"\nRow {i+1}:")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")

# ─────────────────────────────────────────
# TEST 4 — YouTube
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("TEST 4 — YOUTUBE LOADER")
print("=" * 50)
yt_docs = loader.load("https://youtu.be/hIJKZZfB9k8?si=n1MQoGjIco24hrnc")
print(f"Documents: {len(yt_docs)}")
print(f"Transcript preview: {yt_docs[0].page_content[:300]}")
print(f"Metadata: {yt_docs[0].metadata}")

print("\nALL TESTS PASSED!")
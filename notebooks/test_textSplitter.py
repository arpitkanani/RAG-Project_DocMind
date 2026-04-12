import sys
import os
sys.path.append(os.getcwd())

from src.components.document_loader import DocumentLoader
from src.components.text_splitter import TextSplitter

loader=DocumentLoader()
splitter=TextSplitter()


print("\n" + "--" * 50)
print("TEST 2 — TEXT SPLITTING")
print("--" * 50)
txt_docs=loader.load("test.txt")

txt_chunks=splitter.split(txt_docs)
for i,chunk in enumerate(txt_chunks):
    print(f"\n--- Chunk {i+1} ---")
    print(f"Content  : {chunk.page_content}")
    print(f"Metadata : {chunk.metadata}")
    print(f"Size     : {len(chunk.page_content)} chars")


print("\n" + "--" * 50)
print("TEST 2 — PDF SPLITTING")
print("--" * 50)

pdf_docs = loader.load("sample.pdf")

total_chars_before = sum(len(d.page_content) for d in pdf_docs)
print(f"Total chars before: {total_chars_before}")
pdf_chunks = splitter.split(pdf_docs)
print(f"\nAfter split: {len(pdf_chunks)} chunks")
total_chars_after = sum(len(c.page_content) for c in pdf_chunks)
avg_size = total_chars_after // len(pdf_chunks)
print(f"Total chars after : {total_chars_after}")
print(f"Average chunk size: {avg_size} chars")

print("\nFirst 3 chunks preview:")
for i, chunk in enumerate(pdf_chunks[:3]):
    print(f"\n--- Chunk {i+1} ---")
    print(f"Content  : {chunk.page_content[:200]}")
    print(f"Metadata : {chunk.metadata}")
    print(f"Size     : {len(chunk.page_content)} chars")

print("\n" + "-" * 50)
print("TEST 3 — METADATA PRESERVATION CHECK")
print("-" * 50)
print("Checking all chunks have metadata...")

missing_metadata = [
    i for i, c in enumerate(pdf_chunks)
    if not c.metadata
]
if missing_metadata:
    print(f"Chunks missing metadata: {missing_metadata}")
else:
    print("All chunks have metadata preserved!")


print("ALL SPLITTER TESTS COMPLETE!")
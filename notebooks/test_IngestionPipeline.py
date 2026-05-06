import sys
import os
sys.path.append(os.getcwd())

from src.pipelines.ingestion_pipeline import IngestionPipeline

pipeline = IngestionPipeline()

# result = pipeline.run("test.pdf")

# print(f"Success         : {result['success']}")
# print(f"Collection      : {result['collection_name']}")
# print(f"Docs loaded     : {result['documents_loaded']}")
# print(f"Chunks stored   : {result['chunks_stored']}")
# print(f"Is YouTube      : {result['is_youtube']}")

# print("\n\nTEST 2 — PDF INGESTION")
# print("-" * 50)

# result2 = pipeline.run("sample.pdf")
# print(f"Success         : {result2['success']}")
# print(f"Collection      : {result2['collection_name']}")
# print(f"Docs loaded     : {result2['documents_loaded']}")
# print(f"Chunks stored   : {result2['chunks_stored']}")


print("TEST 3 — YOUTUBE INGESTION")
print("-" * 50)

yt_url  = "https://youtu.be/NMEJdVVfqak?si=XhEF52gxExWYykx_"
result3 = pipeline.run(yt_url)
print(f"Success         : {result3['success']}")
print(f"Collection      : {result3['collection_name']}")
print(f"Chunks stored   : {result3['chunks_stored']}")
print(f"Is YouTube      : {result3['is_youtube']}")

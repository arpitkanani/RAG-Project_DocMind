import os
import sys

sys.path.append(os.getcwd())

from src.components.document_loader import DocumentLoader
from src.components.text_splitter import TextSplitter
from src.components.vector_store import VectorStore
from src.chains.summary_chain import get_summary

loader=DocumentLoader()
splitter=TextSplitter()
vs=VectorStore()

# docs=loader.load("test.txt")
# chunks=splitter.split(docs)

# vs.delete_collection()

# vs.add_documents(chunks)



# print(get_summary("summarize the document"))

youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

yt_vs = VectorStore(collection_name="youtube_test")
yt_vs.delete_collection()


yt_docs   = loader.load(youtube_url)
yt_chunks = splitter.split(yt_docs)
yt_vs.add_documents(yt_chunks)
print(f"YouTube chunks stored: {len(yt_chunks)}")
print(get_summary(
    "summarize this video",
    collection_name="youtube_test"
))


print("\n\n")
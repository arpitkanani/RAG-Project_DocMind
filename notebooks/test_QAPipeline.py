import sys
import os
sys.path.append(os.getcwd())

from src.pipelines.qa_pipeline import QAPipeline
from src.pipelines.ingestion_pipeline import IngestionPipeline

from src.components.memory_manager import MemoryManager

ingestion= IngestionPipeline()
r=ingestion.run("test.txt",clear_existing=False)
MemoryManager().clear()

pipeline=QAPipeline(collection_name=r["collection_name"])
r1 = pipeline.run("what is machine learning?")

print(f"Intent    : {r1['intent']}")
print(f"Collection: {r1['collection']}")
print(f"Answer    : {r1['answer']}")

import sys
import os
sys.path.append(os.getcwd())

from src.pipelines.qa_pipeline import QAPipeline
from src.pipelines.ingestion_pipeline import IngestionPipeline

from src.components.memory_manager import MemoryManager

ingestion= IngestionPipeline()

MemoryManager().clear()





r2=ingestion.run("https://youtu.be/F1nQ_Osc6w8?si=0n27twjNb3xX1-Xm")
pipeline=QAPipeline(collection_name=r2["collection_name"])
rx=pipeline.run("write song lyrics minimum 1 song that guy sing about it")
print(f"Intent    : {rx['intent']}")
print(f"Collection: {rx['collection']}")
print(f"Answer    : {rx['answer']}")
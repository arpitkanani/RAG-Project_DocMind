# Component Documentation: `src/components/text_splitter.py`

## 1. Overview & Purpose
`src/components/text_splitter.py` breaks loaded documents and raw text strings into chunk sizes suitable for semantic embeddings and retrieval. It supports batch splitting as well as memory-efficient streaming splitting (`lazy_split`) for large documents.

---

## 2. Key Components & Functions

### `TextSplitter`

#### `__init__()`
- Configures `RecursiveCharacterTextSplitter` from `config/config.yaml`:
  - `chunk_size`: 900 characters (default).
  - `chunk_overlap`: 250 characters (default).
  - `separators`: `["\n\n", "\n", " ", ""]`.

#### `split(docs: List[Document]) -> List[Document]`
- Batch-splits a list of documents.
- Discards short noise chunks (length $< 25$ characters) such as trailing whitespace or isolated table borders.
- Logs chunk counts and average character length.

#### `lazy_split(docs: Iterator[Document]) -> Iterator[Document]`
- **Streaming Pipeline Method:** Takes an iterator of `Document` objects and yields chunks on the fly as each document page is processed.
- Prevents memory spikes when ingesting hundred-page PDFs or books.

#### `split_text(text: str) -> List[str]`
- Directly splits raw strings into text chunk arrays.

---

## 3. Connections & Component Mapping

```mermaid
flowchart LR
    DocLoader["src.components.document_loader: load()"] -->|Iterator[Document]| TextSplitter["TextSplitter.lazy_split()"]
    TextSplitter -->|Iterator[Chunk Documents]| IngestionPipe["src.pipelines.ingestion_pipeline: IngestionPipeline"]
    IngestionPipe --> VectorStore["src.components.vector_store: add_documents()"]
```

### Upstream Callers:
- `src.pipelines.ingestion_pipeline.IngestionPipeline: run()`

### Downstream Dependencies:
- `langchain_text_splitters.RecursiveCharacterTextSplitter`
- `src.exception.CustomException`
- `config/config.yaml` (`splitter` section)

---

## 4. AI & Developer Guidelines
- **Chunk Noise Threshold:** The 25-character minimum chunk threshold ensures isolated table cell artifacts do not pollute retrieval rankings.

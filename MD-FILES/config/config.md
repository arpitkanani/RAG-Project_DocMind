# Configuration Documentation: `config/config.yaml`

## 1. Overview & Purpose
`config/config.yaml` is the central operational configuration file for DocuVortex. It establishes hyperparameters for the embedding models, LLM provider fallbacks, rate-limiting classifiers, recursive text splitting, hybrid MMR and semantic retrieval, Qdrant vector storage, chat memory windows, file upload size limits, and PostgreSQL fallbacks.

---

## 2. Configuration Sections & Schema

### `embedding`
```yaml
embedding:
  model: "BAAI/bge-small-en-v1.5"
  device: "cpu"
```
- **`model`**: HuggingFace sentence-transformers model identifier.
- **`device`**: Compute target (`"cpu"` or `"cuda"`).

### `llm`
```yaml
llm:
  provider: "groq"
  model: "openai/gpt-oss-20b"
  temperature: 0.1
  max_tokens: 1024
  rate_limit:
    patterns:
      rpd: ["requests per day", "daily limit"]
      tpd: ["tokens per day"]
      tpm: ["tokens per minute"]
      rpm: []
    messages:
      rpm: "We're getting a lot of questions right now — please wait about a minute and try again."
      ...
```
- **`provider`**: Default primary LLM service (`"groq"` or `"google"`).
- **`model`**: Primary model ID.
- **`temperature`**: Generation temperature (low temperature = high grounding fidelity).
- **`rate_limit`**: Regex patterns and user-friendly messages for classifying rate-limit errors.

### `splitter`
```yaml
splitter:
  strategy: "recursive"
  chunk_size: 900
  chunk_overlap: 250
  separators: ["\n\n", "\n", " ", ""]
```
- Defines chunk boundaries for `src.components.text_splitter.TextSplitter`.

### `retriever`
```yaml
retriever:
  search_type: "similarity"
  k: 7
  fetch_k: 45
  max_query_variants: 3
  lambda_mult: 0.6
  score_threshold: 0.3
  summary_max_chars: 6000
  collection_margin: 0.8
  doc_margin: 0.7
```
- Controls retrieval behavior:
  - `k`: Maximum chunks passed to generation.
  - `fetch_k`: Initial candidate pool size for MMR and lexical reranking.
  - `max_query_variants`: Number of expanded sub-queries generated.
  - `collection_margin`: Relative score threshold a collection must achieve compared to the top collection to remain competitive.
  - `summary_max_chars`: Maximum character budget for document summaries.

### `vectorstore`
```yaml
vectorstore:
  url: "http://localhost:6333"
  collection_name: "docmind"
```
- Connection endpoint for Qdrant. Overridden by `QDRANT_URL` environment variable if present.

### `memory`
```yaml
memory:
  persist_directory: "data/memory"
  window_days: 7
  recent_messages_verbatim: 6
```
- `window_days`: Retention window for chat history queries.
- `recent_messages_verbatim`: Number of latest turns preserved verbatim before folding older messages into an LLM summary.

### `upload`
```yaml
upload:
  allowed_extensions: [".pdf", ".txt", ".docx", ".csv", ".md", ".xlsx"]
  upload_dir: "data/uploads"
  max_file_size_mb: 100
  max_files_count: 5
```
- Upload directory and security validation constraints.

### `youtube`
```yaml
youtube:
  language: ["en", "hi", "gu"]
  max_chars: 150000
```
- Transcript fetch language priorities and maximum character caps.

### `postgres`
```yaml
postgres:
  host: localhost
  port: 5432
  user: docmind
  password: ...
  database: docmind
```
- Default fallback database configuration if `DATABASE_URL` is omitted.

---

## 3. Component Consumers

- `src.components.embedder`: Reads `embedding`.
- `src.components.text_splitter`: Reads `splitter`.
- `src.components.retriever`: Reads `retriever`.
- `src.components.vector_store`: Reads `vectorstore`.
- `src.components.memory_manager`: Reads `memory`.
- `src.chains.qa_chain`: Reads `llm`, `retriever`.
- `src.utils.file_helper`: Reads `upload`.
- `src.utils.youtube_helper`: Reads `youtube`.
- `src.database.db`: Reads `postgres`.

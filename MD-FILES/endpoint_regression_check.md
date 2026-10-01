# Module Documentation: `endpoint_regression_check.py`

## 1. Overview & Purpose
`endpoint_regression_check.py` is an automated regression test script for DocuVortex HTTP endpoints. It installs mock stubs for external heavy dependencies (Chroma/Qdrant, LangChain embeddings, YouTube Transcript API) and uses FastAPI's `TestClient` to verify router contracts, status codes, and error payload structures across the API lifecycle.

---

## 2. Key Components & Functions

### `install_dependency_stubs()`
- Dynamically injects lightweight mock classes into `sys.modules` for:
  - `langchain_chroma`: Mock `Chroma` vector store and retriever.
  - `langchain_core.documents`, `messages`, `prompts`, `runnables`: Mock primitives.
  - `langchain_community.document_loaders`: Mock file loaders.
  - `youtube_transcript_api`: Mock transcript fetcher.

### Test Execution Suite (via `fastapi.testclient.TestClient`)
Verifies the following endpoints sequentially:
1. `GET /health` -> `200 OK`
2. `POST /sessions` -> Returns valid session ID
3. `GET /sessions` -> Lists active sessions
4. `POST /upload` -> Ingestion pipeline returns success
5. `POST /query` -> Single collection query returns answer
6. `POST /upload` (second document) -> Multi-document upload
7. `POST /query` (multi-collection) -> Multi-collection query routing
8. `POST /youtube` -> YouTube URL processing
9. `GET /collections` -> Returns collection set
10. `GET /sessions/{id}` -> Returns session attachments and history
11. `DELETE /collections/{name}` -> Deletes specific collection
12. `POST /query` (deleted collection) -> Returns `404 collection_not_found`
13. `DELETE /sessions/{id}` -> Deletes single session
14. `DELETE /sessions` -> Clears all sessions
15. `DELETE /memory` -> Flushes memory

---

## 3. Connections & Component Mapping

- **Direct Caller:** CI/CD or local terminal: `python endpoint_regression_check.py`.
- **Target Application:** `app.py` FastAPI app instance.
- **Components Mocked:** `IngestionPipeline`, `QAPipeline`, `VectorStore`.

---

## 4. AI & Developer Guidelines
- Run this regression check before committing major changes to routers or schemas to ensure all HTTP contract guarantees remain intact.

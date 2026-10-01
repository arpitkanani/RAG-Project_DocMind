# Router Documentation: `src/routers/collections.py`

## 1. Overview & Purpose
`src/routers/collections.py` provides REST endpoints for inspecting and dropping vector database collections stored in Qdrant.

---

## 2. Endpoints & Route Definitions

### `GET /collections`
- **Purpose:** Returns a list of all active collection names across the vector database.
- **Handler:** `list_collections()` calls `VectorStore().alist_collections()`.
- **Response:** `{"collections": ["doc_sample_12ab", "youtube_dQw4w9WgXcQ"]}`.

### `DELETE /collections/{collection_name}`
- **Purpose:** Deletes a specific collection from Qdrant and flushes the retriever cache.
- **Handler:** `delete_collection(collection_name)` calls `VectorStore(collection_name).adelete_collection()`.
- **Error Response:** Returns HTTP `404 collection_not_found` if collection does not exist.

---

## 3. Connections & Component Mapping

- **Mounted In:** `app.py: app.include_router(collections_router)`.
- **Interacts With:** `src.components.vector_store.VectorStore`, `src.utils.helpers: build_error_response`.

---

## 4. AI & Developer Guidelines
- Deleting a collection via this endpoint only drops the vector store table. To remove the attachment from a specific chat workspace as well, prefer `DELETE /sessions/{id}/attachments/{collection_name}`.

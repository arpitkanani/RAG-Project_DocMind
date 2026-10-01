# Utility Documentation: `src/utils/helpers.py`

## 1. Overview & Purpose
`src/utils/helpers.py` provides cross-cutting helper utilities for HTML template partial rendering, LLM text extraction, session scope resolution with strict tenant authorization, collection name generation, and mass vector collection cleanup.

---

## 2. Key Components & Functions

### `read_template(template_name: str) -> str`
- Reads an HTML file from `templates/` and dynamically replaces `<!-- INCLUDE: partial_name -->` directives with the contents of `templates/partials/{partial_name}.html`.

### `build_error_response(*, status_code: int, error_code: str, message: str, extra=None) -> JSONResponse`
- Generates standardized JSON error payloads matching frontend toast structures:
  `{"success": false, "error_code": "...", "message": "..."}`.

### `extract_text(content) -> str`
- Universal normalizer converting LLM output objects (`AIMessage`, `AIMessageChunk`, dictionaries, lists of part blocks, or strings) into clean, trimmed plain text.

### `normalize_collection_scope(request: QueryRequest) -> List[str] | None`
- Resolves `collection_names` or legacy `collection_name` into a clean list of strings.

### `build_collection_name(source_name: str, prefix="doc") -> str`
- Creates sanitized collection names: `{prefix}_{safe_stem}_{uuid4[:8]}`.

### `aresolve_session_scope(session_id, requested_scope, user_id) -> List[str]` (Async)
- **Multi-Tenant Ownership Verification:**
  - If `requested_scope` is provided: checks that every requested collection is actually bound to this user's session in PostgreSQL. Prevents IDOR (Insecure Direct Object Reference) exploits where a user attempts to query another tenant's vector collection.
  - If no scope is specified: returns all collections attached to the active session.
  - Raises `KnowledgeBaseEmptyError` if zero collections exist.

### `delete_collections(names)` / `adelete_collections(names) -> List[str]`
- Iterates over collection names and deletes them from Qdrant, returning the list of deleted names.

### `create_session_id() -> str`
- Generates an 8-character unique alphanumeric session identifier (e.g. `b47c1a9e`).

---

## 3. Connections & Component Mapping

- **Imported By:**
  - `src.routers.query`: Scope resolution, error building, text extraction.
  - `src.routers.pages`: Template reading.
  - `src.routers.sessions`: Collection deletion, session ID creation.
  - `src.chains.qa_chain` & `src.graph.nodes_agentic`: Text extraction.

---

## 4. AI & Developer Guidelines
- Always use `aresolve_session_scope` when accepting collection names from user inputs to prevent cross-tenant data leakage.

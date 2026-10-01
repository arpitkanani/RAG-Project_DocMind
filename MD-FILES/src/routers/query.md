# Router Documentation: `src/routers/query.py`

## 1. Overview & Purpose
`src/routers/query.py` is the streaming query execution endpoint of DocuVortex. It accepts user questions via `POST /query`, resolves authorized collection boundaries for the session, streams real-time Server-Sent Events (SSE) from the LangGraph workflow, translates internal graph nodes into user-facing status indicators, dynamically surfaces tool execution badges (stock, weather, search, math, arXiv), and emits typewriter text chunks for the synthesized answer.

---

## 2. Request Handling & Event Protocol

### Endpoint: `POST /query`
- **Request Body:** `src.schemas.QueryRequest` (`query`, `session_id`, `collection_names`, `message_attachments`).
- **Authentication:** `Depends(get_current_user)`.
- **Response:** `StreamingResponse(event_generator(), media_type="text/event-stream")`.

### SSE Event Format
The generator streams events matching this schema:
```text
data: {"type": "status", "stage": "classifying", "message": "Analyzing query..."}

data: {"type": "tool_status", "tool": "get_stock_price", "badge_type": "stock", "message": "📈 Fetching stock price for TSLA..."}

data: {"type": "tool_end", "tool": "get_stock_price"}

data: {"type": "token", "content": "The "}
data: {"type": "token", "content": "latest "}

data: {"type": "done", "status": "completed", "final_answer": "...", "session_id": "..."}
```

### Event Types Dispatched:
1. **`status`**: Emitted when entering new graph nodes (`load_context`, `classify_intent`, `retrieve_qa`, `generate`, etc.).
2. **`tool_status`**: Emitted on `on_tool_start`. Formats dynamic badges and friendly descriptions based on tool inputs (e.g. stock symbol, weather city, search query).
3. **`tool_end`**: Emitted on `on_tool_end` when tool execution completes.
4. **`clarification`**: Emitted if the query is vague, delivering structured question options.
5. **`token`**: Word-by-word typewriter streaming of the finalized answer.
6. **`done`**: Terminal event containing the completed answer and session ID.
7. **`error`**: Structured error payloads on rate limits or graph failures.

---

## 3. Connections & Component Mapping

```mermaid
flowchart TD
    Client[Browser Frontend JS] -->|POST /query (SSE)| QueryRouter["src.routers.query: query()"]
    
    QueryRouter --> Auth["src.auth: get_current_user"]
    QueryRouter --> Scope["src.utils.helpers: aresolve_session_scope()"]
    
    QueryRouter --> GraphStream["target_graph.astream_events(..., version='v2')"]
    GraphStream --> SafeFallback{"Checkpointer Connection Dropped?"}
    SafeFallback -->|Yes| DirectGraph["Fall back seamlessly to uncheckpointed rag_graph"]
    SafeFallback -->|No| StreamActive["Active Event Stream"]
    
    StreamActive --> SSETranslator["Translate Nodes to SSE Status & Tool Events"]
    SSETranslator --> Typewriter["Token Chunk Delivery"]
    Typewriter --> DoneEvent["Done Event"]
    DoneEvent --> Client
```

### Upstream Callers:
- Frontend client (`templates/static/js/app_new.js: streamQuery`).

### Downstream Dependencies:
- `src.auth: get_current_user`
- `src.graph.builder: rag_graph` (and `app.state.rag_graph`)
- `src.utils.helpers: aresolve_session_scope, normalize_collection_scope, extract_text`
- `src.utils.rate_limiter: LLMRateLimitError`

---

## 4. AI & Developer Guidelines
- **Seamless Drop Recovery:** If the checkpointer pool drops a connection during streaming, `_safe_stream_events()` catches the connection drop and instantly transfers streaming to direct `rag_graph` without raising a visible error to the user.
- **Tenant Scope Security:** `aresolve_session_scope` ensures users can never access or query collections belonging to another user.

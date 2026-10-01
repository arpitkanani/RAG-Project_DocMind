# Graph State Documentation: `src/graph/state.py`

## 1. Overview & Purpose
`src/graph/state.py` defines the shared typing schemas and state contracts that flow through LangGraph nodes and conditional edges during agent execution in DocuVortex.

---

## 2. State Schemas

### `RAGState(TypedDict, total=False)`
Primary state for the full RAG workflow compiled in `src.graph.builder:build_rag_graph`.

#### Request Inputs:
- **`question`** (`str`): The user's prompt or question.
- **`collection_names`** (`List[str]`): Scoped document collections for vector search.
- **`session_id`** (`str`): Current chat session identifier.
- **`user_id`** (`str`): Authenticated user UUID.
- **`message_attachments`** (`Optional[List[dict]]`): Active file or video metadata attached to this turn.
- **`source_selected`** (`Optional[bool]`): Flag indicating whether the session has attached knowledge sources.

#### Internal Routing & Context:
- **`is_summary`** (`bool`): Set to `True` if the user asked for a whole-document overview.
- **`chat_history`** (`List[dict]`): Prior message turns loaded from PostgreSQL.
- **`docs`** (`List[Document]`): Passages retrieved from Qdrant.
- **`intent`** (`str`): Classification result (`'conversational'`, `'chitchat'`, `'retrieval'`, or `'clarify'`).
- **`clarification_feedback`** (`Optional[str]`): Message explaining why a query is ambiguous.
- **`grade`** (`str`): Relevance grading result (`'relevant'` or `'irrelevant'`).

#### Output & Persistence:
- **`raw_answer`** (`str`): Direct LLM synthesis output.
- **`final_answer`** (`str`): Polished, grounded response string.
- **`citations`** (`str`): Deterministic source citations block.

#### Error Tracking:
- **`error_code`** (`Optional[str]`): Machine-readable failure tag.
- **`error_message`** (`Optional[str]`): Human-friendly description.

---

### `AgentState(TypedDict)`
State schema for the stateless ReAct agent (`src.graph.agent`):
- **`messages`** (`Annotated[Sequence[BaseMessage], add_messages]`): Message stream with append reducer.
- **`collection_names`** (`Optional[List[str]]`): Scoped collection names.
- **`session_id`** (`str`): Current session.
- **`user_id`** (`str`): User ID.

---

## 3. Connections & Component Mapping

- Used by `src.graph.builder`: StateGraph type parameter `StateGraph(RAGState)`.
- Used by all graph nodes in `src.graph.nodes_agentic`, `src.graph.nodes.retrieve`, and `src.graph.nodes.generate`.
- Transferred across SSE stream events in `src.routers.query`.

---

## 4. AI & Developer Guidelines
- All keys in `RAGState` have `total=False`, allowing nodes to return partial dictionaries updating only the keys they modify.

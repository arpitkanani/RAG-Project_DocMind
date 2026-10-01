# Agent Runner Documentation: `src/graph/agent.py`

## 1. Overview & Purpose
`src/graph/agent.py` compiles the standalone document ReAct agent workflow and provides an asynchronous event streaming generator (`astream_agent_response`) that streams token chunks and tool status updates for real-time client consumption.

---

## 2. Key Components & Functions

### `build_react_agent()`
- Compiles a two-node `StateGraph(AgentState)` (`chat_node` <-> `tools`).
- **Stateless Per-Request:** Does not require a checkpointer because conversation history is loaded and persisted via PostgreSQL `MemoryManager`.

### `astream_agent_response(...) -> AsyncGenerator[Dict[str, Any], None]`
- Streams events from `react_agent.astream_events(..., version="v2")`:
  - **`on_tool_start`**: Yields `{"type": "tool_start", "name": tool_name, "input": ...}`.
  - **`on_tool_end`**: Yields `{"type": "tool_end", "name": tool_name}`.
  - **`on_chat_model_stream`**: Emits user-facing text tokens `{"type": "token", "content": text}`. Suppresses token emission during tool-calling decision phases.
  - **Done Event**: Yields `{"type": "done", "final_answer": final_answer}`.

---

## 3. Connections & Component Mapping

- **Imports From:**
  - `src.graph.agent_nodes`: `chat_node`, `should_continue`, `tools_node`.
  - `src.graph.helpers`: `extract_message_text`.
  - `src.graph.state`: `AgentState`.

---

## 4. AI & Developer Guidelines
- This module provides an alternative streaming ReAct interface for direct document interactions if bypassing the higher-level `build_rag_graph` state machine.

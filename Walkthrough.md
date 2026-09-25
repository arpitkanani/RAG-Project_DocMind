# Walkthrough — ReAct Agent Integration, Forced Tool Calling, PostgreSQL Checkpointer & Web Search Fix

We have upgraded the non-document (`chitchat`) branch of DocuVortex to use LangGraph's prebuilt **ReAct Agent (`create_react_agent`)** with strict system instructions that force tool calling, structured the entire codebase's **LangSmith Tracing** into clean, hierarchical step names, fixed persistent memory amnesia across conversation turns, suppressed intermediate stream leakage, and made the **live web search (`search_tool`)** robust and resilient.

---

## 1. Architectural Summary

```
START → Load_Context → Intent_Classifier
         ├─(docs selected) ──→ Retrieve_QA / Retrieve_Summary ──→ RAG_Generator (grade_docs bypassed) ──→ Finalize_Response ──→ END
         └─(no docs) ────────→ ReAct_Chitchat_Agent (create_react_agent)
                                  │
                                  ├─→ calculator (@traceable)
                                  ├─→ get_stock_price (@traceable)
                                  └─→ search_tool (@traceable - multi-backend DDG + Wiki fallback)
                                  │
                                  ▼
                               Finalize_Response ──→ END
```

---

## 2. Issues Diagnosed & Solutions Applied

### A. Live Web Search Not Returning Web Results
- **Problem**: Queries requesting search (e.g. current events or latest news) were answered from the LLM's static training knowledge or failing silently. The default `DuckDuckGoSearchRun` wrapper was throwing rate limit or backend scraping exceptions without fallbacks, and the tool name was inconsistently named across the backend and frontend.
- **Solution**:
  1. **Multi-Backend Search Implementation (`src/graph/tools.py`)**:
     - Upgraded `search_tool` into a custom `@tool` decorated with `@traceable(name="search_tool")`.
     - **Strategy 1**: Uses the modern `duckduckgo_search` library directly with cycling across backends (`api` → `html` → `lite`) to extract rich title, snippet, and source link metadata.
     - **Strategy 2**: Falls back gracefully to `langchain_community`'s `DuckDuckGoSearchResults` and `DuckDuckGoSearchRun`.
     - **Strategy 3**: Automatically falls back to the Wikipedia REST summary API if search engines block or throttle requests, ensuring facts are retrieved live rather than falling back to static model memory.
  2. **Consistent Tool Routing & Badge Mapping**:
     - Registered `search_tool` across `tool_display_map` in both `src/routers/query.py` and `templates/static/js/app_new.js`.
     - Added dynamic query truncation for search queries to display `🔍 Searching web for '<query>'...`.
  3. **Strict System Instructions**:
     - Instructed `qwen/qwen3.8-27b` explicitly in `nodes_agentic.py` that it MUST invoke `search_tool` for current events, news, or factual lookups and NOT rely on internal static memory.

### B. Conversation Amnesia Across Turns in Chitchat
- **Problem**: When chatting without documents, the chitchat node forgot previous turns (e.g. user name).
- **Solution**:
  - In `src/graph/nodes_agentic.py`, updated `run_react_agent(state, config)` to read `chat_history = state.get("chat_history", [])` (which is loaded asynchronously from PostgreSQL by `load_context_node`) and prepend all prior messages to the agent input before the current question.
  - Attached PostgreSQL `AsyncPostgresSaver` with connection pooling in `app.py` lifespan and passed `thread_id: session_id` through graph execution.

### C. Intermediate Thought / Stream Leakage
- **Problem**: ReAct conversational preambles ("Let me search that for you...") were streaming into the frontend while tools were still in flight.
- **Solution**:
  - In `src/routers/query.py`, tracked `tools_active_count` and `has_tool_run`.
  - Added token buffering prior to tool calls (`pre_tool_tokens`). If `on_tool_start` fires, the buffer is dropped immediately, ensuring only clean tool status badges appear in the UI followed by the final post-tool synthesis.

---

## 3. Changes Made & Files Modified

### 1. `src/graph/tools.py`
- Implemented robust `search_tool(query: str) -> str`:
  - Decorated with `@tool` and `@traceable(name="search_tool")`.
  - Direct `duckduckgo_search.DDGS` multi-backend iteration (`api`, `html`, `lite`).
  - LangChain fallback (`DuckDuckGoSearchResults`, `DuckDuckGoSearchRun`).
  - Wikipedia REST API fallback.
- Retained `@traceable` tools `calculator` and `get_stock_price`.
- Exported `chitchat_tools = [search_tool, calculator, get_stock_price]`.

### 2. `src/graph/nodes_agentic.py`
- Prebuilt ReAct agent setup with `create_react_agent`.
- Clear system prompt instructing tool invocation:
  ```python
  system_instruction = (
      "You are DocuVortex, an advanced AI assistant equipped with real-time tools. "
      "TOOL USAGE RULES: "
      "1. You MUST call `get_stock_price` whenever the user asks for stock quotes, share prices, or market data. "
      "2. You MUST call `calculator` for any mathematical operations or arithmetic calculations. "
      "3. You MUST call `search_tool` for current events, latest news, recent developments, real-world facts, or whenever the user asks to look up, search, or check something on the web. Do NOT rely on static memory for current or verifiable facts. "
      "When tools return results, synthesize the findings into a clear, concise, and helpful response."
  )
  ```
- History integration: Prepends `state.get("chat_history", [])` to inputs.

### 3. `src/routers/query.py`
- Unified tool mappings: Added `"search_tool"` to `tool_display_map` and dynamic argument extraction (`tool_input["query"]`).
- Suppressed token stream when tools are running and discarded preliminary preamble chunks.
- Passed `"thread_id": session_id` in `config["configurable"]`.

### 4. `templates/static/js/app_new.js`
- Added `"search_tool"` to `toolDisplayMap` so `🔍 Searching the web...` displays whenever the agent triggers search.
- Preserved seamless transition from tool badge to response tokens without flicker.

---

## 4. Verification Guide

1. **Web Search Verification (Forces `search_tool`)**:
   - Query: `"Search the web for the latest updates on SpaceX Starship flights in 2025/2026."` or `"Who won the latest Super Bowl?"`
   - **Backend / LangSmith Trace**: `Intent_Classifier` → `ReAct_Chitchat_Agent` → `search_tool` (multi-backend search returns web snippets) → `Finalize_Response`.
   - **UI**: Shows `🔍 Searching web for '...'` badge, then streams the answer with live web facts.

2. **Stock Price Query (Forces `get_stock_price`)**:
   - Query: `"What is the current stock price of Apple (AAPL)?"`
   - **UI**: Shows `📈 Fetching stock price for AAPL...` badge, then streams real-time quote data from Alpha Vantage.

3. **Math Calculation (Forces `calculator`)**:
   - Query: `"Calculate 450 * 32"`
   - **UI**: Shows `🧮 Calculating 450.0 mul 32.0...` badge, then outputs `14400.0`.

4. **Multi-turn Memory (Verifies Context Retention)**:
   - Turn 1: `"My name is Arpit. Remember this."`
   - Turn 2: `"What is my name?"`
   - **Result**: The assistant responds with your name immediately, without forgetting across turns.

5. **Document RAG Integrity**:
   - When a document is attached/selected, the query routes through `Retrieve_QA` → `RAG_Generator` → `Finalize_Response` with source citations, maintaining document isolation.

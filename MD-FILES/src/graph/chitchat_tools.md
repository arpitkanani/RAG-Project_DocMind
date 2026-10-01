# Module Documentation: `src/graph/chitchat_tools.py`

## 1. Overview & Purpose
`src/graph/chitchat_tools.py` is a clean re-export module that exposes conversational tools (`calculator`, `chitchat_tools`, `get_stock_price`, `get_weather`, `search_arxiv`, `search_tool`) from `src.graph.tools` for backwards compatibility with earlier graph iterations.

---

## 2. Exports & Definitions

```python
from src.graph.tools import (
    calculator,
    chitchat_tools,
    get_stock_price,
    get_weather,
    search_arxiv,
    search_tool,
)

__all__ = [
    "calculator",
    "chitchat_tools",
    "get_stock_price",
    "get_weather",
    "search_arxiv",
    "search_tool",
]
```

---

## 3. Connections & Component Mapping

- **Imported By:** `src.graph.chitchat_subgraph` to bind tools to the chitchat LLM.
- **Upstream Source:** `src.graph.tools`.

---

## 4. AI & Developer Guidelines
- When adding new external tools intended for conversational queries, register them in `src.graph.tools` and append their symbol to `__all__` in this file.

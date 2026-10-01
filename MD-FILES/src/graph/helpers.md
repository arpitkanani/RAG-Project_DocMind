# Graph Helpers Documentation: `src/graph/helpers.py`

## 1. Overview & Purpose
`src/graph/helpers.py` provides utility functions for extracting, unnesting, and normalizing message text from heterogeneous LLM responses across different model providers (Gemini 3.x, Anthropic Claude, and Groq).

---

## 2. Key Components & Functions

### `extract_message_text(content: Union[str, List[Any], None]) -> str`
- **Purpose:** Standardizes any model output into a flat string.
- **Handles:**
  - Standard strings: `"Hello world"`.
  - Gemini / Claude block part formats: `[{'type': 'text', 'text': '...'}, {'text': '...'}]`.
  - Nested lists or arrays of text objects.
  - Dicts with `"text"` or `"content"` keys.
  - Objects possessing a `.text` attribute.
  - `None` or empty containers -> returns `""`.

---

## 3. Connections & Component Mapping

- **Imported By:**
  - `src.graph.agent`: Normalizes stream chunk text.
  - `src.graph.agent_nodes`: Unpacks user message content for tool forcing.
  - `src.graph.nodes_agentic`: Extracts text from clarification and chitchat model outputs.

---

## 4. AI & Developer Guidelines
- Always wrap LLM response `.content` in `extract_message_text()` when reading outputs from Gemini 3.x models, as Gemini frequently returns content as an array of structured part blocks rather than a single string.

# Module Documentation: `test.py`

## 1. Overview & Purpose
`test.py` is a model availability and quota verification script. It systematically invokes every active Google Gemini and Groq free-tier model using LangChain interfaces to detect deprecated model names, 429 quota limits, and API authentication issues before runtime.

---

## 2. Key Components & Functions

### Model Rosters Tested:
- **Gemini Models:** `gemini-3.8-flash`, `gemini-3.7-flash`, `gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-3-flash-preview`.
- **Groq Models:** `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `openai/gpt-oss-safeguard-20b`, `qwen/qwen3.6-27b`, `meta-llama/llama-4-maverick-17b-128e-instruct`, `moonshotai/kimi-k2-instruct-0905`, `qwen/qwen3.8-27b`.

### `extract_text(content) -> str`
- Normalizes responses from string, dict, or structured content part arrays (common in Gemini 3.x) into clean text.

### `test_model(label, model_name, llm_factory, delay=2) -> bool`
- Instantiates the LLM client, invokes it with a test prompt, reports `[OK]` or `[FAIL]`, and sleeps for `delay` seconds to respect RPM rate limits.

---

## 3. Connections & Component Mapping

- **Direct Caller:** Terminal CLI: `python test.py`.
- **External APIs:** Google Gemini Generative Language API, Groq Cloud API.
- **Environment Variables:** `GOOGLE_API_KEY`, `GROQ_API_KEY`.

---

## 4. AI & Developer Guidelines
- Run this script if the application starts returning LLM API errors or unexpected fallbacks to see which provider endpoints or model tags are operational.

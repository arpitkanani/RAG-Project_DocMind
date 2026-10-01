# Module Documentation: `src/logger.py`

## 1. Overview & Purpose
`src/logger.py` sets up the centralized logging infrastructure for DocuVortex. It configures the standard Python `logging` module to output formatted logs into rotating timestamped files within the `logs/` directory.

---

## 2. Key Components & Configuration

### Directory & File Setup
- **`LOG_DIRECTORY`**: `logs/` in the workspace root. Created automatically if not present (`os.makedirs(LOG_DIRECTORY, exist_ok=True)`).
- **`LOG_FILE_NAME`**: Formatted with the start timestamp: `MM_DD_YYYY+HH_MM_SS.log`.
- **`LOG_FILE_PATH`**: Full absolute path to the active log file.

### Formatting & Logging Level
```python
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
```
- Includes timestamp, line number, logger name, severity level, and log message.

---

## 3. Connections & Component Mapping

- **Imported By:** Imported as `from src.logger import logging` across all routers, graph nodes, components, pipelines, database adapters, and utility scripts.
- **Log Outputs:** Captures ingestion progress, SQL execution traces, LLM provider fallback alerts, rate-limit warnings, and error tracebacks.

---

## 4. AI & Developer Guidelines
- When debugging issues reported in the frontend or during test runs, inspect the newest `.log` file located in `logs/`.

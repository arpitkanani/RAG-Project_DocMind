# Utility Documentation: `src/utils/job_manager.py`

## 1. Overview & Purpose
`src/utils/job_manager.py` provides an in-memory, thread-safe background job tracker for asynchronous file and YouTube ingestion pipelines. It enables the web client to submit uploads asynchronously and poll for embedding progress without keeping long-running HTTP connections open.

---

## 2. Key Components & Functions

### `JobStatus(str, Enum)`
- `PROCESSING = "processing"`
- `READY = "ready"`
- `FAILED = "failed"`

### `UploadJobManager`
Thread-safe singleton tracking job state dictionaries via `threading.Lock()`:

- **`create_job() -> str`**: Generates a 32-character hexadecimal UUID and sets status to `PROCESSING`.
- **`update_progress(job_id: str, message: str)`**: Updates progress status message (e.g. chunk embedding counts).
- **`mark_ready(job_id: str, result: dict)`**: Sets status to `READY`, attaches the ingestion result payload, and sets message to `"Complete"`.
- **`mark_failed(job_id: str, error: str)`**: Sets status to `FAILED` with the error description.
- **`get_job(job_id: str) -> Optional[dict]`**: Returns a snapshot copy of the job state.

### `upload_job_manager`
- Shared singleton instance imported across the application.

---

## 3. Connections & Component Mapping

```mermaid
flowchart LR
    UploadRouter["src.routers.upload"] -->|create_job()| JobManager["upload_job_manager"]
    YTRouter["src.routers.youtube"] -->|create_job()| JobManager
    
    Worker["BackgroundTasks Ingestion Worker"] -->|update_progress(), mark_ready()| JobManager
    
    StatusRoute["GET /upload/status/{id}"] -->|get_job()| JobManager
    Browser[Frontend Poller] --> StatusRoute
```

### Upstream Callers:
- `src.routers.upload: upload_file, _run_upload_job, get_upload_status`
- `src.routers.youtube: process_youtube, _run_youtube_job`

---

## 4. AI & Developer Guidelines
- **Process Scope:** `UploadJobManager` is stored in process RAM. In a multi-worker deployment (e.g. multiple Gunicorn worker processes), job status must be stored in PostgreSQL or Redis so all workers can see the job ID.

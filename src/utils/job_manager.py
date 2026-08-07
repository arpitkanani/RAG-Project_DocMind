import threading
import uuid
from enum import Enum
from typing import Optional


class JobStatus(str, Enum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class UploadJobManager:
    """
    In-memory tracker for background upload jobs.

    NOTE ON SCOPE: this is per-process state, same caveat as the rate
    limiter. Restarting the app loses in-flight job history (acceptable --
    jobs are short-lived and the frontend only polls while the server is
    up). For a multi-worker deployment, this would need to move to
    Postgres so every worker sees the same job state.
    """

    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def create_job(self) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = {
                "status": JobStatus.PROCESSING,
                "message": "Processing...",
                "result": None,
                "error": None,
            }
        return job_id

    def update_progress(self, job_id: str, message: str):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["message"] = message

    def mark_ready(self, job_id: str, result: dict):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = JobStatus.READY
                self._jobs[job_id]["result"] = result
                self._jobs[job_id]["message"] = "Complete"

    def mark_failed(self, job_id: str, error: str):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = JobStatus.FAILED
                self._jobs[job_id]["error"] = error
                self._jobs[job_id]["message"] = "Failed"

    def get_job(self, job_id: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None


# Shared across the whole app -- one tracker, all upload jobs from all users
upload_job_manager = UploadJobManager()
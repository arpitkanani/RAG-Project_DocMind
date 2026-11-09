import queue
import threading
import uuid
from enum import Enum
from typing import Callable, Optional

from src.logger import logging


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class UploadJobManager:
    """
    In-memory tracker and sequential queue for background ingestion jobs.
    Ensures multiple concurrent uploads are queued and processed one by one
    to prevent memory spikes, CPU exhaustion, and embedding rate limits.
    """

    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._queue = queue.Queue()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="IngestionQueueWorker")
        self._worker_thread.start()

    def _worker_loop(self):
        while True:
            try:
                task_fn, job_id = self._queue.get()
                try:
                    with self._lock:
                        if job_id in self._jobs and self._jobs[job_id]["status"] == JobStatus.QUEUED:
                            self._jobs[job_id]["status"] = JobStatus.PROCESSING
                            self._jobs[job_id]["message"] = "Processing document..."

                    logging.info("Starting queued ingestion job: %s", job_id)
                    task_fn()
                except Exception as e:
                    logging.exception("Error executing queued ingestion job %s: %s", job_id, e)
                    self.mark_failed(job_id, "Could not process this file. Please try uploading it again.")
                finally:
                    self._queue.task_done()
            except Exception as e:
                logging.exception("Unexpected error in ingestion queue worker loop: %s", e)

    def submit_job(self, task_builder: Callable[[str], None], initial_message: str = "Processing...") -> str:
        """
        Creates a job and enqueues it for sequential execution.
        """
        job_id = uuid.uuid4().hex
        with self._lock:
            q_size = self._queue.qsize()
            # If a task is currently executing or other tasks are in queue, mark as queued
            status = JobStatus.QUEUED if q_size > 0 else JobStatus.PROCESSING
            msg = "Waiting in queue..." if q_size > 0 else initial_message
            self._jobs[job_id] = {
                "status": status,
                "message": msg,
                "result": None,
                "error": None,
            }
        self._queue.put((lambda: task_builder(job_id), job_id))
        return job_id

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
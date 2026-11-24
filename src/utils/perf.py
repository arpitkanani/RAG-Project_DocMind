"""Performance instrumentation module for measuring auth and database latencies.
Active only when PERF_DEBUG=1.
Zero behavior changes.
Never logs passwords, tokens, cookies, API keys, or full email addresses.
"""

import os
import re
import time
import logging
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

# Setup rotating perf logger: 1 MB x 3 backups
os.makedirs("logs", exist_ok=True)
PERF_LOG_PATH = os.path.abspath("logs/perf.log")
_perf_logger = logging.getLogger("perf_telemetry")
_perf_logger.setLevel(logging.INFO)
if not _perf_logger.handlers:
    _rfh = RotatingFileHandler("logs/perf.log", maxBytes=1024 * 1024, backupCount=3, encoding="utf-8")
    _rfh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    _perf_logger.addHandler(_rfh)
    _perf_logger.propagate = False

_perf_ctx: ContextVar[Optional[dict]] = ContextVar("perf_ctx", default=None)
_rag_perf_collector: ContextVar[Optional[dict]] = ContextVar("rag_perf_collector", default=None)
_GLOBAL_RAG_COLD = True


def is_rag_cold() -> int:
    global _GLOBAL_RAG_COLD
    val = 1 if _GLOBAL_RAG_COLD else 0
    _GLOBAL_RAG_COLD = False
    return val


@contextmanager
def capture_rag_perf():
    """Captures granular sub-step timings for RAG retrieval."""
    stats = {
        "collection_exists_check_ms": 0.0,
        "query_variants_build_ms": 0.0,
        "emb_calls": 0,
        "emb_ms": [],
        "emb_provider": "unknown",
        "emb_model": "unknown",
        "emb_batched": 0,
        "qdrant_calls": 0,
        "qdrant_ms": [],
        "rerank_ms": 0.0,
        "merge_and_format_ms": 0.0,
        "total_rag_ms": 0.0,
        "cold": is_rag_cold(),
    }
    token = _rag_perf_collector.set(stats)
    try:
        yield stats
    finally:
        _rag_perf_collector.reset(token)


def get_rag_perf() -> Optional[dict]:
    return _rag_perf_collector.get()


def record_rag_embedding(dur_ms: float, count: int, provider: str, model: str, batched: int):
    collector = _rag_perf_collector.get()
    if collector is not None:
        collector["emb_calls"] += 1
        collector["emb_ms"].append(round(dur_ms, 1))
        collector["emb_provider"] = provider
        collector["emb_model"] = model
        collector["emb_batched"] = batched


def record_rag_qdrant(dur_ms: float):
    collector = _rag_perf_collector.get()
    if collector is not None:
        collector["qdrant_calls"] += 1
        collector["qdrant_ms"].append(round(dur_ms, 1))


EMAIL_MASK_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_MASK_REGEX = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,9}")


def mask_pii(text: str) -> str:
    """Masks emails and phone numbers from queries or text before logging."""
    if not text:
        return ""
    clean = EMAIL_MASK_REGEX.sub("[email-redacted]", str(text))
    clean = PHONE_MASK_REGEX.sub("[phone-redacted]", clean)
    return clean


def is_perf_enabled() -> bool:
    return os.getenv("PERF_DEBUG", "0").strip().lower() in ("1", "true", "yes")


def log_perf_kv(tag: str, **kwargs) -> None:
    """
    Logs single-line key=value telemetry to stdout and logs/perf.log if PERF_DEBUG=1.
    Never logs passwords, tokens, API keys, full prompts, or raw PII.
    """
    if not is_perf_enabled():
        return
    items = []
    for k, v in kwargs.items():
        if k in ("query", "user_query", "text", "query_masked"):
            v_str = repr(mask_pii(str(v)[:80]))
        elif k in ("prompt", "system_prompt", "full_reply"):
            v_str = f"len={len(str(v))}"
        elif isinstance(v, float):
            v_str = f"{v:.1f}"
        else:
            v_str = str(v)
        items.append(f"{k}={v_str}")
    line = f"[perf:{tag}] " + " ".join(items)
    print(line, flush=True)
    _perf_logger.info(line)



def init_perf_context(method: str, path: str) -> None:
    """Initialize a performance tracking context for the current request."""
    if not is_perf_enabled():
        return
    _perf_ctx.set({
        "method": method,
        "path": path,
        "start": time.perf_counter(),
        "steps": {},
        "counts": {"db_checkouts": 0, "db_queries": 0},
    })


def get_perf_context() -> Optional[dict]:
    return _perf_ctx.get()


@contextmanager
def perf_step(name: str):
    """Time a specific operation block if PERF_DEBUG=1."""
    ctx = _perf_ctx.get()
    if ctx is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        ctx["steps"][name] = ctx["steps"].get(name, 0.0) + dt


def perf_count(name: str, inc: int = 1) -> None:
    """Increment a metric counter if PERF_DEBUG=1."""
    ctx = _perf_ctx.get()
    if ctx is not None:
        ctx["counts"][name] = ctx["counts"].get(name, 0) + inc


def finish_perf_context() -> tuple[Optional[str], Optional[str]]:
    """
    Finalizes the context and returns (summary_line, server_timing_header).
    Prints to stdout and appends to logs/perf.log.
    """
    ctx = _perf_ctx.get()
    if ctx is None:
        return None, None

    total_ms = (time.perf_counter() - ctx["start"]) * 1000.0
    db_checkouts = ctx["counts"].get("db_checkouts", 0)

    # Standard ordered keys
    ordered_keys = [
        "db_checkout",
        "db_probe",
        "db_query",
        "bcrypt_hash",
        "bcrypt_verify",
        "session_insert",
        "cookie_set",
    ]

    parts = [f"[perf] {ctx['method']} {ctx['path']} total={total_ms:.1f}ms db_checkouts={db_checkouts}"]
    server_timing_entries = [f"total;dur={total_ms:.1f}"]

    for k in ordered_keys:
        if k in ctx["steps"]:
            dur_ms = ctx["steps"][k] * 1000.0
            parts.append(f"{k}={dur_ms:.1f}ms")
            server_timing_entries.append(f"{k};dur={dur_ms:.1f}")

    for k, dur in ctx["steps"].items():
        if k not in ordered_keys:
            dur_ms = dur * 1000.0
            parts.append(f"{k}={dur_ms:.1f}ms")
            server_timing_entries.append(f"{k};dur={dur_ms:.1f}")

    summary_line = " ".join(parts)
    server_timing_header = ", ".join(server_timing_entries)

    # Output to stdout and write to rotating perf log
    print(summary_line, flush=True)
    _perf_logger.info(summary_line)

    # Reset context
    _perf_ctx.set(None)
    return summary_line, server_timing_header

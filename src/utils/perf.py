"""Performance instrumentation module for measuring auth and database latencies.
Active only when PERF_DEBUG=1.
Zero behavior changes.
Never logs passwords, tokens, cookies, API keys, or full email addresses.
"""

import os
import time
import logging
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

# Setup rotating perf logger: 1 MB x 3 backups
os.makedirs("logs", exist_ok=True)
_perf_logger = logging.getLogger("perf_telemetry")
_perf_logger.setLevel(logging.INFO)
if not _perf_logger.handlers:
    _rfh = RotatingFileHandler("logs/perf.log", maxBytes=1024 * 1024, backupCount=3, encoding="utf-8")
    _rfh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    _perf_logger.addHandler(_rfh)
    _perf_logger.propagate = False

_perf_ctx: ContextVar[Optional[dict]] = ContextVar("perf_ctx", default=None)


def is_perf_enabled() -> bool:
    return os.getenv("PERF_DEBUG", "0").strip().lower() in ("1", "true", "yes")


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

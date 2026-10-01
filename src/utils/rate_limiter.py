import asyncio
import threading
import time
from collections import deque

import yaml

from src.logger import logging

with open("config/config.yaml") as f:
    _config = yaml.safe_load(f)


class RateLimiter:
    """Sliding-window rate limiter supporting both sync and async acquisition."""

    def __init__(self, max_requests: int, per_seconds: float):
        self.max_requests = max_requests
        self.per_seconds = per_seconds
        self._calls = deque()
        self._sync_lock = threading.Lock()
        self._async_lock = None

    def _get_async_lock(self) -> asyncio.Lock:
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        return self._async_lock

    def acquire(self, weight: int = 1):
        """Synchronously wait for capacity."""
        weight = max(1, weight)
        while True:
            with self._sync_lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] > self.per_seconds:
                    self._calls.popleft()

                if len(self._calls) + weight <= self.max_requests:
                    for _ in range(weight):
                        self._calls.append(now)
                    return

                wait_time = self.per_seconds - (now - self._calls[0]) + 0.1

            logging.info("Rate limiter: waiting %.1fs for capacity (weight=%d)", wait_time, weight)
            time.sleep(max(wait_time, 0.1))

    async def aacquire(self, weight: int = 1):
        """Asynchronously wait for capacity without blocking the event loop."""
        weight = max(1, weight)
        lock = self._get_async_lock()
        while True:
            async with lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] > self.per_seconds:
                    self._calls.popleft()

                if len(self._calls) + weight <= self.max_requests:
                    for _ in range(weight):
                        self._calls.append(now)
                    return

                wait_time = self.per_seconds - (now - self._calls[0]) + 0.1

            logging.info("Async Rate limiter: waiting %.1fs for capacity (weight=%d)", wait_time, weight)
            await asyncio.sleep(max(wait_time, 0.1))


_llm_rpm_limit = _config.get("llm", {}).get("max_requests_per_minute", 25)
llm_rate_limiter = RateLimiter(max_requests=_llm_rpm_limit, per_seconds=60)

_RATE_LIMIT_CONFIG = _config.get("llm", {}).get("rate_limit", {})
_PATTERNS: dict[str, list[str]] = _RATE_LIMIT_CONFIG.get("patterns", {})
_MESSAGES: dict[str, str] = _RATE_LIMIT_CONFIG.get("messages", {})

_GENERIC_RATE_LIMIT_MARKERS = [
    "rate limit",
    "rate_limit",
    "resource_exhausted",
    "quota",
    "429",
    "too many requests",
]


class LLMRateLimitError(Exception):
    def __init__(self, kind: str, message: str):
        self.kind = kind
        self.message = message
        super().__init__(message)


def is_rate_limit_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in _GENERIC_RATE_LIMIT_MARKERS)


def classify_rate_limit(error: Exception) -> str:
    text = str(error).lower()
    for kind, patterns in _PATTERNS.items():
        if kind == "rpm":
            continue
        for pattern in patterns:
            if pattern and pattern.lower() in text:
                return kind
    return "rpm"


def get_rate_limit_message(kind: str) -> str:
    return (
        _MESSAGES.get(kind)
        or _MESSAGES.get("default")
        or "We're experiencing high demand right now. Please wait a moment and try again."
    )


def raise_as_rate_limit_error(error: Exception):
    if not is_rate_limit_error(error):
        raise error
    kind = classify_rate_limit(error)
    message = get_rate_limit_message(kind)
    logging.warning("LLM rate limit hit | kind: %s | error: %s", kind, error)
    raise LLMRateLimitError(kind, message) from error
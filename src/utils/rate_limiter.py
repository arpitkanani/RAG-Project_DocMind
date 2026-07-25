import threading
import time
from collections import deque

import yaml

from src.logger import logging

with open("config/config.yaml") as f:
    _config = yaml.safe_load(f)


class RateLimiter:
    """
    Thread-safe sliding-window rate limiter, shared across the whole process.

    Instead of firing a request and reacting to a 429 after the fact, callers
    call acquire() BEFORE making the request. If the window is full, acquire()
    simply sleeps the calling thread until a slot frees up, then proceeds --
    turning "occasionally fails" into "occasionally takes a bit longer."

    NOTE ON SCOPE: this limits requests per PYTHON PROCESS. Uvicorn's default
    single-worker mode (what you're running) means this is accurate. If you
    ever deploy with multiple worker processes (`uvicorn --workers 4`), each
    worker gets its OWN independent limiter, and the true combined request
    rate across all workers could exceed this limit -- a shared limiter across
    processes would need external state (e.g. Redis) at that point.
    """

    def __init__(self, max_requests: int, per_seconds: float):
        self.max_requests = max_requests
        self.per_seconds = per_seconds
        self._calls = deque()
        self._lock = threading.Lock()

    def acquire(self, weight: int = 1):
        """Block until `weight` request-slots are available, then consume them."""
        weight = max(1, weight)
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] > self.per_seconds:
                    self._calls.popleft()

                if len(self._calls) + weight <= self.max_requests:
                    for _ in range(weight):
                        self._calls.append(now)
                    return

                wait_time = self.per_seconds - (now - self._calls[0]) + 0.1

            logging.info(
                "Rate limiter: waiting %.1fs for capacity (weight=%d)",
                wait_time,
                weight,
            )
            time.sleep(max(wait_time, 0.1))


# ---------------------------------------------------------------------------
# LLM (Groq/Google) proactive rate limiter.
#
# Embeddings are now local (BGE via sentence-transformers) and no longer
# need rate limiting at all. This limiter protects the LLM inference calls
# instead, which DO still go through an external provider (Groq's free
# tier: 30 requests/minute, shared across every user of this app). Capped
# below the real limit (see config.yaml -> llm.max_requests_per_minute) to
# leave a safety margin.
#
# When you switch providers (groq -> google) later, just update
# max_requests_per_minute in config.yaml to match the new provider's
# per-minute limit -- no code change needed here.
# ---------------------------------------------------------------------------
_llm_rpm_limit = _config.get("llm", {}).get("max_requests_per_minute", 25)
llm_rate_limiter = RateLimiter(max_requests=_llm_rpm_limit, per_seconds=60)


# ---------------------------------------------------------------------------
# Reactive rate-limit classification -- provider-agnostic.
#
# Even with the proactive limiter above, a 429 can still occasionally slip
# through (right after a restart, multi-process drift, or a daily/token
# quota rather than a per-minute one). This section classifies WHICH kind
# of limit was hit, purely from config-defined text patterns matched
# against the error message -- so switching providers is a config-only
# change here too.
# ---------------------------------------------------------------------------
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
    """
    Raised when the LLM provider (Groq, Google, etc.) returns a
    rate-limit or quota error. Carries a `kind` (rpm/tpm/rpd/tpd) so the
    caller can show a precise, user-friendly message instead of a
    generic failure toast.
    """

    def __init__(self, kind: str, message: str):
        self.kind = kind
        self.message = message
        super().__init__(message)


def is_rate_limit_error(error: Exception) -> bool:
    """True if this looks like a rate-limit/quota error from ANY provider."""
    text = str(error).lower()
    return any(marker in text for marker in _GENERIC_RATE_LIMIT_MARKERS)


def classify_rate_limit(error: Exception) -> str:
    """
    Determine WHICH kind of limit was hit (rpm/tpm/rpd/tpd), purely from
    config-defined text patterns matched against the error message.
    Falls back to "rpm" (the shortest, most common window) if no more
    specific pattern matches but it's still clearly a rate-limit error.
    """
    text = str(error).lower()

    for kind, patterns in _PATTERNS.items():
        if kind == "rpm":
            continue  # rpm is the fallback, checked last
        for pattern in patterns:
            if pattern and pattern.lower() in text:
                return kind

    return "rpm"


def get_rate_limit_message(kind: str) -> str:
    """User-friendly message for a given limit kind, sourced from config."""
    return (
        _MESSAGES.get(kind)
        or _MESSAGES.get("default")
        or "We're experiencing high demand right now. Please wait a moment and try again."
    )


def raise_as_rate_limit_error(error: Exception):
    """
    Given an exception caught from an LLM call, classify it and raise
    LLMRateLimitError with the right user-facing message -- OR re-raise
    the original error unchanged if it isn't actually a rate-limit issue.
    """
    if not is_rate_limit_error(error):
        raise error

    kind = classify_rate_limit(error)
    message = get_rate_limit_message(kind)
    logging.warning("LLM rate limit hit | kind: %s | provider error: %s", kind, error)
    raise LLMRateLimitError(kind, message) from error
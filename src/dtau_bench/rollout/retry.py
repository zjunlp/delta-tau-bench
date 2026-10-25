"""Retry helpers for rollout LLM calls and transient failure handling."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


TRANSIENT_ERROR_MARKERS = (
    "serviceunavailableerror",
    "service temporarily unavailable",
    "internalservererror",
    "rate limit",
    "ratelimiterror",
    "apiconnectionerror",
    "apitimerouterror",
    "apitimeouterror",
    "timeout",
    "timed out",
    "connection error",
    "connection reset",
    "connection aborted",
    "connection refused",
    "connection closed",
    "stream disconnected",
    "broken pipe",
    "bad gateway",
    "gateway timeout",
    "temporarily unavailable",
    "server overloaded",
    "try again later",
    "error code: 429",
    "error code: 500",
    "error code: 502",
    "error code: 503",
    "error code: 504",
)


@dataclass(frozen=True)
class RetryConfig:
    """Configuration for dtau_bench-managed retry loops."""

    max_attempts: int = 50
    initial_delay: float = 2.0
    max_delay: float = 60.0
    total_timeout: Optional[float] = 1200.0
    jitter: bool = True


def is_transient_error(error: Any) -> bool:
    """Return True when an error string/exception looks recoverable."""
    if error is None:
        return False
    text = f"{type(error).__name__}: {error}".lower()
    return any(marker in text for marker in TRANSIENT_ERROR_MARKERS)


def compute_retry_delay(config: RetryConfig, attempt_index: int) -> float:
    """Compute exponential-backoff delay for a failed attempt.

    attempt_index is 1-based for the failed attempt that is about to be retried.
    """
    delay = min(config.max_delay, config.initial_delay * (2 ** max(attempt_index - 1, 0)))
    if config.jitter and delay > 0:
        delay = random.uniform(delay * 0.5, delay)
    return delay


def call_with_retry(
    fn: Callable[[], T],
    *,
    config: RetryConfig,
    operation: str,
    is_retryable: Callable[[BaseException], bool] = is_transient_error,
) -> T:
    """Call fn with dtau_bench-managed retry for transient failures."""
    max_attempts = max(1, int(config.max_attempts))
    start = time.monotonic()
    last_error: Optional[BaseException] = None

    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            retryable = is_retryable(exc)
            elapsed = time.monotonic() - start
            attempts_left = attempt < max_attempts
            within_budget = (
                config.total_timeout is None
                or elapsed < float(config.total_timeout)
            )

            if not retryable or not attempts_left or not within_budget:
                logger.error(
                    "%s failed after attempt %s/%s (retryable=%s, elapsed=%.1fs): %s",
                    operation,
                    attempt,
                    max_attempts,
                    retryable,
                    elapsed,
                    exc,
                )
                raise

            delay = compute_retry_delay(config, attempt)
            if config.total_timeout is not None:
                remaining = float(config.total_timeout) - elapsed
                delay = max(0.0, min(delay, remaining))

            logger.warning(
                "%s transient failure on attempt %s/%s; retrying in %.1fs: %s",
                operation,
                attempt,
                max_attempts,
                delay,
                exc,
            )
            time.sleep(delay)

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{operation} failed without an exception")

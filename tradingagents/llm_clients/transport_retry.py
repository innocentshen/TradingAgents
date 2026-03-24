import time

import httpx


DEFAULT_TRANSPORT_MAX_RETRIES = 4
DEFAULT_TRANSPORT_RETRY_BACKOFF = 1.0
DEFAULT_TRANSPORT_RETRY_BACKOFF_MULTIPLIER = 2.0
DEFAULT_TRANSPORT_RETRY_MAX_BACKOFF = 12.0

RETRYABLE_EXCEPTION_NAME_MARKERS = (
    "transporterror",
    "timeoutexception",
    "remoteprotocolerror",
    "serviceunavailable",
    "internalservererror",
    "deadlineexceeded",
    "resourceexhausted",
    "toomanyrequests",
    "apiconnectionerror",
    "ratelimiterror",
)

RETRYABLE_MESSAGE_MARKERS = (
    "server disconnected without sending a response",
    "connection reset",
    "connection aborted",
    "connection refused",
    "connection dropped",
    "temporarily unavailable",
    "deadline exceeded",
    "timed out",
    "timeout",
    "503",
    "502",
    "504",
    "429",
    "rate limit",
    "resource exhausted",
    "remote protocol error",
    "connection error",
)

NON_RETRYABLE_MESSAGE_MARKERS = (
    "api key not valid",
    "permission denied",
    "forbidden",
    "invalid argument",
    "unsupported",
    "unauthenticated",
    "authentication",
    "incorrect api key",
    "401",
    "403",
)


def is_retryable_transport_exception(exc: Exception) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True

    message = str(exc).lower()
    if any(marker in message for marker in NON_RETRYABLE_MESSAGE_MARKERS):
        return False

    exc_name = exc.__class__.__name__.lower()
    if any(marker in exc_name for marker in RETRYABLE_EXCEPTION_NAME_MARKERS):
        return True

    return any(marker in message for marker in RETRYABLE_MESSAGE_MARKERS)


def retry_delay_seconds(
    attempt: int,
    backoff: float = DEFAULT_TRANSPORT_RETRY_BACKOFF,
    multiplier: float = DEFAULT_TRANSPORT_RETRY_BACKOFF_MULTIPLIER,
    max_backoff: float = DEFAULT_TRANSPORT_RETRY_MAX_BACKOFF,
) -> float:
    delay = max(0.0, backoff) * (max(1.0, multiplier) ** max(0, attempt - 1))
    return min(delay, max(0.0, max_backoff))


def sleep_before_retry(
    attempt: int,
    backoff: float = DEFAULT_TRANSPORT_RETRY_BACKOFF,
    multiplier: float = DEFAULT_TRANSPORT_RETRY_BACKOFF_MULTIPLIER,
    max_backoff: float = DEFAULT_TRANSPORT_RETRY_MAX_BACKOFF,
) -> None:
    delay = retry_delay_seconds(attempt, backoff, multiplier, max_backoff)
    if delay > 0:
        time.sleep(delay)

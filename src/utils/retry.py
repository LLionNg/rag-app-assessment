from __future__ import annotations

from openai import APIConnectionError, APIStatusError, APITimeoutError

# Overload, conflict and rate-limit are worth repeating; other 4xx are not.
_RETRYABLE_STATUS = frozenset({408, 409, 429})


def is_retryable_openai_error(exc: Exception) -> bool:
    """Whether an OpenAI-SDK failure is transient.

    Retrying a 401 or a malformed request only delays the error the caller
    needs to see, so only connection faults, timeouts and server-side or
    throttling responses are repeated.
    """
    if isinstance(exc, APIConnectionError | APITimeoutError | TimeoutError):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in _RETRYABLE_STATUS or exc.status_code >= 500
    return False

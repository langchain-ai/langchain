"""Shared retry utilities for agent middleware.

This module contains common constants, utilities, and logic used by both
model and tool retry middleware implementations.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Literal

# Type aliases
RetryOn = tuple[type[Exception], ...] | Callable[[Exception], bool]
"""Type for specifying which exceptions to retry on.

Can be either:
- A tuple of exception types to retry on (based on `isinstance` checks)
- A callable that takes an exception and returns `True` if it should be retried
"""

OnFailure = Literal["error", "continue"] | Callable[[Exception], str]
"""Type for specifying failure handling behavior.

Can be either:
- A literal action string (`'error'` or `'continue'`)
    - `'error'`: Re-raise the exception, stopping agent execution.
    - `'continue'`: Inject a message with the error details, allowing the agent to continue.
       For tool retries, a `ToolMessage` with the error details will be injected.
       For model retries, an `AIMessage` with the error details will be returned.
- A callable that takes an exception and returns a string for error message content
"""


def validate_retry_params(
    max_retries: int,
    initial_delay: float,
    max_delay: float,
    backoff_factor: float,
) -> None:
    """Validate retry parameters.

    Args:
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay in seconds between retries.
        backoff_factor: Multiplier for exponential backoff.

    Raises:
        ValueError: If any parameter is invalid (negative values).
    """
    if max_retries < 0:
        msg = "max_retries must be >= 0"
        raise ValueError(msg)
    if initial_delay < 0:
        msg = "initial_delay must be >= 0"
        raise ValueError(msg)
    if max_delay < 0:
        msg = "max_delay must be >= 0"
        raise ValueError(msg)
    if backoff_factor < 0:
        msg = "backoff_factor must be >= 0"
        raise ValueError(msg)


def should_retry_exception(
    exc: Exception,
    retry_on: RetryOn,
) -> bool:
    """Check if an exception should trigger a retry.

    Args:
        exc: The exception that occurred.
        retry_on: Either a tuple of exception types to retry on, or a callable
            that takes an exception and returns `True` if it should be retried.

    Returns:
        `True` if the exception should be retried, `False` otherwise.
    """
    if callable(retry_on):
        return retry_on(exc)
    return isinstance(exc, retry_on)


def calculate_delay(
    retry_number: int,
    *,
    backoff_factor: float,
    initial_delay: float,
    max_delay: float,
    jitter: bool,
) -> float:
    """Calculate delay for a retry attempt with exponential backoff and optional jitter.

    Args:
        retry_number: The retry attempt number (0-indexed).
        backoff_factor: Multiplier for exponential backoff.

            Set to `0.0` for constant delay.
        initial_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay in seconds between retries.

            Caps exponential backoff growth.
        jitter: Whether to add random jitter to delay to avoid thundering herd.

    Returns:
        Delay in seconds before next retry.
    """
    if backoff_factor == 0.0:
        delay = initial_delay
    else:
        delay = initial_delay * (backoff_factor**retry_number)

    # Cap at max_delay
    delay = min(delay, max_delay)

    if jitter and delay > 0:
        jitter_amount = delay * 0.25  # ±25% jitter
        delay = delay + random.uniform(-jitter_amount, jitter_amount)  # noqa: S311
        # Ensure delay is not negative after jitter
        delay = max(0, delay)

    return delay

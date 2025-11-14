"""Model retry middleware for agents."""

from __future__ import annotations

import asyncio
import random
import time
from typing import TYPE_CHECKING, Literal

from langchain.agents.middleware.types import AgentMiddleware, ModelCallResult

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelRequest, ModelResponse


class ModelRetryMiddleware(AgentMiddleware):
    """Middleware that automatically retries failed model calls with configurable backoff.

    Uses `model.with_fallbacks()` under the hood. Supports retrying on specific exceptions
    and exponential backoff.

    Examples:
        Basic usage with default settings (2 retries, exponential backoff):

        ```python
        from langchain.agents import create_agent
        from langchain.agents.middleware import ModelRetryMiddleware

        agent = create_agent(model, tools=[search_tool], middleware=[ModelRetryMiddleware()])
        ```

        Retry specific exceptions only:

        ```python
        from requests.exceptions import RequestException, Timeout

        retry = ModelRetryMiddleware(
            max_retries=4,
            retry_on=(RequestException, Timeout),
            backoff_factor=1.5,
        )
        ```

        Custom exception filtering:

        ```python
        from requests.exceptions import HTTPError


        def should_retry(exc: Exception) -> bool:
            # Only retry on 5xx errors
            if isinstance(exc, HTTPError):
                return 500 <= exc.status_code < 600
            return False


        retry = ModelRetryMiddleware(
            max_retries=3,
            retry_on=should_retry,
        )
        ```

        Constant backoff (no exponential growth):

        ```python
        retry = ModelRetryMiddleware(
            max_retries=5,
            backoff_factor=0.0,  # No exponential growth
            initial_delay=2.0,  # Always wait 2 seconds
        )
        ```

        Raise exception on failure:

        ```python
        retry = ModelRetryMiddleware(
            max_retries=2,
            on_failure="raise",  # Re-raise exception instead of returning default
        )
        ```
    """

    def __init__(
        self,
        *,
        max_retries: int = 2,
        retry_on: tuple[type[Exception], ...] | Callable[[Exception], bool] = (Exception,),
        on_failure: Literal["raise"] = "raise",
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
    ) -> None:
        """Initialize `ModelRetryMiddleware`.

        Args:
            max_retries: Maximum number of retry attempts after the initial call.
                Default is `2` retries (`3` total attempts). Must be `>= 0`.
            retry_on: Either a tuple of exception types to retry on, or a callable
                that takes an exception and returns `True` if it should be retried.

                Default is to retry on all exceptions.
            on_failure: Behavior when all retries are exhausted.

                Currently only `'raise'` is supported (re-raises the exception).
            backoff_factor: Multiplier for exponential backoff.

                Each retry waits `initial_delay * (backoff_factor ** retry_number)`
                seconds.

                Set to `0.0` for constant delay.
            initial_delay: Initial delay in seconds before first retry.
            max_delay: Maximum delay in seconds between retries.

                Caps exponential backoff growth.
            jitter: Whether to add random jitter (`±25%`) to delay to avoid thundering herd.

        Raises:
            ValueError: If `max_retries < 0` or delays are negative.
        """
        super().__init__()

        # Validate parameters
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
        if on_failure != "raise":
            msg = "Currently only on_failure='raise' is supported"
            raise ValueError(msg)

        self.max_retries = max_retries
        self.tools = []  # No additional tools registered by this middleware
        self.retry_on = retry_on
        self.on_failure = on_failure
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.jitter = jitter

    def _should_retry_exception(self, exc: Exception) -> bool:
        """Check if the exception should trigger a retry.

        Args:
            exc: The exception that occurred.

        Returns:
            `True` if the exception should be retried, `False` otherwise.
        """
        if callable(self.retry_on):
            return self.retry_on(exc)
        return isinstance(exc, self.retry_on)

    def _calculate_delay(self, retry_number: int) -> float:
        """Calculate delay for the given retry attempt.

        Args:
            retry_number: The retry attempt number (0-indexed).

        Returns:
            Delay in seconds before next retry.
        """
        if self.backoff_factor == 0.0:
            delay = self.initial_delay
        else:
            delay = self.initial_delay * (self.backoff_factor**retry_number)

        # Cap at max_delay
        delay = min(delay, self.max_delay)

        if self.jitter and delay > 0:
            jitter_amount = delay * 0.25
            delay = delay + random.uniform(-jitter_amount, jitter_amount)  # noqa: S311
            # Ensure delay is not negative after jitter
            delay = max(0, delay)

        return delay

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Intercept model execution and retry on failure.

        Args:
            request: Model request with model, messages, state, and runtime.
            handler: Callable to execute the model (can be called multiple times).

        Returns:
            `ModelResponse` or `AIMessage` (the final result).

        Raises:
            Exception: If all retries are exhausted and `on_failure='raise'`.
        """
        # Initial attempt + retries
        for attempt in range(self.max_retries + 1):
            try:
                return handler(request)
            except Exception as exc:
                # Check if we should retry this exception
                if not self._should_retry_exception(exc):
                    # Exception is not retryable, raise immediately
                    raise

                # Check if we have more retries left
                if attempt < self.max_retries:
                    # Calculate and apply backoff delay
                    delay = self._calculate_delay(attempt)
                    if delay > 0:
                        time.sleep(delay)
                    # Continue to next retry
                else:
                    # No more retries, raise the exception
                    raise

        # Unreachable: loop always returns via handler success or raises
        msg = "Unexpected: retry loop completed without returning"
        raise RuntimeError(msg)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Intercept and control async model execution with retry logic.

        Args:
            request: Model request with model, messages, state, and runtime.
            handler: Async callable to execute the model and returns `ModelResponse`.

        Returns:
            `ModelResponse` or `AIMessage` (the final result).

        Raises:
            Exception: If all retries are exhausted and `on_failure='raise'`.
        """
        # Initial attempt + retries
        for attempt in range(self.max_retries + 1):
            try:
                return await handler(request)
            except Exception as exc:
                # Check if we should retry this exception
                if not self._should_retry_exception(exc):
                    # Exception is not retryable, raise immediately
                    raise

                # Check if we have more retries left
                if attempt < self.max_retries:
                    # Calculate and apply backoff delay
                    delay = self._calculate_delay(attempt)
                    if delay > 0:
                        await asyncio.sleep(delay)
                    # Continue to next retry
                else:
                    # No more retries, raise the exception
                    raise

        # Unreachable: loop always returns via handler success or raises
        msg = "Unexpected: retry loop completed without returning"
        raise RuntimeError(msg)

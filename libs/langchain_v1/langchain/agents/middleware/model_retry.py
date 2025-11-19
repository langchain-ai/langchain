"""Model retry middleware for agents."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage

from langchain.agents.middleware._retry import (
    OnFailure,
    RetryOn,
    calculate_delay,
    should_retry_exception,
    validate_retry_params,
)
from langchain.agents.middleware.types import AgentMiddleware, ModelResponse

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelRequest


class ModelRetryMiddleware(AgentMiddleware):
    """Middleware that automatically retries failed model calls with configurable backoff.

    Supports retrying on specific exceptions and exponential backoff.

    Examples:
        !!! example "Basic usage with default settings (2 retries, exponential backoff)"

            ```python
            from langchain.agents import create_agent
            from langchain.agents.middleware import ModelRetryMiddleware

            agent = create_agent(model, tools=[search_tool], middleware=[ModelRetryMiddleware()])
            ```

        !!! example "Retry specific exceptions only"

            ```python
            from anthropic import RateLimitError
            from openai import APITimeoutError

            retry = ModelRetryMiddleware(
                max_retries=4,
                retry_on=(APITimeoutError, RateLimitError),
                backoff_factor=1.5,
            )
            ```

        !!! example "Custom exception filtering"

            ```python
            from anthropic import APIStatusError


            def should_retry(exc: Exception) -> bool:
                # Only retry on 5xx errors
                if isinstance(exc, APIStatusError):
                    return 500 <= exc.status_code < 600
                return False


            retry = ModelRetryMiddleware(
                max_retries=3,
                retry_on=should_retry,
            )
            ```

        !!! example "Custom error handling"

            ```python
            def format_error(exc: Exception) -> str:
                return "Model temporarily unavailable. Please try again later."


            retry = ModelRetryMiddleware(
                max_retries=4,
                on_failure=format_error,
            )
            ```

        !!! example "Constant backoff (no exponential growth)"

            ```python
            retry = ModelRetryMiddleware(
                max_retries=5,
                backoff_factor=0.0,  # No exponential growth
                initial_delay=2.0,  # Always wait 2 seconds
            )
            ```

        !!! example "Raise exception on failure"

            ```python
            retry = ModelRetryMiddleware(
                max_retries=2,
                on_failure="error",  # Re-raise exception instead of returning message
            )
            ```
    """

    def __init__(
        self,
        *,
        max_retries: int = 2,
        retry_on: RetryOn = (Exception,),
        on_failure: OnFailure = "continue",
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
    ) -> None:
        """Initialize `ModelRetryMiddleware`.

        Args:
            max_retries: Maximum number of retry attempts after the initial call.

                Must be `>= 0`.
            retry_on: Either a tuple of exception types to retry on, or a callable
                that takes an exception and returns `True` if it should be retried.

                Default is to retry on all exceptions.
            on_failure: Behavior when all retries are exhausted.

                Options:

                - `'continue'`: Return an `AIMessage` with error details,
                    allowing the agent to continue with an error response.
                - `'error'`: Re-raise the exception, stopping agent execution.
                - **Custom callable:** Function that takes the exception and returns a
                    string for the `AIMessage` content, allowing custom error
                    formatting.
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
        validate_retry_params(max_retries, initial_delay, max_delay, backoff_factor)

        self.max_retries = max_retries
        self.tools = []  # No additional tools registered by this middleware
        self.retry_on = retry_on
        self.on_failure = on_failure
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.jitter = jitter

    def _format_failure_message(self, exc: Exception, attempts_made: int) -> AIMessage:
        """Format the failure message when retries are exhausted.

        Args:
            exc: The exception that caused the failure.
            attempts_made: Number of attempts actually made.

        Returns:
            `AIMessage` with formatted error message.
        """
        exc_type = type(exc).__name__
        exc_msg = str(exc)
        attempt_word = "attempt" if attempts_made == 1 else "attempts"
        content = (
            f"Model call failed after {attempts_made} {attempt_word} with {exc_type}: {exc_msg}"
        )
        return AIMessage(content=content)

    def _handle_failure(self, exc: Exception, attempts_made: int) -> ModelResponse:
        """Handle failure when all retries are exhausted.

        Args:
            exc: The exception that caused the failure.
            attempts_made: Number of attempts actually made.

        Returns:
            `ModelResponse` with error details.

        Raises:
            Exception: If `on_failure` is `'error'`, re-raises the exception.
        """
        if self.on_failure == "error":
            raise exc

        if callable(self.on_failure):
            content = self.on_failure(exc)
            ai_msg = AIMessage(content=content)
        else:
            ai_msg = self._format_failure_message(exc, attempts_made)

        return ModelResponse(result=[ai_msg])

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | AIMessage:
        """Intercept model execution and retry on failure.

        Args:
            request: Model request with model, messages, state, and runtime.
            handler: Callable to execute the model (can be called multiple times).

        Returns:
            `ModelResponse` or `AIMessage` (the final result).
        """
        # Initial attempt + retries
        for attempt in range(self.max_retries + 1):
            try:
                return handler(request)
            except Exception as exc:  # noqa: BLE001
                attempts_made = attempt + 1  # attempt is 0-indexed

                # Check if we should retry this exception
                if not should_retry_exception(exc, self.retry_on):
                    # Exception is not retryable, handle failure immediately
                    return self._handle_failure(exc, attempts_made)

                # Check if we have more retries left
                if attempt < self.max_retries:
                    # Calculate and apply backoff delay
                    delay = calculate_delay(
                        attempt,
                        backoff_factor=self.backoff_factor,
                        initial_delay=self.initial_delay,
                        max_delay=self.max_delay,
                        jitter=self.jitter,
                    )
                    if delay > 0:
                        time.sleep(delay)
                    # Continue to next retry
                else:
                    # No more retries, handle failure
                    return self._handle_failure(exc, attempts_made)

        # Unreachable: loop always returns via handler success or _handle_failure
        msg = "Unexpected: retry loop completed without returning"
        raise RuntimeError(msg)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage:
        """Intercept and control async model execution with retry logic.

        Args:
            request: Model request with model, messages, state, and runtime.
            handler: Async callable to execute the model and returns `ModelResponse`.

        Returns:
            `ModelResponse` or `AIMessage` (the final result).
        """
        # Initial attempt + retries
        for attempt in range(self.max_retries + 1):
            try:
                return await handler(request)
            except Exception as exc:  # noqa: BLE001
                attempts_made = attempt + 1  # attempt is 0-indexed

                # Check if we should retry this exception
                if not should_retry_exception(exc, self.retry_on):
                    # Exception is not retryable, handle failure immediately
                    return self._handle_failure(exc, attempts_made)

                # Check if we have more retries left
                if attempt < self.max_retries:
                    # Calculate and apply backoff delay
                    delay = calculate_delay(
                        attempt,
                        backoff_factor=self.backoff_factor,
                        initial_delay=self.initial_delay,
                        max_delay=self.max_delay,
                        jitter=self.jitter,
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)
                    # Continue to next retry
                else:
                    # No more retries, handle failure
                    return self._handle_failure(exc, attempts_made)

        # Unreachable: loop always returns via handler success or _handle_failure
        msg = "Unexpected: retry loop completed without returning"
        raise RuntimeError(msg)

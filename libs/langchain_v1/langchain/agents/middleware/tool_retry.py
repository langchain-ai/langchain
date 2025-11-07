"""Tool retry middleware for agents."""

from __future__ import annotations

import asyncio
import random
import time
from typing import TYPE_CHECKING, Literal

from langchain_core.messages import ToolMessage

from langchain.agents.middleware.types import AgentMiddleware

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.types import Command

    from langchain.agents.middleware.types import ToolCallRequest
    from langchain.tools import BaseTool


class ToolRetryMiddleware(AgentMiddleware):
    """Middleware that automatically retries failed tool calls with configurable backoff.

    Supports retrying on specific exceptions and exponential backoff.

    Examples:
        Basic usage with default settings (2 retries, exponential backoff):
        ```python
        from langchain.agents import create_agent
        from langchain.agents.middleware import ToolRetryMiddleware

        agent = create_agent(model, tools=[search_tool], middleware=[ToolRetryMiddleware()])
        ```

        Retry specific exceptions only:
        ```python
        from requests.exceptions import RequestException, Timeout

        retry = ToolRetryMiddleware(
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


        retry = ToolRetryMiddleware(
            max_retries=3,
            retry_on=should_retry,
        )
        ```

        Apply to specific tools with custom error handling:
        ```python
        def format_error(exc: Exception) -> str:
            return "Database temporarily unavailable. Please try again later."


        retry = ToolRetryMiddleware(
            max_retries=4,
            tools=["search_database"],
            on_failure=format_error,
        )
        ```

        Apply to specific tools using BaseTool instances:
        ```python
        from langchain_core.tools import tool


        @tool
        def search_database(query: str) -> str:
            '''Search the database.'''
            return results


        retry = ToolRetryMiddleware(
            max_retries=4,
            tools=[search_database],  # Pass BaseTool instance
        )
        ```

        Constant backoff (no exponential growth):
        ```python
        retry = ToolRetryMiddleware(
            max_retries=5,
            backoff_factor=0.0,  # No exponential growth
            initial_delay=2.0,  # Always wait 2 seconds
        )
        ```

        Raise exception on failure:
        ```python
        retry = ToolRetryMiddleware(
            max_retries=2,
            on_failure="raise",  # Re-raise exception instead of returning message
        )
        ```
    """

    def __init__(
        self,
        *,
        max_retries: int = 2,
        tools: list[BaseTool | str] | None = None,
        retry_on: tuple[type[Exception], ...] | Callable[[Exception], bool] = (Exception,),
        on_failure: (
            Literal["raise", "return_message"] | Callable[[Exception], str]
        ) = "return_message",
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
    ) -> None:
        """Initialize ToolRetryMiddleware.

        Args:
            max_retries: Maximum number of retry attempts after the initial call.
                Default is 2 retries (3 total attempts). Must be >= 0.
            tools: Optional list of tools or tool names to apply retry logic to.
                Can be a list of `BaseTool` instances or tool name strings.
                If `None`, applies to all tools. Default is `None`.
            retry_on: Either a tuple of exception types to retry on, or a callable
                that takes an exception and returns `True` if it should be retried.
                Default is to retry on all exceptions.
            on_failure: Behavior when all retries are exhausted. Options:
                - `"return_message"` (default): Return a ToolMessage with error details,
                  allowing the LLM to handle the failure and potentially recover.
                - `"raise"`: Re-raise the exception, stopping agent execution.
                - Custom callable: Function that takes the exception and returns a string
                  for the ToolMessage content, allowing custom error formatting.
            backoff_factor: Multiplier for exponential backoff. Each retry waits
                `initial_delay * (backoff_factor ** retry_number)` seconds.
                Set to 0.0 for constant delay. Default is 2.0.
            initial_delay: Initial delay in seconds before first retry. Default is 1.0.
            max_delay: Maximum delay in seconds between retries. Caps exponential
                backoff growth. Default is 60.0.
            jitter: Whether to add random jitter (±25%) to delay to avoid thundering herd.
                Default is `True`.

        Raises:
            ValueError: If max_retries < 0 or delays are negative.
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

        self.max_retries = max_retries

        # Extract tool names from BaseTool instances or strings
        self._tool_filter: list[str] | None
        if tools is not None:
            self._tool_filter = [tool.name if not isinstance(tool, str) else tool for tool in tools]
        else:
            self._tool_filter = None

        self.tools = []  # No additional tools registered by this middleware
        self.retry_on = retry_on
        self.on_failure = on_failure
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.jitter = jitter

    def _should_retry_tool(self, tool_name: str) -> bool:
        """Check if retry logic should apply to this tool.

        Args:
            tool_name: Name of the tool being called.

        Returns:
            `True` if retry logic should apply, `False` otherwise.
        """
        if self._tool_filter is None:
            return True
        return tool_name in self._tool_filter

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

    def _format_failure_message(self, tool_name: str, exc: Exception, attempts_made: int) -> str:
        """Format the failure message when retries are exhausted.

        Args:
            tool_name: Name of the tool that failed.
            exc: The exception that caused the failure.
            attempts_made: Number of attempts actually made.

        Returns:
            Formatted error message string.
        """
        exc_type = type(exc).__name__
        attempt_word = "attempt" if attempts_made == 1 else "attempts"
        return f"Tool '{tool_name}' failed after {attempts_made} {attempt_word} with {exc_type}"

    def _handle_failure(
        self, tool_name: str, tool_call_id: str | None, exc: Exception, attempts_made: int
    ) -> ToolMessage:
        """Handle failure when all retries are exhausted.

        Args:
            tool_name: Name of the tool that failed.
            tool_call_id: ID of the tool call (may be None).
            exc: The exception that caused the failure.
            attempts_made: Number of attempts actually made.

        Returns:
            ToolMessage with error details.

        Raises:
            Exception: If on_failure is "raise", re-raises the exception.
        """
        if self.on_failure == "raise":
            raise exc

        if callable(self.on_failure):
            content = self.on_failure(exc)
        else:
            content = self._format_failure_message(tool_name, exc, attempts_made)

        return ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            name=tool_name,
            status="error",
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Intercept tool execution and retry on failure.

        Args:
            request: Tool call request with call dict, BaseTool, state, and runtime.
            handler: Callable to execute the tool (can be called multiple times).

        Returns:
            ToolMessage or Command (the final result).
        """
        tool_name = request.tool.name if request.tool else request.tool_call["name"]

        # Check if retry should apply to this tool
        if not self._should_retry_tool(tool_name):
            return handler(request)

        tool_call_id = request.tool_call["id"]

        # Initial attempt + retries
        for attempt in range(self.max_retries + 1):
            try:
                return handler(request)
            except Exception as exc:  # noqa: BLE001
                attempts_made = attempt + 1  # attempt is 0-indexed

                # Check if we should retry this exception
                if not self._should_retry_exception(exc):
                    # Exception is not retryable, handle failure immediately
                    return self._handle_failure(tool_name, tool_call_id, exc, attempts_made)

                # Check if we have more retries left
                if attempt < self.max_retries:
                    # Calculate and apply backoff delay
                    delay = self._calculate_delay(attempt)
                    if delay > 0:
                        time.sleep(delay)
                    # Continue to next retry
                else:
                    # No more retries, handle failure
                    return self._handle_failure(tool_name, tool_call_id, exc, attempts_made)

        # Unreachable: loop always returns via handler success or _handle_failure
        msg = "Unexpected: retry loop completed without returning"
        raise RuntimeError(msg)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Intercept and control async tool execution with retry logic.

        Args:
            request: Tool call request with call dict, BaseTool, state, and runtime.
            handler: Async callable to execute the tool and returns ToolMessage or Command.

        Returns:
            ToolMessage or Command (the final result).
        """
        tool_name = request.tool.name if request.tool else request.tool_call["name"]

        # Check if retry should apply to this tool
        if not self._should_retry_tool(tool_name):
            return await handler(request)

        tool_call_id = request.tool_call["id"]

        # Initial attempt + retries
        for attempt in range(self.max_retries + 1):
            try:
                return await handler(request)
            except Exception as exc:  # noqa: BLE001
                attempts_made = attempt + 1  # attempt is 0-indexed

                # Check if we should retry this exception
                if not self._should_retry_exception(exc):
                    # Exception is not retryable, handle failure immediately
                    return self._handle_failure(tool_name, tool_call_id, exc, attempts_made)

                # Check if we have more retries left
                if attempt < self.max_retries:
                    # Calculate and apply backoff delay
                    delay = self._calculate_delay(attempt)
                    if delay > 0:
                        await asyncio.sleep(delay)
                    # Continue to next retry
                else:
                    # No more retries, handle failure
                    return self._handle_failure(tool_name, tool_call_id, exc, attempts_made)

        # Unreachable: loop always returns via handler success or _handle_failure
        msg = "Unexpected: retry loop completed without returning"
        raise RuntimeError(msg)

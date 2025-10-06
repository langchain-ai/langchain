"""Middleware for handling tool execution errors in agents.

This module provides composable middleware for error handling, retries,
and error-to-message conversion in tool execution workflows.
"""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Union, get_args, get_origin, get_type_hints

from langchain_core.messages import ToolMessage

from langchain.agents.middleware.types import AgentMiddleware

# Import ToolResponse locally to avoid circular import
from langchain.tools.tool_node import ToolResponse

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import UnionType

    from langchain.tools.tool_node import ToolRequest, ToolResponse

logger = logging.getLogger(__name__)


# Default retriable exception types - transient errors that may succeed on retry
DEFAULT_RETRIABLE_EXCEPTIONS = (
    # Network and connection errors
    ConnectionError,
    TimeoutError,
    # HTTP client errors are typically not retriable, but these are exceptions:
    # - 429: Rate limit (temporary)
    # - 503: Service unavailable (temporary)
    # Note: Specific HTTP libraries may define their own exception types
)


def _infer_retriable_types(
    predicate: Callable[[Exception], bool],
) -> tuple[type[Exception], ...]:
    """Infer exception types from a retry predicate function's type annotations.

    Analyzes the type annotations of a predicate function to determine which
    exception types it's designed to handle for retry decisions.

    Args:
        predicate: A callable that takes an exception and returns whether to retry.
            The first parameter should be type-annotated with exception type(s).

    Returns:
        Tuple of exception types that the predicate handles. Returns (Exception,)
        if no specific type information is available.

    Raises:
        ValueError: If the predicate's annotation contains non-Exception types.
    """
    sig = inspect.signature(predicate)
    params = list(sig.parameters.values())
    if params:
        # Skip self/cls if it's a method
        if params[0].name in ["self", "cls"] and len(params) == 2:
            first_param = params[1]
        else:
            first_param = params[0]

        type_hints = get_type_hints(predicate)
        if first_param.name in type_hints:
            origin = get_origin(first_param.annotation)
            # Handle Union types
            if origin in [Union, UnionType]:  # type: ignore[has-type]
                args = get_args(first_param.annotation)
                if all(isinstance(arg, type) and issubclass(arg, Exception) for arg in args):
                    return tuple(args)
                msg = (
                    "All types in retry predicate annotation must be Exception types. "
                    "For example, `def should_retry(e: Union[TimeoutError, ConnectionError]) -> bool`. "
                    f"Got '{first_param.annotation}' instead."
                )
                raise ValueError(msg)

            # Handle single exception type
            exception_type = type_hints[first_param.name]
            if isinstance(exception_type, type) and issubclass(exception_type, Exception):
                return (exception_type,)
            msg = (
                "Retry predicate must be annotated with Exception type(s). "
                "For example, `def should_retry(e: TimeoutError) -> bool` or "
                "`def should_retry(e: Union[TimeoutError, ConnectionError]) -> bool`. "
                f"Got '{exception_type}' instead."
            )
            raise ValueError(msg)

    # No type information - return Exception for backward compatibility
    return (Exception,)


class RetryMiddleware(AgentMiddleware):
    """Retry failed tool calls with constant delay.

    This middleware catches tool execution errors and retries them up to a maximum
    number of attempts with a constant delay between retries. It operates at the
    outermost layer of middleware composition to catch all errors.

    Examples:
        Retry only network errors:

        ```python
        from langchain.agents.middleware import RetryMiddleware

        middleware = RetryMiddleware(
            max_retries=3,
            delay=2.0,
            retry_on=(TimeoutError, ConnectionError),
        )

        agent = create_agent(
            model="openai:gpt-4o",
            tools=[my_tool],
            middleware=[middleware],
        )
        ```

        Use predicate function for custom retry logic:

        ```python
        from langchain.tools.tool_node import ToolInvocationError


        def should_retry(e: Exception) -> bool:
            # Don't retry validation errors from LLM
            if isinstance(e, ToolInvocationError):
                return False
            # Retry network errors
            if isinstance(e, (TimeoutError, ConnectionError)):
                return True
            return False


        middleware = RetryMiddleware(
            max_retries=3,
            retry_on=should_retry,
        )
        ```

        Compose with error conversion:

        ```python
        from langchain.agents.middleware import (
            RetryMiddleware,
            ErrorToMessageMiddleware,
        )

        agent = create_agent(
            model="openai:gpt-4o",
            tools=[my_tool],
            middleware=[
                # Outer: retry network errors
                RetryMiddleware(
                    max_retries=3,
                    delay=2.0,
                    retry_on=(TimeoutError, ConnectionError),
                ),
                # Inner: convert validation errors to messages
                ErrorToMessageMiddleware(
                    exception_types=(ValidationError,),
                ),
            ],
        )
        ```
    """

    def __init__(
        self,
        *,
        max_retries: int = 3,
        delay: float = 1.0,
        retry_on: type[Exception]
        | tuple[type[Exception], ...]
        | Callable[[Exception], bool] = DEFAULT_RETRIABLE_EXCEPTIONS,
    ) -> None:
        """Initialize retry middleware.

        Args:
            max_retries: Maximum number of retry attempts. Total attempts will be
                max_retries + 1 (initial attempt plus retries).
            delay: Constant delay in seconds between retry attempts.
            retry_on: Specifies which exceptions should be retried. Can be:
                - **type[Exception]**: Retry only this exception type
                - **tuple[type[Exception], ...]**: Retry these exception types
                - **Callable[[Exception], bool]**: Predicate function that returns
                  True if the exception should be retried. Type annotations on the
                  callable are used to filter which exceptions are passed to it.
                Defaults to ``DEFAULT_RETRIABLE_EXCEPTIONS`` (ConnectionError, TimeoutError).
        """
        super().__init__()
        if max_retries < 0:
            msg = "max_retries must be non-negative"
            raise ValueError(msg)
        if delay < 0:
            msg = "delay must be non-negative"
            raise ValueError(msg)

        self.max_retries = max_retries
        self.delay = delay
        self._retry_on = retry_on

        # Determine which exception types to check
        if isinstance(retry_on, type) and issubclass(retry_on, Exception):
            self._retriable_types = (retry_on,)
            self._retry_predicate = None
        elif isinstance(retry_on, tuple):
            if not retry_on:
                msg = "retry_on tuple must not be empty"
                raise ValueError(msg)
            if not all(isinstance(t, type) and issubclass(t, Exception) for t in retry_on):
                msg = "All elements in retry_on tuple must be Exception types"
                raise ValueError(msg)
            self._retriable_types = retry_on
            self._retry_predicate = None
        elif callable(retry_on):
            self._retriable_types = _infer_retriable_types(retry_on)
            self._retry_predicate = retry_on
        else:
            msg = (
                "retry_on must be an Exception type, tuple of Exception types, "
                f"or callable. Got {type(retry_on)}"
            )
            raise ValueError(msg)

    def on_tool_call(
        self, request: ToolRequest
    ) -> Generator[ToolRequest, ToolResponse, ToolResponse]:
        """Retry tool execution on failures."""
        for attempt in range(1, self.max_retries + 2):  # +1 for initial, +1 for inclusive
            response = yield request

            # Success - return immediately
            if response.action == "return":
                return response

            # Error - check if we should retry
            if response.action == "raise":
                exception = response.exception
                if exception is None:
                    msg = "ToolResponse with action='raise' must have an exception"
                    raise ValueError(msg)

                # Check if this exception type is retriable
                if not isinstance(exception, self._retriable_types):
                    logger.debug(
                        "Exception %s is not retriable for tool %s",
                        type(exception).__name__,
                        request.tool_call["name"],
                    )
                    return response

                # If predicate is provided, check if we should retry
                if self._retry_predicate is not None:
                    if not self._retry_predicate(exception):
                        logger.debug(
                            "Retry predicate returned False for %s in tool %s",
                            type(exception).__name__,
                            request.tool_call["name"],
                        )
                        return response

                # Last attempt - return error
                if attempt > self.max_retries:
                    logger.debug(
                        "Max retries (%d) reached for tool %s",
                        self.max_retries,
                        request.tool_call["name"],
                    )
                    return response

                # Retry - log and delay
                logger.debug(
                    "Retrying tool %s (attempt %d/%d) after error: %s",
                    request.tool_call["name"],
                    attempt,
                    self.max_retries + 1,
                    type(exception).__name__,
                )
                time.sleep(self.delay)
                continue

        # Should never reach here
        msg = f"Unexpected control flow in RetryMiddleware for tool {request.tool_call['name']}"
        raise RuntimeError(msg)


class ErrorToMessageMiddleware(AgentMiddleware):
    """Convert specific exception types to ToolMessages.

    This middleware intercepts errors and converts them into ToolMessages that
    can be sent back to the model as feedback. This is useful for errors caused
    by invalid model inputs where the model needs feedback to correct its behavior.

    Examples:
        Convert validation errors to messages:

        ```python
        from langchain.agents.middleware import ErrorToMessageMiddleware
        from langchain.tools.tool_node import ToolInvocationError

        middleware = ErrorToMessageMiddleware(
            exception_types=(ToolInvocationError,),
            message_template="Invalid arguments: {error}. Please fix and try again.",
        )

        agent = create_agent(
            model="openai:gpt-4o",
            tools=[my_tool],
            middleware=[middleware],
        )
        ```

        Compose with retry for network errors:

        ```python
        from langchain.agents.middleware import (
            RetryMiddleware,
            ErrorToMessageMiddleware,
        )

        agent = create_agent(
            model="openai:gpt-4o",
            tools=[my_tool],
            middleware=[
                # Outer: retry all errors
                RetryMiddleware(max_retries=3),
                # Inner: convert validation errors to messages
                ErrorToMessageMiddleware(
                    exception_types=(ValidationError,),
                ),
            ],
        )
        ```
    """

    def __init__(
        self,
        *,
        exception_types: tuple[type[Exception], ...],
        message_template: str = "Error: {error}",
    ) -> None:
        """Initialize error conversion middleware.

        Args:
            exception_types: Tuple of exception types to convert to messages.
            message_template: Template string for error messages. Can use ``{error}``
                placeholder for the exception string representation.
        """
        super().__init__()
        if not exception_types:
            msg = "exception_types must not be empty"
            raise ValueError(msg)

        self.exception_types = exception_types
        self.message_template = message_template

    def on_tool_call(
        self, request: ToolRequest
    ) -> Generator[ToolRequest, ToolResponse, ToolResponse]:
        """Convert matching errors to ToolMessages."""
        response = yield request

        # Success - pass through
        if response.action == "return":
            return response

        # Error - check if we should convert
        if response.action == "raise":
            exception = response.exception
            if exception is None:
                msg = "ToolResponse with action='raise' must have an exception"
                raise ValueError(msg)

            # Check if exception type matches
            if not isinstance(exception, self.exception_types):
                return response

            # Convert to ToolMessage
            logger.debug(
                "Converting %s to ToolMessage for tool %s",
                type(exception).__name__,
                request.tool_call["name"],
            )

            error_message = self.message_template.format(error=str(exception))
            tool_message = ToolMessage(
                content=error_message,
                name=request.tool_call["name"],
                tool_call_id=request.tool_call["id"],
                status="error",
            )

            return ToolResponse(
                action="return",
                result=tool_message,
                exception=exception,  # Preserve for logging/debugging
            )

        return response

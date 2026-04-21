"""Tool retry middleware for agents."""

from __future__ import annotations

import asyncio
import time
import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any

from langchain_core.messages import ToolMessage
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.types import Command
from typing_extensions import NotRequired

from langchain.agents.middleware._progress import (
    AgentProgressStalledError,
    build_tool_failure_signature,
    default_progress_output_normalizer,
    validate_max_consecutive_steps,
)
from langchain.agents.middleware._retry import (
    OnFailure,
    RetryOn,
    calculate_delay,
    should_retry_exception,
    validate_retry_params,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    PrivateStateAttr,
    ResponseT,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ToolCallRequest
    from langchain.tools import BaseTool

_PROGRESS_SIGNATURE_KEY = "run_tool_retry_last_failure_signature"
_PROGRESS_COUNT_KEY = "run_tool_retry_consecutive_failure_count"


class ToolRetryState(AgentState[ResponseT]):
    """State schema for `ToolRetryMiddleware`.

    Adds run-scoped private fields used only when retry progress detection is enabled.
    """

    run_tool_retry_last_failure_signature: NotRequired[
        Annotated[str | None, UntrackedValue, PrivateStateAttr]
    ]
    run_tool_retry_consecutive_failure_count: NotRequired[
        Annotated[int, UntrackedValue, PrivateStateAttr]
    ]


class ToolRetryMiddleware(AgentMiddleware[ToolRetryState[ResponseT], ContextT, ResponseT]):
    """Middleware that automatically retries failed tool calls with configurable backoff.

    Supports retrying on specific exceptions and exponential backoff.

    Examples:
        !!! example "Basic usage with default settings (2 retries, exponential backoff)"

            ```python
            from langchain.agents import create_agent
            from langchain.agents.middleware import ToolRetryMiddleware

            agent = create_agent(model, tools=[search_tool], middleware=[ToolRetryMiddleware()])
            ```

        !!! example "Retry specific exceptions only"

            ```python
            from requests.exceptions import RequestException, Timeout

            retry = ToolRetryMiddleware(
                max_retries=4,
                retry_on=(RequestException, Timeout),
                backoff_factor=1.5,
            )
            ```

        !!! example "Custom exception filtering"

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

        !!! example "Apply to specific tools with custom error handling"

            ```python
            def format_error(exc: Exception) -> str:
                return "Database temporarily unavailable. Please try again later."


            retry = ToolRetryMiddleware(
                max_retries=4,
                tools=["search_database"],
                on_failure=format_error,
            )
            ```

        !!! example "Apply to specific tools using `BaseTool` instances"

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

        !!! example "Constant backoff (no exponential growth)"

            ```python
            retry = ToolRetryMiddleware(
                max_retries=5,
                backoff_factor=0.0,  # No exponential growth
                initial_delay=2.0,  # Always wait 2 seconds
            )
            ```

        !!! example "Raise exception on failure"

            ```python
            retry = ToolRetryMiddleware(
                max_retries=2,
                on_failure="error",  # Re-raise exception instead of returning message
            )
            ```

        !!! example "Stop repeated retry-exhaustion loops"

            Use this when the model keeps calling the same failing tool after retries
            have already been exhausted.

            ```python
            from langchain.agents.middleware import (
                AgentProgressStalledError,
                ToolRetryMiddleware,
            )


            retry = ToolRetryMiddleware(
                max_retries=2,
                max_consecutive_identical_failures=3,
            )

            try:
                agent.invoke({"messages": [{"role": "user", "content": "search again"}]})
            except AgentProgressStalledError as exc:
                print(exc.reason)  # "no_progress_detected"
            ```

            Stall detection is disabled by default. When enabled, it counts one failure
            per tool call after retries are exhausted. Retry attempts inside the same
            tool call do not count separately.

            Keep the default `on_failure="continue"`. Do not use
            `on_failure="error"` with this option. The constructor raises `ValueError`
            for that combination because raw exceptions stop the run before repeated
            failures can be counted.

            Do not also enable `ProgressGuardMiddleware` for the same tool unless you
            intentionally want layered detection.

            Custom tool-call wrappers should pass through both `ToolMessage` and
            `Command` results.
    """

    state_schema = ToolRetryState  # type: ignore[assignment]

    def __init__(
        self,
        *,
        max_retries: int = 2,
        tools: list[BaseTool | str] | None = None,
        retry_on: RetryOn = (Exception,),
        on_failure: OnFailure = "continue",
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
        max_consecutive_identical_failures: int | None = None,
        stall_error_normalizer: Callable[[str], str] | None = None,
    ) -> None:
        """Initialize `ToolRetryMiddleware`.

        Args:
            max_retries: Maximum number of retry attempts after the initial call.

                Must be `>= 0`.
            tools: Optional list of tools or tool names to apply retry logic to.

                Can be a list of `BaseTool` instances or tool name strings.

                If `None`, applies to all tools.
            retry_on: Either a tuple of exception types to retry on, or a callable
                that takes an exception and returns `True` if it should be retried.

                Default is to retry on all exceptions.
            on_failure: Behavior when all retries are exhausted.

                Options:

                - `'continue'`: Return a `ToolMessage` with error details,
                    allowing the LLM to handle the failure and potentially recover.
                - `'error'`: Re-raise the exception, stopping agent execution.
                - **Custom callable:** Function that takes the exception and returns a
                    string for the `ToolMessage` content, allowing custom error
                    formatting.

                **Deprecated values** (for backwards compatibility):

                - `'return_message'`: Use `'continue'` instead.
                - `'raise'`: Use `'error'` instead.
            backoff_factor: Multiplier for exponential backoff.

                Each retry waits `initial_delay * (backoff_factor ** retry_number)`
                seconds.

                Set to `0.0` for constant delay.
            initial_delay: Initial delay in seconds before first retry.
            max_delay: Maximum delay in seconds between retries.

                Caps exponential backoff growth.
            jitter: Whether to add random jitter (`±25%`) to delay to avoid thundering herd.
            max_consecutive_identical_failures: Optional threshold for stopping an
                agent loop that repeatedly exhausts retries for the same tool, same
                arguments, and same normalized final error. Must be `>= 2` when set.

                `None` disables stall detection. Retry attempts inside one tool call do
                not increment this counter; only the final failure returned to the
                model counts.
            stall_error_normalizer: Optional callable used to normalize final failure
                messages before comparing them for stall detection. The default
                normalizer strips common volatile details such as tracebacks, request
                IDs, UUIDs, timestamps, and repeated whitespace.

        Raises:
            ValueError: If `max_retries < 0`, delays are negative, the stall
                threshold is invalid, or stall detection is enabled with
                `on_failure='error'`.
        """
        super().__init__()

        # Validate parameters
        validate_retry_params(max_retries, initial_delay, max_delay, backoff_factor)
        if max_consecutive_identical_failures is not None:
            validate_max_consecutive_steps(
                max_consecutive_identical_failures,
                parameter_name="max_consecutive_identical_failures",
            )

        # Handle backwards compatibility for deprecated on_failure values
        if on_failure == "raise":  # type: ignore[comparison-overlap]
            msg = (  # type: ignore[unreachable]
                "on_failure='raise' is deprecated and will be removed in a future version. "
                "Use on_failure='error' instead."
            )
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            on_failure = "error"
        elif on_failure == "return_message":  # type: ignore[comparison-overlap]
            msg = (  # type: ignore[unreachable]
                "on_failure='return_message' is deprecated and will be removed "
                "in a future version. Use on_failure='continue' instead."
            )
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            on_failure = "continue"

        if max_consecutive_identical_failures is not None and on_failure == "error":
            msg = (
                "max_consecutive_identical_failures cannot be used with "
                "on_failure='error' because raw exceptions abort the run before "
                "repeated failure observations can be counted."
            )
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
        self.max_consecutive_identical_failures = max_consecutive_identical_failures
        self._failure_output_normalizer = (
            stall_error_normalizer or default_progress_output_normalizer
        )

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

    @staticmethod
    def _format_failure_message(tool_name: str, exc: Exception, attempts_made: int) -> str:
        """Format the failure message when retries are exhausted.

        Args:
            tool_name: Name of the tool that failed.
            exc: The exception that caused the failure.
            attempts_made: Number of attempts actually made.

        Returns:
            Formatted error message string.
        """
        exc_type = type(exc).__name__
        exc_msg = str(exc)
        attempt_word = "attempt" if attempts_made == 1 else "attempts"
        return (
            f"Tool '{tool_name}' failed after {attempts_made} {attempt_word} "
            f"with {exc_type}: {exc_msg}. Please try again."
        )

    def _handle_failure(
        self, tool_name: str, tool_call_id: str | None, exc: Exception, attempts_made: int
    ) -> ToolMessage:
        """Handle failure when all retries are exhausted.

        Args:
            tool_name: Name of the tool that failed.
            tool_call_id: ID of the tool call (may be `None`).
            exc: The exception that caused the failure.
            attempts_made: Number of attempts actually made.

        Returns:
            `ToolMessage` with error details.

        Raises:
            Exception: If `on_failure` is `'error'`, re-raises the exception.
        """
        if self.on_failure == "error":
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

    def _progress_detection_enabled(self) -> bool:
        """Check whether retry-exhaustion stall detection is enabled."""
        return self.max_consecutive_identical_failures is not None

    @staticmethod
    def _has_progress_state(state: Any) -> bool:
        """Check whether there is retry stall state to clear."""
        if not isinstance(state, Mapping):
            return False
        return state.get(_PROGRESS_SIGNATURE_KEY) is not None or bool(
            state.get(_PROGRESS_COUNT_KEY)
        )

    @staticmethod
    def _reset_progress_state_update() -> dict[str, Any]:
        """Build an update that clears retry stall state for this run."""
        return {
            _PROGRESS_SIGNATURE_KEY: None,
            _PROGRESS_COUNT_KEY: 0,
        }

    @staticmethod
    def _as_tool_message_command(
        tool_message: ToolMessage, state_update: dict[str, Any]
    ) -> Command[Any]:
        """Attach private state updates while preserving the tool message."""
        return Command(update={"messages": [tool_message], **state_update})

    @staticmethod
    def _as_command_with_state_update(
        command: Command[Any], state_update: dict[str, Any]
    ) -> Command[Any]:
        """Attach private state updates to a command with a dict update."""
        if not isinstance(command.update, Mapping):
            return command

        return Command(
            graph=command.graph,
            update={**command.update, **state_update},
            resume=command.resume,
            goto=command.goto,
        )

    def _handle_success_result(
        self,
        request: ToolCallRequest,
        result: ToolMessage | Command[Any],
    ) -> ToolMessage | Command[Any]:
        """Reset stall state after a successful observed tool result."""
        if not self._progress_detection_enabled():
            return result

        if not self._has_progress_state(request.state):
            return result

        if isinstance(result, ToolMessage) and result.status != "error":
            return self._as_tool_message_command(result, self._reset_progress_state_update())

        if isinstance(result, Command):
            messages = result.update.get("messages") if isinstance(result.update, Mapping) else None
            if isinstance(messages, list) and any(
                isinstance(message, ToolMessage) and message.status != "error"
                for message in messages
            ):
                return self._as_command_with_state_update(
                    result,
                    self._reset_progress_state_update(),
                )

        return result

    def _handle_exhausted_failure(
        self,
        request: ToolCallRequest,
        tool_name: str,
        tool_message: ToolMessage,
    ) -> ToolMessage | Command[Any]:
        """Count final retry failures and raise if they form a stall."""
        if not self._progress_detection_enabled():
            return tool_message

        state = request.state if isinstance(request.state, Mapping) else {}
        signature = build_tool_failure_signature(
            tool_name=tool_name,
            tool_args=request.tool_call["args"],
            error_message=tool_message.text,
            error_normalizer=self._failure_output_normalizer,
        )
        previous_signature = state.get(_PROGRESS_SIGNATURE_KEY)
        previous_count = state.get(_PROGRESS_COUNT_KEY, 0)
        consecutive_failures = previous_count + 1 if previous_signature == signature else 1

        max_consecutive_failures = self.max_consecutive_identical_failures
        if (
            max_consecutive_failures is not None
            and consecutive_failures >= max_consecutive_failures
        ):
            raise AgentProgressStalledError(
                consecutive_steps=consecutive_failures,
                max_consecutive_identical_steps=max_consecutive_failures,
                description=(
                    f"{tool_name}({request.tool_call['args']!r}) -> error: {tool_message.text}"
                ),
                exchange_signature=signature,
            )

        return self._as_tool_message_command(
            tool_message,
            {
                _PROGRESS_SIGNATURE_KEY: signature,
                _PROGRESS_COUNT_KEY: consecutive_failures,
            },
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Intercept tool execution and retry on failure.

        Args:
            request: Tool call request with call dict, `BaseTool`, state, and runtime.
            handler: Callable to execute the tool (can be called multiple times).

        Returns:
            `ToolMessage` or `Command` (the final result).

        Raises:
            RuntimeError: If the retry loop completes without returning. This should not happen.
        """
        tool_name = request.tool.name if request.tool else request.tool_call["name"]

        # Check if retry should apply to this tool
        if not self._should_retry_tool(tool_name):
            return handler(request)

        tool_call_id = request.tool_call["id"]

        # Initial attempt + retries
        for attempt in range(self.max_retries + 1):
            try:
                return self._handle_success_result(request, handler(request))
            except Exception as exc:
                attempts_made = attempt + 1  # attempt is 0-indexed

                # Check if we should retry this exception
                if not should_retry_exception(exc, self.retry_on):
                    # Exception is not retryable, handle failure immediately
                    tool_message = self._handle_failure(tool_name, tool_call_id, exc, attempts_made)
                    return self._handle_exhausted_failure(request, tool_name, tool_message)

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
                    tool_message = self._handle_failure(tool_name, tool_call_id, exc, attempts_made)
                    return self._handle_exhausted_failure(request, tool_name, tool_message)

        # Unreachable: loop always returns via handler success or _handle_failure
        msg = "Unexpected: retry loop completed without returning"
        raise RuntimeError(msg)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Intercept and control async tool execution with retry logic.

        Args:
            request: Tool call request with call `dict`, `BaseTool`, state, and runtime.
            handler: Async callable to execute the tool and returns `ToolMessage` or
                `Command`.

        Returns:
            `ToolMessage` or `Command` (the final result).

        Raises:
            RuntimeError: If the retry loop completes without returning. This should not happen.
        """
        tool_name = request.tool.name if request.tool else request.tool_call["name"]

        # Check if retry should apply to this tool
        if not self._should_retry_tool(tool_name):
            return await handler(request)

        tool_call_id = request.tool_call["id"]

        # Initial attempt + retries
        for attempt in range(self.max_retries + 1):
            try:
                return self._handle_success_result(request, await handler(request))
            except Exception as exc:
                attempts_made = attempt + 1  # attempt is 0-indexed

                # Check if we should retry this exception
                if not should_retry_exception(exc, self.retry_on):
                    # Exception is not retryable, handle failure immediately
                    tool_message = self._handle_failure(tool_name, tool_call_id, exc, attempts_made)
                    return self._handle_exhausted_failure(request, tool_name, tool_message)

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
                    tool_message = self._handle_failure(tool_name, tool_call_id, exc, attempts_made)
                    return self._handle_exhausted_failure(request, tool_name, tool_message)

        # Unreachable: loop always returns via handler success or _handle_failure
        msg = "Unexpected: retry loop completed without returning"
        raise RuntimeError(msg)

"""Progress guard middleware for agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Literal

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langgraph.channels.untracked_value import UntrackedValue
from typing_extensions import NotRequired, override

from langchain.agents.middleware._progress import (
    AgentProgressStalledError,
    build_progress_stalled_message,
    build_tool_exchange_signature,
    default_progress_output_normalizer,
    stable_json_dumps,
    summarize_progress_output,
    validate_max_consecutive_steps,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    PrivateStateAttr,
    ResponseT,
    ToolCallRequest,
    hook_config,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.tools import BaseTool
    from langgraph.runtime import Runtime
    from langgraph.types import Command


_INITIALIZED_KEY = "run_progress_guard_initialized"
_LAST_PROCESSED_EXCHANGE_KEY = "run_progress_guard_last_processed_exchange_id"
_LAST_SIGNATURE_KEY = "run_progress_guard_last_signature"
_CONSECUTIVE_COUNT_KEY = "run_progress_guard_consecutive_count"


@dataclass(frozen=True)
class _ProgressObservation:
    """Stable description of a completed AI/tool exchange."""

    exchange_id: str
    signature: str
    description: str


class ProgressGuardState(AgentState[ResponseT]):
    """State schema for `ProgressGuardMiddleware`.

    The middleware keeps a run-scoped ledger of the last processed AI/tool exchange and
    the current consecutive equivalent exchange streak.
    """

    run_progress_guard_initialized: NotRequired[
        Annotated[bool, UntrackedValue, PrivateStateAttr]
    ]
    run_progress_guard_last_processed_exchange_id: NotRequired[
        Annotated[str | None, UntrackedValue, PrivateStateAttr]
    ]
    run_progress_guard_last_signature: NotRequired[
        Annotated[str | None, UntrackedValue, PrivateStateAttr]
    ]
    run_progress_guard_consecutive_count: NotRequired[
        Annotated[int, UntrackedValue, PrivateStateAttr]
    ]


def _fetch_last_ai_and_tool_messages(
    messages: list[AnyMessage],
) -> tuple[AIMessage | None, list[ToolMessage]]:
    """Return the last AI message and any subsequent tool messages."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, AIMessage):
            last_ai_message = message
            tool_messages = [msg for msg in messages[index + 1 :] if isinstance(msg, ToolMessage)]
            return last_ai_message, tool_messages
    return None, []


def _build_exchange_id(message: AIMessage) -> str:
    """Build a stable identifier for the latest AI/tool exchange."""
    if message.id:
        return str(message.id)
    return stable_json_dumps(
        {
            "content": message.content,
            "tool_calls": message.tool_calls,
        }
    )


class ProgressGuardMiddleware(
    AgentMiddleware[ProgressGuardState[ResponseT], ContextT, ResponseT]
):
    """Stop agent loops that repeat the same completed tool exchange.

    Use this guard when an agent keeps making the same tool calls and receives the
    same tool outputs. It detects repeated successful outputs and repeated error
    outputs by comparing the tool name, tool arguments, output status, and normalized
    output content.

    By default, this middleware only observes completed `ToolMessage` objects that
    already exist. It does not catch raw tool exceptions. Set
    `catch_tool_exceptions=True` only when this guard should also catch monitored tool
    exceptions and convert them into observable error `ToolMessage`s.

    !!! warning

        This middleware is opt-in because repeated tool calls can be valid for polling,
        pagination, or workflows where unchanged output is expected. Scope `tools` or
        raise `max_consecutive_identical_steps` for tools that may legitimately repeat.

    Prefer `ToolRetryMiddleware(max_consecutive_identical_failures=...)` for the
    narrower case where you only need to stop repeated retry-exhausted tool failures.
    Do not enable both mechanisms for the same tool unless layered detection is
    intentional.

    Examples:
        !!! example "Stop repeated no-progress tool exchanges"

            ```python
            from langchain.agents import create_agent
            from langchain.agents.middleware import ProgressGuardMiddleware

            agent = create_agent(
                model,
                tools=[search_tool],
                middleware=[ProgressGuardMiddleware(max_consecutive_identical_steps=3)],
            )
            ```

        !!! example "End gracefully instead of raising"

            ```python
            guard = ProgressGuardMiddleware(
                max_consecutive_identical_steps=3,
                exit_behavior="end",
            )
            ```

            With `exit_behavior="end"`, the agent appends a final `AIMessage` explaining
            that no progress was detected and exits normally.

        !!! example "Catch raw tool exceptions in the guard"

            ```python
            guard = ProgressGuardMiddleware(
                max_consecutive_identical_steps=3,
                catch_tool_exceptions=True,
            )
            ```

            Without `catch_tool_exceptions=True`, raw tool exceptions still abort the
            run.

        !!! example "Normalize volatile output"

            ```python
            import re


            def normalize_search_output(output: str) -> str:
                return re.sub("request_id=[A-Za-z0-9_]+", "request_id=<id>", output)


            guard = ProgressGuardMiddleware(output_normalizer=normalize_search_output)
            ```
    """

    state_schema = ProgressGuardState  # type: ignore[assignment]

    def __init__(
        self,
        *,
        max_consecutive_identical_steps: int = 3,
        tools: list[BaseTool | str] | None = None,
        exit_behavior: Literal["error", "end"] = "error",
        output_normalizer: Callable[[str], str] | None = None,
        catch_tool_exceptions: bool = False,
    ) -> None:
        """Initialize the progress guard.

        Args:
            max_consecutive_identical_steps: Number of equivalent completed tool
                exchanges allowed before treating the loop as stalled. Must be `>= 2`.
            tools: Optional list of tools or tool names to monitor. If `None`,
                monitors every tool.
            exit_behavior: What to do once no progress is detected.

                - `'error'`: Raise `AgentProgressStalledError`.
                - `'end'`: Append a final `AIMessage` with the `no_progress_detected`
                    reason and stop the agent normally.
            output_normalizer: Optional callable that turns raw tool output into a
                stable comparison key. The default normalizer strips common volatile
                details such as tracebacks, request IDs, UUIDs, timestamps, and repeated
                whitespace.
            catch_tool_exceptions: Whether to convert monitored tool exceptions into
                `ToolMessage(status="error")`.

        Raises:
            ValueError: If `max_consecutive_identical_steps < 2` or `exit_behavior`
                is invalid.
        """
        super().__init__()

        validate_max_consecutive_steps(
            max_consecutive_identical_steps,
            parameter_name="max_consecutive_identical_steps",
        )

        if exit_behavior not in {"error", "end"}:
            msg = "exit_behavior must be 'error' or 'end'"
            raise ValueError(msg)

        self.max_consecutive_identical_steps = max_consecutive_identical_steps
        self.exit_behavior = exit_behavior
        self._output_normalizer = output_normalizer or default_progress_output_normalizer
        self.catch_tool_exceptions = catch_tool_exceptions
        self.tools = []
        self._tool_filter = (
            {tool.name if not isinstance(tool, str) else tool for tool in tools}
            if tools is not None
            else None
        )

    def _matches_tool_filter(self, tool_name: str) -> bool:
        """Check whether the progress guard should monitor the tool."""
        if self._tool_filter is None:
            return True
        return tool_name in self._tool_filter

    @staticmethod
    def _format_tool_failure_message(tool_name: str, exc: Exception) -> str:
        """Format tool exceptions as stable error messages for the model."""
        exc_type = type(exc).__name__
        summarized_error = summarize_progress_output(str(exc))
        return f"Tool '{tool_name}' failed with {exc_type}: {summarized_error}. Please try again."

    def _extract_progress_observation(
        self,
        *,
        last_ai_message: AIMessage | None,
        tool_messages: list[ToolMessage],
    ) -> _ProgressObservation | None:
        """Extract a stable signature from the latest completed tool exchange."""
        if last_ai_message is None or not last_ai_message.tool_calls:
            return None

        tool_calls: list[dict[str, Any]] = []
        tool_outputs: list[dict[str, Any]] = []
        descriptions: list[str] = []

        for tool_call in last_ai_message.tool_calls:
            tool_name = tool_call["name"]
            if not self._matches_tool_filter(tool_name):
                return None

            matching_tool_messages = [
                message for message in tool_messages if message.tool_call_id == tool_call["id"]
            ]
            if len(matching_tool_messages) != 1:
                return None
            tool_message = matching_tool_messages[0]

            raw_output = tool_message.text
            normalized_output = self._output_normalizer(raw_output)
            tool_calls.append(
                {
                    "name": tool_name,
                    "args": tool_call["args"],
                }
            )
            tool_outputs.append(
                {
                    "name": tool_message.name or tool_name,
                    "status": tool_message.status,
                    "output": normalized_output,
                }
            )
            descriptions.append(
                f"{tool_name}({stable_json_dumps(tool_call['args'])}) -> "
                f"{tool_message.status}: {summarize_progress_output(raw_output)}"
            )

        signature = build_tool_exchange_signature(
            tool_calls=tool_calls,
            tool_outputs=tool_outputs,
        )
        description = "; ".join(descriptions)

        return _ProgressObservation(
            exchange_id=_build_exchange_id(last_ai_message),
            signature=signature,
            description=description,
        )

    @staticmethod
    def _reset_state(*, exchange_id: str | None) -> dict[str, Any]:
        """Clear the current progress-stalled streak."""
        return {
            _LAST_PROCESSED_EXCHANGE_KEY: exchange_id,
            _LAST_SIGNATURE_KEY: None,
            _CONSECUTIVE_COUNT_KEY: 0,
        }

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Optionally convert monitored tool exceptions into error `ToolMessage`s."""
        tool_name = request.tool.name if request.tool is not None else request.tool_call["name"]
        if not self.catch_tool_exceptions or not self._matches_tool_filter(tool_name):
            return handler(request)

        try:
            return handler(request)
        except Exception as exc:
            return ToolMessage(
                content=self._format_tool_failure_message(tool_name, exc),
                tool_call_id=request.tool_call["id"],
                name=tool_name,
                status="error",
            )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Async version of `wrap_tool_call`."""
        tool_name = request.tool.name if request.tool is not None else request.tool_call["name"]
        if not self.catch_tool_exceptions or not self._matches_tool_filter(tool_name):
            return await handler(request)

        try:
            return await handler(request)
        except Exception as exc:
            return ToolMessage(
                content=self._format_tool_failure_message(tool_name, exc),
                tool_call_id=request.tool_call["id"],
                name=tool_name,
                status="error",
            )

    @hook_config(can_jump_to=["end"])
    @override
    def before_model(
        self,
        state: ProgressGuardState[ResponseT],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Inspect the latest completed tool exchange before the next model call."""
        messages = state.get("messages", [])
        last_ai_message, tool_messages = _fetch_last_ai_and_tool_messages(messages)
        current_exchange_id = (
            _build_exchange_id(last_ai_message) if last_ai_message is not None else None
        )

        if not state.get(_INITIALIZED_KEY, False):
            return {
                _INITIALIZED_KEY: True,
                **self._reset_state(exchange_id=current_exchange_id),
            }

        if current_exchange_id is None:
            return self._reset_state(exchange_id=None)

        if current_exchange_id == state.get(_LAST_PROCESSED_EXCHANGE_KEY):
            return None

        observation = self._extract_progress_observation(
            last_ai_message=last_ai_message,
            tool_messages=tool_messages,
        )
        if observation is None:
            return self._reset_state(exchange_id=current_exchange_id)

        previous_signature = state.get(_LAST_SIGNATURE_KEY)
        previous_count = state.get(_CONSECUTIVE_COUNT_KEY, 0)
        consecutive_steps = previous_count + 1 if previous_signature == observation.signature else 1

        if consecutive_steps >= self.max_consecutive_identical_steps:
            if self.exit_behavior == "error":
                raise AgentProgressStalledError(
                    consecutive_steps=consecutive_steps,
                    max_consecutive_identical_steps=self.max_consecutive_identical_steps,
                    description=observation.description,
                    exchange_signature=observation.signature,
                )

            return {
                "jump_to": "end",
                "messages": [
                    AIMessage(
                        content=build_progress_stalled_message(
                            consecutive_steps=consecutive_steps,
                            description=observation.description,
                        )
                    )
                ],
                _LAST_PROCESSED_EXCHANGE_KEY: observation.exchange_id,
                _LAST_SIGNATURE_KEY: observation.signature,
                _CONSECUTIVE_COUNT_KEY: consecutive_steps,
            }

        return {
            _LAST_PROCESSED_EXCHANGE_KEY: observation.exchange_id,
            _LAST_SIGNATURE_KEY: observation.signature,
            _CONSECUTIVE_COUNT_KEY: consecutive_steps,
        }

    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self,
        state: ProgressGuardState[ResponseT],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Async version of `before_model`."""
        return self.before_model(state, runtime)

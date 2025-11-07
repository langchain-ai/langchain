"""Tool call limit middleware for agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Generic, Literal

from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.typing import ContextT
from typing_extensions import NotRequired

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    PrivateStateAttr,
    ResponseT,
    hook_config,
)

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

ExitBehavior = Literal["continue", "error", "end"]
"""How to handle execution when tool call limits are exceeded.

- `"continue"`: Block exceeded tools with error messages, let other tools continue (default)
- `"error"`: Raise a `ToolCallLimitExceededError` exception
- `"end"`: Stop execution immediately, injecting a ToolMessage and an AI message
    for the single tool call that exceeded the limit. Raises `NotImplementedError`
    if there are other pending tool calls (due to parallel tool calling).
"""


class ToolCallLimitState(AgentState[ResponseT], Generic[ResponseT]):
    """State schema for ToolCallLimitMiddleware.

    Extends AgentState with tool call tracking fields.

    The count fields are dictionaries mapping tool names to execution counts.
    This allows multiple middleware instances to track different tools independently.
    The special key "__all__" is used for tracking all tool calls globally.
    """

    thread_tool_call_count: NotRequired[Annotated[dict[str, int], PrivateStateAttr]]
    run_tool_call_count: NotRequired[Annotated[dict[str, int], UntrackedValue, PrivateStateAttr]]


def _build_tool_message_content(tool_name: str | None) -> str:
    """Build the error message content for ToolMessage when limit is exceeded.

    This message is sent to the model, so it should not reference thread/run concepts
    that the model has no notion of.

    Args:
        tool_name: Tool name being limited (if specific tool), or None for all tools.

    Returns:
        A concise message instructing the model not to call the tool again.
    """
    # Always instruct the model not to call again, regardless of which limit was hit
    if tool_name:
        return f"Tool call limit exceeded. Do not call '{tool_name}' again."
    return "Tool call limit exceeded. Do not make additional tool calls."


def _build_final_ai_message_content(
    thread_count: int,
    run_count: int,
    thread_limit: int | None,
    run_limit: int | None,
    tool_name: str | None,
) -> str:
    """Build the final AI message content for 'end' behavior.

    This message is displayed to the user, so it should include detailed information
    about which limits were exceeded.

    Args:
        thread_count: Current thread tool call count.
        run_count: Current run tool call count.
        thread_limit: Thread tool call limit (if set).
        run_limit: Run tool call limit (if set).
        tool_name: Tool name being limited (if specific tool), or None for all tools.

    Returns:
        A formatted message describing which limits were exceeded.
    """
    tool_desc = f"'{tool_name}' tool" if tool_name else "Tool"
    exceeded_limits = []

    if thread_limit is not None and thread_count > thread_limit:
        exceeded_limits.append(f"thread limit exceeded ({thread_count}/{thread_limit} calls)")
    if run_limit is not None and run_count > run_limit:
        exceeded_limits.append(f"run limit exceeded ({run_count}/{run_limit} calls)")

    limits_text = " and ".join(exceeded_limits)
    return f"{tool_desc} call limit reached: {limits_text}."


class ToolCallLimitExceededError(Exception):
    """Exception raised when tool call limits are exceeded.

    This exception is raised when the configured exit behavior is 'error'
    and either the thread or run tool call limit has been exceeded.
    """

    def __init__(
        self,
        thread_count: int,
        run_count: int,
        thread_limit: int | None,
        run_limit: int | None,
        tool_name: str | None = None,
    ) -> None:
        """Initialize the exception with call count information.

        Args:
            thread_count: Current thread tool call count.
            run_count: Current run tool call count.
            thread_limit: Thread tool call limit (if set).
            run_limit: Run tool call limit (if set).
            tool_name: Tool name being limited (if specific tool), or None for all tools.
        """
        self.thread_count = thread_count
        self.run_count = run_count
        self.thread_limit = thread_limit
        self.run_limit = run_limit
        self.tool_name = tool_name

        msg = _build_final_ai_message_content(
            thread_count, run_count, thread_limit, run_limit, tool_name
        )
        super().__init__(msg)


class ToolCallLimitMiddleware(
    AgentMiddleware[ToolCallLimitState[ResponseT], ContextT],
    Generic[ResponseT, ContextT],
):
    """Track tool call counts and enforces limits during agent execution.

    This middleware monitors the number of tool calls made and can terminate or
    restrict execution when limits are exceeded. It supports both thread-level
    (persistent across runs) and run-level (per invocation) call counting.

    Configuration:
        - `exit_behavior`: How to handle when limits are exceeded
          - `"continue"`: Block exceeded tools, let execution continue (default)
          - `"error"`: Raise an exception
          - `"end"`: Stop immediately with a ToolMessage + AI message for the single
            tool call that exceeded the limit (raises `NotImplementedError` if there
            are other pending tool calls (due to parallel tool calling).

    Examples:
        Continue execution with blocked tools (default):
        ```python
        from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware
        from langchain.agents import create_agent

        # Block exceeded tools but let other tools and model continue
        limiter = ToolCallLimitMiddleware(
            thread_limit=20,
            run_limit=10,
            exit_behavior="continue",  # default
        )

        agent = create_agent("openai:gpt-4o", middleware=[limiter])
        ```

        Stop immediately when limit exceeded:
        ```python
        # End execution immediately with an AI message
        limiter = ToolCallLimitMiddleware(run_limit=5, exit_behavior="end")

        agent = create_agent("openai:gpt-4o", middleware=[limiter])
        ```

        Raise exception on limit:
        ```python
        # Strict limit with exception handling
        limiter = ToolCallLimitMiddleware(tool_name="search", thread_limit=5, exit_behavior="error")

        agent = create_agent("openai:gpt-4o", middleware=[limiter])

        try:
            result = await agent.invoke({"messages": [HumanMessage("Task")]})
        except ToolCallLimitExceededError as e:
            print(f"Search limit exceeded: {e}")
        ```

    """

    state_schema = ToolCallLimitState  # type: ignore[assignment]

    def __init__(
        self,
        *,
        tool_name: str | None = None,
        thread_limit: int | None = None,
        run_limit: int | None = None,
        exit_behavior: ExitBehavior = "continue",
    ) -> None:
        """Initialize the tool call limit middleware.

        Args:
            tool_name: Name of the specific tool to limit. If `None`, limits apply
                to all tools. Defaults to `None`.
            thread_limit: Maximum number of tool calls allowed per thread.
                `None` means no limit. Defaults to `None`.
            run_limit: Maximum number of tool calls allowed per run.
                `None` means no limit. Defaults to `None`.
            exit_behavior: How to handle when limits are exceeded.
                - `"continue"`: Block exceeded tools with error messages, let other
                  tools continue. Model decides when to end. (default)
                - `"error"`: Raise a `ToolCallLimitExceededError` exception
                - `"end"`: Stop execution immediately with a ToolMessage + AI message
                  for the single tool call that exceeded the limit. Raises
                  `NotImplementedError` if there are multiple parallel tool
                  calls to other tools or multiple pending tool calls.

        Raises:
            ValueError: If both limits are `None`, if exit_behavior is invalid,
                or if run_limit exceeds thread_limit.
        """
        super().__init__()

        if thread_limit is None and run_limit is None:
            msg = "At least one limit must be specified (thread_limit or run_limit)"
            raise ValueError(msg)

        valid_behaviors = ("continue", "error", "end")
        if exit_behavior not in valid_behaviors:
            msg = f"Invalid exit_behavior: {exit_behavior!r}. Must be one of {valid_behaviors}"
            raise ValueError(msg)

        if thread_limit is not None and run_limit is not None and run_limit > thread_limit:
            msg = (
                f"run_limit ({run_limit}) cannot exceed thread_limit ({thread_limit}). "
                "The run limit should be less than or equal to the thread limit."
            )
            raise ValueError(msg)

        self.tool_name = tool_name
        self.thread_limit = thread_limit
        self.run_limit = run_limit
        self.exit_behavior = exit_behavior

    @property
    def name(self) -> str:
        """The name of the middleware instance.

        Includes the tool name if specified to allow multiple instances
        of this middleware with different tool names.
        """
        base_name = self.__class__.__name__
        if self.tool_name:
            return f"{base_name}[{self.tool_name}]"
        return base_name

    def _would_exceed_limit(self, thread_count: int, run_count: int) -> bool:
        """Check if incrementing the counts would exceed any configured limit.

        Args:
            thread_count: Current thread call count.
            run_count: Current run call count.

        Returns:
            True if either limit would be exceeded by one more call.
        """
        return (self.thread_limit is not None and thread_count + 1 > self.thread_limit) or (
            self.run_limit is not None and run_count + 1 > self.run_limit
        )

    def _matches_tool_filter(self, tool_call: ToolCall) -> bool:
        """Check if a tool call matches this middleware's tool filter.

        Args:
            tool_call: The tool call to check.

        Returns:
            True if this middleware should track this tool call.
        """
        return self.tool_name is None or tool_call["name"] == self.tool_name

    def _separate_tool_calls(
        self, tool_calls: list[ToolCall], thread_count: int, run_count: int
    ) -> tuple[list[ToolCall], list[ToolCall], int, int]:
        """Separate tool calls into allowed and blocked based on limits.

        Args:
            tool_calls: List of tool calls to evaluate.
            thread_count: Current thread call count.
            run_count: Current run call count.

        Returns:
            Tuple of (allowed_calls, blocked_calls, final_thread_count, final_run_count).
        """
        allowed_calls: list[ToolCall] = []
        blocked_calls: list[ToolCall] = []
        temp_thread_count = thread_count
        temp_run_count = run_count

        for tool_call in tool_calls:
            if not self._matches_tool_filter(tool_call):
                continue

            if self._would_exceed_limit(temp_thread_count, temp_run_count):
                blocked_calls.append(tool_call)
            else:
                allowed_calls.append(tool_call)
                temp_thread_count += 1
                temp_run_count += 1

        return allowed_calls, blocked_calls, temp_thread_count, temp_run_count

    @hook_config(can_jump_to=["end"])
    def after_model(
        self,
        state: ToolCallLimitState[ResponseT],
        runtime: Runtime[ContextT],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Increment tool call counts after a model call and check limits.

        Args:
            state: The current agent state.
            runtime: The langgraph runtime.

        Returns:
            State updates with incremented tool call counts. If limits are exceeded
            and exit_behavior is "end", also includes a jump to end with a ToolMessage
            and AI message for the single exceeded tool call.

        Raises:
            ToolCallLimitExceededError: If limits are exceeded and exit_behavior
                is "error".
            NotImplementedError: If limits are exceeded, exit_behavior is "end",
                and there are multiple tool calls.
        """
        # Get the last AIMessage to check for tool calls
        messages = state.get("messages", [])
        if not messages:
            return None

        # Find the last AIMessage
        last_ai_message = None
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                last_ai_message = message
                break

        if not last_ai_message or not last_ai_message.tool_calls:
            return None

        # Get the count key for this middleware instance
        count_key = self.tool_name if self.tool_name else "__all__"

        # Get current counts
        thread_counts = state.get("thread_tool_call_count", {}).copy()
        run_counts = state.get("run_tool_call_count", {}).copy()
        current_thread_count = thread_counts.get(count_key, 0)
        current_run_count = run_counts.get(count_key, 0)

        # Separate tool calls into allowed and blocked
        allowed_calls, blocked_calls, new_thread_count, new_run_count = self._separate_tool_calls(
            last_ai_message.tool_calls, current_thread_count, current_run_count
        )

        # Update counts to include only allowed calls for thread count
        # (blocked calls don't count towards thread-level tracking)
        # But run count includes blocked calls since they were attempted in this run
        thread_counts[count_key] = new_thread_count
        run_counts[count_key] = new_run_count + len(blocked_calls)

        # If no tool calls are blocked, just update counts
        if not blocked_calls:
            if allowed_calls:
                return {
                    "thread_tool_call_count": thread_counts,
                    "run_tool_call_count": run_counts,
                }
            return None

        # Get final counts for building messages
        final_thread_count = thread_counts[count_key]
        final_run_count = run_counts[count_key]

        # Handle different exit behaviors
        if self.exit_behavior == "error":
            # Use hypothetical thread count to show which limit was exceeded
            hypothetical_thread_count = final_thread_count + len(blocked_calls)
            raise ToolCallLimitExceededError(
                thread_count=hypothetical_thread_count,
                run_count=final_run_count,
                thread_limit=self.thread_limit,
                run_limit=self.run_limit,
                tool_name=self.tool_name,
            )

        # Build tool message content (sent to model - no thread/run details)
        tool_msg_content = _build_tool_message_content(self.tool_name)

        # Inject artificial error ToolMessages for blocked tool calls
        artificial_messages: list[ToolMessage | AIMessage] = [
            ToolMessage(
                content=tool_msg_content,
                tool_call_id=tool_call["id"],
                name=tool_call.get("name"),
                status="error",
            )
            for tool_call in blocked_calls
        ]

        if self.exit_behavior == "end":
            # Check if there are tool calls to other tools that would continue executing
            other_tools = [
                tc
                for tc in last_ai_message.tool_calls
                if self.tool_name is not None and tc["name"] != self.tool_name
            ]

            if other_tools:
                tool_names = ", ".join({tc["name"] for tc in other_tools})
                msg = (
                    f"Cannot end execution with other tool calls pending. "
                    f"Found calls to: {tool_names}. Use 'continue' or 'error' behavior instead."
                )
                raise NotImplementedError(msg)

            # Build final AI message content (displayed to user - includes thread/run details)
            # Use hypothetical thread count (what it would have been if call wasn't blocked)
            # to show which limit was actually exceeded
            hypothetical_thread_count = final_thread_count + len(blocked_calls)
            final_msg_content = _build_final_ai_message_content(
                hypothetical_thread_count,
                final_run_count,
                self.thread_limit,
                self.run_limit,
                self.tool_name,
            )
            artificial_messages.append(AIMessage(content=final_msg_content))

            return {
                "thread_tool_call_count": thread_counts,
                "run_tool_call_count": run_counts,
                "jump_to": "end",
                "messages": artificial_messages,
            }

        # For exit_behavior="continue", return error messages to block exceeded tools
        return {
            "thread_tool_call_count": thread_counts,
            "run_tool_call_count": run_counts,
            "messages": artificial_messages,
        }

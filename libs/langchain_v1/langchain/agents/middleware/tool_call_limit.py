"""Tool call limit middleware for agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Generic, Literal

from langchain_core.messages import AIMessage, ToolMessage
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

ExitBehavior = Literal["error", "end", "continue"]
"""How to handle execution when tool call limits are exceeded.

- `"error"`: Raise a `ToolCallLimitExceededError` exception
- `"end"`: Stop execution immediately, injecting a ToolMessage and an AI message
    indicating that the limit was exceeded.
- `"continue"`: Block exceeded tools with error messages, let other tools continue (default)
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


def _build_tool_limit_exceeded_message(
    thread_count: int,
    run_count: int,
    thread_limit: int | None,
    run_limit: int | None,
    tool_name: str | None,
) -> str:
    """Build a message indicating which tool call limits were exceeded.

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

    # Build a concise message
    if tool_name:
        return f"{tool_desc} call limit reached: {limits_text}. Do not call '{tool_name}' again."
    return f"{tool_desc} call limit reached: {limits_text}. Do not make additional tool calls."


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

        msg = _build_tool_limit_exceeded_message(
            thread_count, run_count, thread_limit, run_limit, tool_name
        )
        super().__init__(msg)


class ToolCallLimitMiddleware(
    AgentMiddleware[ToolCallLimitState[ResponseT], ContextT],
    Generic[ResponseT, ContextT],
):
    """Tracks tool call counts and enforces limits during agent execution.

    This middleware monitors the number of tool calls made and can terminate or
    restrict execution when limits are exceeded. It supports both thread-level
    (persistent across runs) and run-level (per invocation) call counting.

    Configuration:
        - `exit_behavior`: How to handle when limits are exceeded
          - `"continue"`: Block exceeded tools, let execution continue (default)
          - `"end"`: Stop immediately with an AI message
          - `"error"`: Raise an exception

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
                - `"end"`: Stop execution immediately with an AI message
                - `"error"`: Raise a `ToolCallLimitExceededError` exception

        Raises:
            ValueError: If both limits are `None`.
        """
        super().__init__()

        if thread_limit is None and run_limit is None:
            msg = "At least one limit must be specified (thread_limit or run_limit)"
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
            and exit_behavior is "end", also includes a jump to end with an AI message.

        Raises:
            ToolCallLimitExceededError: If limits are exceeded and exit_behavior
                is "error".
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

        # Count relevant tool calls (filter by tool_name if specified)
        tool_call_count = 0
        for tool_call in last_ai_message.tool_calls:
            if self.tool_name is None or tool_call["name"] == self.tool_name:
                tool_call_count += 1

        if tool_call_count == 0:
            return None

        # Get the count key for this middleware instance
        count_key = self.tool_name if self.tool_name else "__all__"

        # Get current counts
        thread_counts = state.get("thread_tool_call_count", {}).copy()
        run_counts = state.get("run_tool_call_count", {}).copy()

        # Increment counts for this key
        new_thread_count = thread_counts.get(count_key, 0) + tool_call_count
        new_run_count = run_counts.get(count_key, 0) + tool_call_count

        thread_counts[count_key] = new_thread_count
        run_counts[count_key] = new_run_count

        # Check if any limits are exceeded after incrementing
        thread_limit_exceeded = (
            self.thread_limit is not None and new_thread_count > self.thread_limit
        )
        run_limit_exceeded = self.run_limit is not None and new_run_count > self.run_limit

        if thread_limit_exceeded or run_limit_exceeded:
            limit_message = _build_tool_limit_exceeded_message(
                new_thread_count, new_run_count, self.thread_limit, self.run_limit, self.tool_name
            )

            if self.exit_behavior == "error":
                raise ToolCallLimitExceededError(
                    thread_count=new_thread_count,
                    run_count=new_run_count,
                    thread_limit=self.thread_limit,
                    run_limit=self.run_limit,
                    tool_name=self.tool_name,
                )
            if self.exit_behavior == "end":
                # Add an AI message explaining why we're stopping and jump to end
                limit_ai_message = AIMessage(content=limit_message)

                return {
                    "thread_tool_call_count": thread_counts,
                    "run_tool_call_count": run_counts,
                    "jump_to": "end",
                    "messages": [limit_ai_message],
                }
            # For exit_behavior="continue", inject error ToolMessages for exceeded tool calls
            # This prevents the tools from being called but lets the model see the errors
            error_messages: list[ToolMessage] = []
            for tool_call in last_ai_message.tool_calls:
                # Only inject errors for tool calls that match our filter
                if self.tool_name is None or tool_call["name"] == self.tool_name:
                    error_tool_message = ToolMessage(
                        content=limit_message,
                        tool_call_id=tool_call["id"],
                        name=tool_call.get("name"),
                        status="error",
                    )
                    error_messages.append(error_tool_message)

            return {
                "thread_tool_call_count": thread_counts,
                "run_tool_call_count": run_counts,
                "messages": error_messages,
            }

        return {
            "thread_tool_call_count": thread_counts,
            "run_tool_call_count": run_counts,
        }

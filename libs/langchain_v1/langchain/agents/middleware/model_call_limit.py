"""Call tracking middleware for agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

from langchain_core.messages import AIMessage
from langgraph.channels.untracked_value import UntrackedValue
from typing_extensions import NotRequired, override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    PrivateStateAttr,
    hook_config,
)

if TYPE_CHECKING:
    from langgraph.runtime import Runtime


class ModelCallLimitState(AgentState[Any]):
    """State schema for `ModelCallLimitMiddleware`.

    Extends `AgentState` with model call tracking fields.
    """

    thread_model_call_count: NotRequired[Annotated[int, PrivateStateAttr]]
    run_model_call_count: NotRequired[Annotated[int, UntrackedValue, PrivateStateAttr]]


def _build_limit_exceeded_message(
    thread_count: int,
    run_count: int,
    thread_limit: int | None,
    run_limit: int | None,
) -> str:
    """Build a message indicating which limits were exceeded.

    Args:
        thread_count: Current thread model call count.
        run_count: Current run model call count.
        thread_limit: Thread model call limit (if set).
        run_limit: Run model call limit (if set).

    Returns:
        A formatted message describing which limits were exceeded.
    """
    exceeded_limits = []
    if thread_limit is not None and thread_count >= thread_limit:
        exceeded_limits.append(f"thread limit ({thread_count}/{thread_limit})")
    if run_limit is not None and run_count >= run_limit:
        exceeded_limits.append(f"run limit ({run_count}/{run_limit})")

    return f"Model call limits exceeded: {', '.join(exceeded_limits)}"


class ModelCallLimitExceededError(Exception):
    """Exception raised when model call limits are exceeded.

    This exception is raised when the configured exit behavior is `'error'` and either
    the thread or run model call limit has been exceeded.
    """

    def __init__(
        self,
        thread_count: int,
        run_count: int,
        thread_limit: int | None,
        run_limit: int | None,
    ) -> None:
        """Initialize the exception with call count information.

        Args:
            thread_count: Current thread model call count.
            run_count: Current run model call count.
            thread_limit: Thread model call limit (if set).
            run_limit: Run model call limit (if set).
        """
        self.thread_count = thread_count
        self.run_count = run_count
        self.thread_limit = thread_limit
        self.run_limit = run_limit

        msg = _build_limit_exceeded_message(thread_count, run_count, thread_limit, run_limit)
        super().__init__(msg)


class ModelCallLimitMiddleware(AgentMiddleware[ModelCallLimitState, Any]):
    """Tracks model call counts and enforces limits.

    This middleware monitors the number of model calls made during agent execution
    and can terminate the agent when specified limits are reached. It supports
    both thread-level and run-level call counting with configurable exit behaviors.

    Thread-level: The middleware tracks the number of model calls and persists
    call count across multiple runs (invocations) of the agent.

    Run-level: The middleware tracks the number of model calls made during a single
    run (invocation) of the agent.

    Example:
        ```python
        from langchain.agents.middleware.call_tracking import ModelCallLimitMiddleware
        from langchain.agents import create_agent

        # Create middleware with limits
        call_tracker = ModelCallLimitMiddleware(thread_limit=10, run_limit=5, exit_behavior="end")

        agent = create_agent("openai:gpt-4o", middleware=[call_tracker])

        # Agent will automatically jump to end when limits are exceeded
        result = await agent.invoke({"messages": [HumanMessage("Help me with a task")]})
        ```
    """

    state_schema = ModelCallLimitState

    def __init__(
        self,
        *,
        thread_limit: int | None = None,
        run_limit: int | None = None,
        exit_behavior: Literal["end", "error"] = "end",
    ) -> None:
        """Initialize the call tracking middleware.

        Args:
            thread_limit: Maximum number of model calls allowed per thread.

                `None` means no limit.
            run_limit: Maximum number of model calls allowed per run.

                `None` means no limit.
            exit_behavior: What to do when limits are exceeded.

                - `'end'`: Jump to the end of the agent execution and
                    inject an artificial AI message indicating that the limit was
                    exceeded.
                - `'error'`: Raise a `ModelCallLimitExceededError`

        Raises:
            ValueError: If both limits are `None` or if `exit_behavior` is invalid.
        """
        super().__init__()

        if thread_limit is None and run_limit is None:
            msg = "At least one limit must be specified (thread_limit or run_limit)"
            raise ValueError(msg)

        if exit_behavior not in {"end", "error"}:
            msg = f"Invalid exit_behavior: {exit_behavior}. Must be 'end' or 'error'"
            raise ValueError(msg)

        self.thread_limit = thread_limit
        self.run_limit = run_limit
        self.exit_behavior = exit_behavior

    @hook_config(can_jump_to=["end"])
    @override
    def before_model(self, state: ModelCallLimitState, runtime: Runtime) -> dict[str, Any] | None:
        """Check model call limits before making a model call.

        Args:
            state: The current agent state containing call counts.
            runtime: The langgraph runtime.

        Returns:
            If limits are exceeded and exit_behavior is `'end'`, returns
                a `Command` to jump to the end with a limit exceeded message. Otherwise
                returns `None`.

        Raises:
            ModelCallLimitExceededError: If limits are exceeded and `exit_behavior`
                is `'error'`.
        """
        thread_count = state.get("thread_model_call_count", 0)
        run_count = state.get("run_model_call_count", 0)

        # Check if any limits will be exceeded after the next call
        thread_limit_exceeded = self.thread_limit is not None and thread_count >= self.thread_limit
        run_limit_exceeded = self.run_limit is not None and run_count >= self.run_limit

        if thread_limit_exceeded or run_limit_exceeded:
            if self.exit_behavior == "error":
                raise ModelCallLimitExceededError(
                    thread_count=thread_count,
                    run_count=run_count,
                    thread_limit=self.thread_limit,
                    run_limit=self.run_limit,
                )
            if self.exit_behavior == "end":
                # Create a message indicating the limit was exceeded
                limit_message = _build_limit_exceeded_message(
                    thread_count, run_count, self.thread_limit, self.run_limit
                )
                limit_ai_message = AIMessage(content=limit_message)

                return {"jump_to": "end", "messages": [limit_ai_message]}

        return None

    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self,
        state: ModelCallLimitState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Async check model call limits before making a model call.

        Args:
            state: The current agent state containing call counts.
            runtime: The langgraph runtime.

        Returns:
            If limits are exceeded and exit_behavior is `'end'`, returns
                a `Command` to jump to the end with a limit exceeded message. Otherwise
                returns `None`.

        Raises:
            ModelCallLimitExceededError: If limits are exceeded and `exit_behavior`
                is `'error'`.
        """
        return self.before_model(state, runtime)

    @override
    def after_model(self, state: ModelCallLimitState, runtime: Runtime) -> dict[str, Any] | None:
        """Increment model call counts after a model call.

        Args:
            state: The current agent state.
            runtime: The langgraph runtime.

        Returns:
            State updates with incremented call counts.
        """
        return {
            "thread_model_call_count": state.get("thread_model_call_count", 0) + 1,
            "run_model_call_count": state.get("run_model_call_count", 0) + 1,
        }

    async def aafter_model(
        self,
        state: ModelCallLimitState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Async increment model call counts after a model call.

        Args:
            state: The current agent state.
            runtime: The langgraph runtime.

        Returns:
            State updates with incremented call counts.
        """
        return self.after_model(state, runtime)

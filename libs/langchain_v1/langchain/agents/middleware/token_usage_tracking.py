"""Token usage tracking middleware for agents."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Any, Literal

from langchain_core.messages import AIMessage
from langgraph.channels.untracked_value import UntrackedValue
from typing_extensions import NotRequired, override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    PrivateStateAttr,
    ResponseT,
    hook_config,
)

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)


class TokenUsageState(AgentState[ResponseT]):
    """State schema for `TokenUsageTrackingMiddleware`.

    Extends `AgentState` with token usage tracking fields.

    Type Parameters:
        ResponseT: The type of the structured response. Defaults to `Any`.
    """

    thread_input_tokens: NotRequired[Annotated[int, PrivateStateAttr]]
    thread_output_tokens: NotRequired[Annotated[int, PrivateStateAttr]]
    thread_total_tokens: NotRequired[Annotated[int, PrivateStateAttr]]
    run_input_tokens: NotRequired[Annotated[int, UntrackedValue, PrivateStateAttr]]
    run_output_tokens: NotRequired[Annotated[int, UntrackedValue, PrivateStateAttr]]
    run_total_tokens: NotRequired[Annotated[int, UntrackedValue, PrivateStateAttr]]


def _build_budget_exceeded_message(
    thread_total: int,
    run_total: int,
    thread_budget: int | None,
    run_budget: int | None,
) -> str:
    """Build a message indicating which token budgets were exceeded.

    Args:
        thread_total: Current thread total token count.
        run_total: Current run total token count.
        thread_budget: Thread token budget (if set).
        run_budget: Run token budget (if set).

    Returns:
        A formatted message describing which budgets were exceeded.
    """
    exceeded = []
    if thread_budget is not None and thread_total >= thread_budget:
        exceeded.append(f"thread budget ({thread_total}/{thread_budget} tokens)")
    if run_budget is not None and run_total >= run_budget:
        exceeded.append(f"run budget ({run_total}/{run_budget} tokens)")
    return f"Token budget exceeded: {', '.join(exceeded)}"


class TokenBudgetExceededError(Exception):
    """Exception raised when token budgets are exceeded.

    This exception is raised when the configured exit behavior is `'error'` and either
    the thread or run token budget has been exceeded.
    """

    def __init__(
        self,
        thread_total: int,
        run_total: int,
        thread_budget: int | None,
        run_budget: int | None,
    ) -> None:
        """Initialize the exception with token usage information.

        Args:
            thread_total: Current thread total token count.
            run_total: Current run total token count.
            thread_budget: Thread token budget (if set).
            run_budget: Run token budget (if set).
        """
        self.thread_total = thread_total
        self.run_total = run_total
        self.thread_budget = thread_budget
        self.run_budget = run_budget

        msg = _build_budget_exceeded_message(thread_total, run_total, thread_budget, run_budget)
        super().__init__(msg)


class TokenUsageTrackingMiddleware(
    AgentMiddleware[TokenUsageState[ResponseT], ContextT, ResponseT]
):
    """Tracks cumulative token usage across model calls and optionally enforces budgets.

    This middleware extracts `usage_metadata` from `AIMessage` responses to track
    input tokens, output tokens, and total tokens at both thread and run levels.

    Token counts are accumulated on each `after_model` call. When a budget is
    configured, the middleware checks token totals in `before_model` and can
    either stop the agent gracefully or raise an error.

    Example:
        ```python
        from langchain.agents.middleware import TokenUsageTrackingMiddleware
        from langchain.agents import create_agent

        # Track usage without limits (observability only)
        tracker = TokenUsageTrackingMiddleware()

        agent = create_agent("openai:gpt-4o", middleware=[tracker])
        result = await agent.invoke({"messages": [HumanMessage("Hello")]})

        # Access token counts from the state
        # result["run_total_tokens"], result["thread_total_tokens"], etc.
        ```

        ```python
        # Enforce a per-run token budget
        tracker = TokenUsageTrackingMiddleware(run_budget=10000, exit_behavior="end")

        agent = create_agent("openai:gpt-4o", tools=[search], middleware=[tracker])
        result = await agent.invoke({"messages": [HumanMessage("Research topic X")]})
        ```
    """

    state_schema = TokenUsageState  # type: ignore[assignment]

    def __init__(
        self,
        *,
        thread_budget: int | None = None,
        run_budget: int | None = None,
        exit_behavior: Literal["end", "error"] = "end",
    ) -> None:
        """Initialize the token usage tracking middleware.

        Args:
            thread_budget: Maximum total tokens allowed per thread.

                `None` means no limit (tracking only).
            run_budget: Maximum total tokens allowed per run.

                `None` means no limit (tracking only).
            exit_behavior: What to do when a budget is exceeded.

                - `'end'`: Jump to the end of the agent execution and inject an
                    `AIMessage` indicating the budget was exceeded.
                - `'error'`: Raise a `TokenBudgetExceededError`.

                Only takes effect when at least one budget is set.

        Raises:
            ValueError: If `exit_behavior` is not `'end'` or `'error'`.
        """
        super().__init__()

        if exit_behavior not in {"end", "error"}:
            msg = f"Invalid exit_behavior: {exit_behavior!r}. Must be 'end' or 'error'."
            raise ValueError(msg)

        self.thread_budget = thread_budget
        self.run_budget = run_budget
        self.exit_behavior = exit_behavior

    @hook_config(can_jump_to=["end"])
    @override
    def before_model(
        self, state: TokenUsageState[ResponseT], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Check token budgets before making a model call.

        Args:
            state: The current agent state containing token counts.
            runtime: The langgraph runtime.

        Returns:
            If a budget is exceeded and `exit_behavior` is `'end'`, returns a state
            update that jumps to end with a budget-exceeded message. Otherwise `None`.

        Raises:
            TokenBudgetExceededError: If a budget is exceeded and `exit_behavior`
                is `'error'`.
        """
        if self.thread_budget is None and self.run_budget is None:
            return None

        thread_total = state.get("thread_total_tokens", 0)
        run_total = state.get("run_total_tokens", 0)

        thread_exceeded = (
            self.thread_budget is not None and thread_total >= self.thread_budget
        )
        run_exceeded = (
            self.run_budget is not None and run_total >= self.run_budget
        )

        if thread_exceeded or run_exceeded:
            if self.exit_behavior == "error":
                raise TokenBudgetExceededError(
                    thread_total=thread_total,
                    run_total=run_total,
                    thread_budget=self.thread_budget,
                    run_budget=self.run_budget,
                )
            message = _build_budget_exceeded_message(
                thread_total, run_total, self.thread_budget, self.run_budget
            )
            return {"jump_to": "end", "messages": [AIMessage(content=message)]}

        return None

    @hook_config(can_jump_to=["end"])
    @override
    async def abefore_model(
        self,
        state: TokenUsageState[ResponseT],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Async check token budgets before making a model call.

        Args:
            state: The current agent state containing token counts.
            runtime: The langgraph runtime.

        Returns:
            If a budget is exceeded and `exit_behavior` is `'end'`, returns a state
            update that jumps to end with a budget-exceeded message. Otherwise `None`.

        Raises:
            TokenBudgetExceededError: If a budget is exceeded and `exit_behavior`
                is `'error'`.
        """
        return self.before_model(state, runtime)

    @override
    def after_model(
        self, state: TokenUsageState[ResponseT], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Extract and accumulate token usage from the latest model response.

        Reads `usage_metadata` from the most recent `AIMessage` in the state and
        increments both thread-level and run-level token counters.

        Args:
            state: The current agent state.
            runtime: The langgraph runtime.

        Returns:
            State updates with incremented token counts, or `None` if no
            usage metadata is available.
        """
        input_tokens, output_tokens, total_tokens = _extract_usage(state)

        if total_tokens == 0:
            return None

        return {
            "thread_input_tokens": state.get("thread_input_tokens", 0) + input_tokens,
            "thread_output_tokens": state.get("thread_output_tokens", 0) + output_tokens,
            "thread_total_tokens": state.get("thread_total_tokens", 0) + total_tokens,
            "run_input_tokens": state.get("run_input_tokens", 0) + input_tokens,
            "run_output_tokens": state.get("run_output_tokens", 0) + output_tokens,
            "run_total_tokens": state.get("run_total_tokens", 0) + total_tokens,
        }

    @override
    async def aafter_model(
        self, state: TokenUsageState[ResponseT], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async extract and accumulate token usage from the latest model response.

        Args:
            state: The current agent state.
            runtime: The langgraph runtime.

        Returns:
            State updates with incremented token counts, or `None` if no
            usage metadata is available.
        """
        return self.after_model(state, runtime)


def _extract_usage(state: TokenUsageState[Any]) -> tuple[int, int, int]:
    """Extract token usage from the most recent AIMessage in state.

    Args:
        state: The agent state containing messages.

    Returns:
        A tuple of (input_tokens, output_tokens, total_tokens).
        Returns (0, 0, 0) if no usage metadata is found.
    """
    messages = state.get("messages", [])

    # Walk backwards to find the latest AIMessage with usage_metadata
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.usage_metadata is not None:
            return (
                msg.usage_metadata.get("input_tokens", 0),
                msg.usage_metadata.get("output_tokens", 0),
                msg.usage_metadata.get("total_tokens", 0),
            )

    return (0, 0, 0)

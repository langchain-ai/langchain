"""Token budget middleware for agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

from langchain_core.messages import AIMessage
from langchain_core.messages.utils import count_tokens_approximately
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

ExitBehavior = Literal["continue", "error", "end"]
"""How to handle execution when token budgets are exceeded.

- ``'continue'``: Inject a warning AI message and let the agent continue (default)
- ``'error'``: Raise a ``TokenBudgetExceededError`` exception
- ``'end'``: Stop execution immediately with an AI message explaining the limit
"""


class TokenBudgetState(AgentState[ResponseT]):
    """State schema for ``TokenBudgetMiddleware``.

    Extends ``AgentState`` with token usage tracking fields.

    Token usage is stored as a dictionary with keys ``'input_tokens'``,
    ``'output_tokens'``, and ``'total_tokens'``.

    Type Parameters:
        ResponseT: The type of the structured response. Defaults to ``Any``.
    """

    thread_token_usage: NotRequired[Annotated[dict[str, int], PrivateStateAttr]]
    run_token_usage: NotRequired[
        Annotated[dict[str, int], UntrackedValue, PrivateStateAttr]
    ]


def _empty_usage() -> dict[str, int]:
    """Return a zero-initialized token usage dictionary.

    Returns:
        Dictionary with ``'input_tokens'``, ``'output_tokens'``, and
        ``'total_tokens'`` all set to ``0``.
    """
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _add_usage(
    current: dict[str, int], input_tokens: int, output_tokens: int
) -> dict[str, int]:
    """Add token counts to a usage dictionary and return a new copy.

    Args:
        current: The current token usage dictionary.
        input_tokens: Number of input tokens to add.
        output_tokens: Number of output tokens to add.

    Returns:
        A new dictionary with updated token counts.
    """
    return {
        "input_tokens": current.get("input_tokens", 0) + input_tokens,
        "output_tokens": current.get("output_tokens", 0) + output_tokens,
        "total_tokens": (
            current.get("total_tokens", 0) + input_tokens + output_tokens
        ),
    }


def _calculate_cost(
    usage: dict[str, int],
    cost_per_input_token: float,
    cost_per_output_token: float,
) -> float:
    """Calculate estimated cost from token usage.

    Args:
        usage: Token usage dictionary.
        cost_per_input_token: Cost per input token in dollars.
        cost_per_output_token: Cost per output token in dollars.

    Returns:
        Estimated cost in dollars.
    """
    return (
        usage.get("input_tokens", 0) * cost_per_input_token
        + usage.get("output_tokens", 0) * cost_per_output_token
    )


def _build_limit_message(
    thread_usage: dict[str, int],
    run_usage: dict[str, int],
    thread_token_limit: int | None,
    run_token_limit: int | None,
    thread_cost_limit: float | None,
    run_cost_limit: float | None,
    cost_per_input_token: float | None,
    cost_per_output_token: float | None,
) -> str:
    """Build a message describing which token or cost limits were exceeded.

    Args:
        thread_usage: Current thread token usage.
        run_usage: Current run token usage.
        thread_token_limit: Thread token limit (if set).
        run_token_limit: Run token limit (if set).
        thread_cost_limit: Thread cost limit (if set).
        run_cost_limit: Run cost limit (if set).
        cost_per_input_token: Cost per input token (if set).
        cost_per_output_token: Cost per output token (if set).

    Returns:
        A formatted message describing which limits were exceeded.
    """
    exceeded = []

    thread_total = thread_usage.get("total_tokens", 0)
    run_total = run_usage.get("total_tokens", 0)

    if thread_token_limit is not None and thread_total >= thread_token_limit:
        exceeded.append(
            f"thread token limit ({thread_total:,}/{thread_token_limit:,} tokens)"
        )
    if run_token_limit is not None and run_total >= run_token_limit:
        exceeded.append(
            f"run token limit ({run_total:,}/{run_token_limit:,} tokens)"
        )

    if cost_per_input_token is not None and cost_per_output_token is not None:
        if thread_cost_limit is not None:
            thread_cost = _calculate_cost(
                thread_usage, cost_per_input_token, cost_per_output_token
            )
            if thread_cost >= thread_cost_limit:
                exceeded.append(
                    f"thread cost limit (${thread_cost:.4f}/${thread_cost_limit:.4f})"
                )
        if run_cost_limit is not None:
            run_cost = _calculate_cost(
                run_usage, cost_per_input_token, cost_per_output_token
            )
            if run_cost >= run_cost_limit:
                exceeded.append(
                    f"run cost limit (${run_cost:.4f}/${run_cost_limit:.4f})"
                )

    return f"Token budget exceeded: {', '.join(exceeded)}."


class TokenBudgetExceededError(Exception):
    """Exception raised when token or cost budget is exceeded.

    This exception is raised when the configured exit behavior is ``'error'`` and
    either the thread or run token/cost limit has been exceeded.
    """

    def __init__(
        self,
        thread_usage: dict[str, int],
        run_usage: dict[str, int],
        thread_token_limit: int | None,
        run_token_limit: int | None,
        thread_cost_limit: float | None = None,
        run_cost_limit: float | None = None,
        estimated_cost: float | None = None,
    ) -> None:
        """Initialize the exception with usage information.

        Args:
            thread_usage: Current thread token usage.
            run_usage: Current run token usage.
            thread_token_limit: Thread token limit (if set).
            run_token_limit: Run token limit (if set).
            thread_cost_limit: Thread cost limit (if set).
            run_cost_limit: Run cost limit (if set).
            estimated_cost: Estimated cost in dollars (if calculated).
        """
        self.thread_usage = thread_usage
        self.run_usage = run_usage
        self.thread_token_limit = thread_token_limit
        self.run_token_limit = run_token_limit
        self.thread_cost_limit = thread_cost_limit
        self.run_cost_limit = run_cost_limit
        self.estimated_cost = estimated_cost

        msg = _build_limit_message(
            thread_usage,
            run_usage,
            thread_token_limit,
            run_token_limit,
            thread_cost_limit,
            run_cost_limit,
            None,
            None,
        )
        super().__init__(msg)


class TokenBudgetMiddleware(
    AgentMiddleware[TokenBudgetState[ResponseT], ContextT, ResponseT]
):
    """Track token usage and enforce budgets during agent execution.

    This middleware monitors cumulative input and output tokens across model calls
    and can terminate or restrict execution when configured budgets are exceeded.
    It supports both thread-level (persistent across runs) and run-level
    (per invocation) budget enforcement.

    Token counts are extracted from ``AIMessage.usage_metadata`` when available,
    falling back to approximate counting via ``count_tokens_approximately`` when
    the model does not report usage metadata.

    Optionally tracks estimated cost using configurable per-token pricing.

    Examples:
        !!! example "Basic token limit per run"

            ```python
            from langchain.agents import create_agent
            from langchain.agents.middleware.token_budget import TokenBudgetMiddleware

            budget = TokenBudgetMiddleware(run_token_limit=50_000)
            agent = create_agent("openai:gpt-4o", middleware=[budget])
            ```

        !!! example "Thread-level limit (persists across invocations)"

            ```python
            budget = TokenBudgetMiddleware(
                thread_token_limit=200_000,
                run_token_limit=50_000,
            )
            agent = create_agent("openai:gpt-4o", middleware=[budget])
            ```

        !!! example "Cost-based limit"

            ```python
            budget = TokenBudgetMiddleware(
                run_cost_limit=0.50,
                cost_per_input_token=0.000003,
                cost_per_output_token=0.000015,
            )
            agent = create_agent("openai:gpt-4o", middleware=[budget])
            ```

        !!! example "Raise exception on budget exceeded"

            ```python
            budget = TokenBudgetMiddleware(
                run_token_limit=10_000,
                exit_behavior="error",
            )
            agent = create_agent("openai:gpt-4o", middleware=[budget])

            try:
                result = await agent.invoke({"messages": [HumanMessage("Task")]})
            except TokenBudgetExceededError as e:
                print(f"Budget exceeded: {e}")
            ```

    """

    state_schema = TokenBudgetState  # type: ignore[assignment]

    def __init__(
        self,
        *,
        thread_token_limit: int | None = None,
        run_token_limit: int | None = None,
        thread_cost_limit: float | None = None,
        run_cost_limit: float | None = None,
        cost_per_input_token: float | None = None,
        cost_per_output_token: float | None = None,
        exit_behavior: ExitBehavior = "continue",
    ) -> None:
        """Initialize the token budget middleware.

        Args:
            thread_token_limit: Maximum total tokens allowed per thread.
                ``None`` means no token limit at thread level.
            run_token_limit: Maximum total tokens allowed per run.
                ``None`` means no token limit at run level.
            thread_cost_limit: Maximum estimated cost in dollars per thread.
                ``None`` means no cost limit at thread level.
                Requires ``cost_per_input_token`` and ``cost_per_output_token``.
            run_cost_limit: Maximum estimated cost in dollars per run.
                ``None`` means no cost limit at run level.
                Requires ``cost_per_input_token`` and ``cost_per_output_token``.
            cost_per_input_token: Cost per input token in dollars.
                Required if using cost-based limits.
            cost_per_output_token: Cost per output token in dollars.
                Required if using cost-based limits.
            exit_behavior: How to handle when budgets are exceeded.

                - ``'continue'``: Inject a warning AI message, let execution continue.
                    The model may decide to stop based on the warning.
                - ``'error'``: Raise a ``TokenBudgetExceededError`` exception.
                - ``'end'``: Stop execution immediately with an AI message
                    explaining the budget limit.

        Raises:
            ValueError: If no limits are specified, if cost limits are set without
                per-token pricing, if ``exit_behavior`` is invalid, or if
                ``run_token_limit`` exceeds ``thread_token_limit``.
        """
        super().__init__()

        has_token_limit = (
            thread_token_limit is not None or run_token_limit is not None
        )
        has_cost_limit = (
            thread_cost_limit is not None or run_cost_limit is not None
        )

        if not has_token_limit and not has_cost_limit:
            msg = (
                "At least one limit must be specified "
                "(thread_token_limit, run_token_limit, thread_cost_limit, "
                "or run_cost_limit)"
            )
            raise ValueError(msg)

        if has_cost_limit and (
            cost_per_input_token is None or cost_per_output_token is None
        ):
            msg = (
                "cost_per_input_token and cost_per_output_token must both be "
                "specified when using cost-based limits"
            )
            raise ValueError(msg)

        valid_behaviors = ("continue", "error", "end")
        if exit_behavior not in valid_behaviors:
            msg = (
                f"Invalid exit_behavior: {exit_behavior!r}. "
                f"Must be one of {valid_behaviors}"
            )
            raise ValueError(msg)

        if (
            thread_token_limit is not None
            and run_token_limit is not None
            and run_token_limit > thread_token_limit
        ):
            msg = (
                f"run_token_limit ({run_token_limit}) cannot exceed "
                f"thread_token_limit ({thread_token_limit}). "
                "The run limit should be less than or equal to the thread limit."
            )
            raise ValueError(msg)

        self.thread_token_limit = thread_token_limit
        self.run_token_limit = run_token_limit
        self.thread_cost_limit = thread_cost_limit
        self.run_cost_limit = run_cost_limit
        self.cost_per_input_token = cost_per_input_token
        self.cost_per_output_token = cost_per_output_token
        self.exit_behavior = exit_behavior

    def _is_budget_exceeded(
        self, thread_usage: dict[str, int], run_usage: dict[str, int]
    ) -> bool:
        """Check if any configured budget would be exceeded.

        Args:
            thread_usage: Current thread token usage.
            run_usage: Current run token usage.

        Returns:
            ``True`` if any budget limit is exceeded.
        """
        thread_total = thread_usage.get("total_tokens", 0)
        run_total = run_usage.get("total_tokens", 0)

        if (
            self.thread_token_limit is not None
            and thread_total >= self.thread_token_limit
        ):
            return True
        if (
            self.run_token_limit is not None
            and run_total >= self.run_token_limit
        ):
            return True

        if (
            self.cost_per_input_token is not None
            and self.cost_per_output_token is not None
        ):
            if self.thread_cost_limit is not None:
                thread_cost = _calculate_cost(
                    thread_usage,
                    self.cost_per_input_token,
                    self.cost_per_output_token,
                )
                if thread_cost >= self.thread_cost_limit:
                    return True
            if self.run_cost_limit is not None:
                run_cost = _calculate_cost(
                    run_usage,
                    self.cost_per_input_token,
                    self.cost_per_output_token,
                )
                if run_cost >= self.run_cost_limit:
                    return True

        return False

    def _handle_exceeded(
        self,
        thread_usage: dict[str, int],
        run_usage: dict[str, int],
    ) -> dict[str, Any]:
        """Handle budget exceeded based on configured exit behavior.

        Args:
            thread_usage: Current thread token usage.
            run_usage: Current run token usage.

        Returns:
            State updates with appropriate messages and jump directives.

        Raises:
            TokenBudgetExceededError: If ``exit_behavior`` is ``'error'``.
        """
        limit_message = _build_limit_message(
            thread_usage,
            run_usage,
            self.thread_token_limit,
            self.run_token_limit,
            self.thread_cost_limit,
            self.run_cost_limit,
            self.cost_per_input_token,
            self.cost_per_output_token,
        )

        if self.exit_behavior == "error":
            estimated_cost = None
            if (
                self.cost_per_input_token is not None
                and self.cost_per_output_token is not None
            ):
                estimated_cost = _calculate_cost(
                    thread_usage,
                    self.cost_per_input_token,
                    self.cost_per_output_token,
                )
            raise TokenBudgetExceededError(
                thread_usage=thread_usage,
                run_usage=run_usage,
                thread_token_limit=self.thread_token_limit,
                run_token_limit=self.run_token_limit,
                thread_cost_limit=self.thread_cost_limit,
                run_cost_limit=self.run_cost_limit,
                estimated_cost=estimated_cost,
            )

        ai_message = AIMessage(content=limit_message)

        if self.exit_behavior == "end":
            return {
                "jump_to": "end",
                "messages": [ai_message],
            }

        # exit_behavior == "continue": inject warning but don't stop
        return {
            "messages": [ai_message],
        }

    def _extract_token_usage(
        self, state: TokenBudgetState[ResponseT]
    ) -> tuple[int, int] | None:
        """Extract token usage from the last AI message in state.

        First tries to read ``usage_metadata`` from the ``AIMessage``. Falls back
        to ``count_tokens_approximately`` if ``usage_metadata`` is not available.

        Args:
            state: The current agent state.

        Returns:
            A tuple of ``(input_tokens, output_tokens)`` or ``None`` if no
            AI message is found.
        """
        messages = state.get("messages", [])
        if not messages:
            return None

        # Find the last AIMessage
        last_ai_message = None
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                last_ai_message = message
                break

        if last_ai_message is None:
            return None

        # Try usage_metadata first (provider-reported, most accurate)
        if last_ai_message.usage_metadata:
            return (
                last_ai_message.usage_metadata.get("input_tokens", 0),
                last_ai_message.usage_metadata.get("output_tokens", 0),
            )

        # Fallback: approximate output tokens from message content
        output_tokens = count_tokens_approximately([last_ai_message])

        # Approximate input tokens from all non-AI messages preceding the last AI
        input_messages = []
        for message in messages:
            if message is last_ai_message:
                break
            input_messages.append(message)
        input_tokens = (
            count_tokens_approximately(input_messages)
            if input_messages
            else 0
        )

        return (input_tokens, output_tokens)

    @hook_config(can_jump_to=["end"])
    @override
    def before_model(
        self,
        state: TokenBudgetState[ResponseT],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Check token budgets before making a model call.

        If the budget is already exceeded from previous calls, this will prevent
        the next model call based on the configured ``exit_behavior``.

        Args:
            state: The current agent state containing token usage.
            runtime: The langgraph runtime.

        Returns:
            If budgets are exceeded and ``exit_behavior`` is ``'end'`` or
            ``'continue'``, returns state updates with messages. Otherwise
            returns ``None``.

        Raises:
            TokenBudgetExceededError: If budgets are exceeded and ``exit_behavior``
                is ``'error'``.
        """
        thread_usage = state.get("thread_token_usage", _empty_usage())
        run_usage = state.get("run_token_usage", _empty_usage())

        if self._is_budget_exceeded(thread_usage, run_usage):
            result = self._handle_exceeded(thread_usage, run_usage)
            result["thread_token_usage"] = thread_usage
            result["run_token_usage"] = run_usage
            return result

        return None

    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self,
        state: TokenBudgetState[ResponseT],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Async check token budgets before making a model call.

        Args:
            state: The current agent state containing token usage.
            runtime: The langgraph runtime.

        Returns:
            If budgets are exceeded and ``exit_behavior`` is ``'end'`` or
            ``'continue'``, returns state updates with messages. Otherwise
            returns ``None``.

        Raises:
            TokenBudgetExceededError: If budgets are exceeded and ``exit_behavior``
                is ``'error'``.
        """
        return self.before_model(state, runtime)

    @override
    def after_model(
        self,
        state: TokenBudgetState[ResponseT],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Extract token usage after a model call and update cumulative counts.

        Reads ``usage_metadata`` from the ``AIMessage`` in the state. Falls back
        to ``count_tokens_approximately`` if ``usage_metadata`` is not available.

        Args:
            state: The current agent state.
            runtime: The langgraph runtime.

        Returns:
            State updates with incremented token usage counts, or ``None`` if
            no usage data could be extracted.
        """
        usage = self._extract_token_usage(state)
        if usage is None:
            return None

        input_tokens, output_tokens = usage

        thread_usage = state.get("thread_token_usage", _empty_usage())
        run_usage = state.get("run_token_usage", _empty_usage())

        new_thread_usage = _add_usage(thread_usage, input_tokens, output_tokens)
        new_run_usage = _add_usage(run_usage, input_tokens, output_tokens)

        return {
            "thread_token_usage": new_thread_usage,
            "run_token_usage": new_run_usage,
        }

    async def aafter_model(
        self,
        state: TokenBudgetState[ResponseT],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Async extract token usage after a model call and update counts.

        Args:
            state: The current agent state.
            runtime: The langgraph runtime.

        Returns:
            State updates with incremented token usage counts, or ``None`` if
            no usage data could be extracted.
        """
        return self.after_model(state, runtime)

"""Token budget middleware for agents."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from langchain_core.messages import AIMessage, HumanMessage, MessageLikeRepresentation
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import BaseTool
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.types import Command
from typing_extensions import NotRequired, override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
    ResponseT,
    hook_config,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from langgraph.runtime import Runtime

TokenCounter = Callable[[Iterable[MessageLikeRepresentation]], int]
"""Callable used to estimate token usage when the model reports no `usage_metadata`."""

ToolFilter = Sequence[str] | Callable[[BaseTool | dict[str, Any]], bool]
"""Selects tools by name, or with a predicate over the tool object.

Tools are `BaseTool` instances, or `dict` specs for provider built-in tools.
"""

ExitBehavior = Literal["check_in", "end", "error"]
"""How to handle execution when a token budget is exceeded.

- `'check_in'`: Inject a synthetic `HumanMessage` asking the model to stop, summarize
    its progress, and ask the user how to proceed. Restricted tools are removed from
    the following model calls so the run ends with the model's reply (default).
- `'end'`: Jump to the end of the agent execution and inject an artificial
    `AIMessage` indicating that the budget was exceeded.
- `'error'`: Raise a `TokenBudgetExceededError`.
"""

DEFAULT_CHECK_IN_PROMPT = (
    "The token budget for this turn has been used up, so you must stop working on the "
    "task now. Do not call any tools. Reply to the user with a concise summary of what "
    "you have done so far, what remains to be done, and any decisions or open questions. "
    "Then ask the user how they would like you to proceed."
)
"""Default content of the synthetic `HumanMessage` injected by `'check_in'`."""

CHECK_IN_SOURCE = "token_budget"
"""Value of `additional_kwargs['lc_source']` on injected check-in messages."""


class TokenBudgetState(AgentState[ResponseT]):
    """State schema for `TokenBudgetMiddleware`.

    Extends `AgentState` with token usage tracking fields.

    Type Parameters:
        ResponseT: The type of the structured response. Defaults to `Any`.
    """

    thread_token_usage: NotRequired[Annotated[int, PrivateStateAttr]]
    turn_token_usage: NotRequired[Annotated[int, UntrackedValue, PrivateStateAttr]]
    token_budget_check_in_sent: NotRequired[Annotated[bool, UntrackedValue, PrivateStateAttr]]
    token_budget_exempt_tools: NotRequired[Annotated[list[str], UntrackedValue, PrivateStateAttr]]


def _tool_name(tool: BaseTool | dict[str, Any]) -> str | None:
    """Return the name of a tool, or `None` if it cannot be determined."""
    if isinstance(tool, BaseTool):
        return tool.name
    name = tool.get("name")
    if name is None and isinstance(tool.get("function"), dict):
        name = tool["function"].get("name")
    return name if isinstance(name, str) else None


def _build_limit_exceeded_message(
    thread_usage: int,
    turn_usage: int,
    thread_limit: int | None,
    turn_limit: int | None,
) -> str:
    """Build a message indicating which budgets were exceeded.

    Args:
        thread_usage: Tokens used so far in the thread.
        turn_usage: Tokens used so far in the turn.
        thread_limit: Thread token budget (if set).
        turn_limit: Turn token budget (if set).

    Returns:
        A formatted message describing which budgets were exceeded.
    """
    exceeded_limits = []
    if thread_limit is not None and thread_usage >= thread_limit:
        exceeded_limits.append(f"thread limit ({thread_usage}/{thread_limit})")
    if turn_limit is not None and turn_usage >= turn_limit:
        exceeded_limits.append(f"turn limit ({turn_usage}/{turn_limit})")

    return f"Token budget exceeded: {', '.join(exceeded_limits)}"


class TokenBudgetExceededError(Exception):
    """Exception raised when a token budget is exceeded.

    This exception is raised when the configured exit behavior is `'error'` and either
    the thread or turn token budget has been exceeded.
    """

    def __init__(
        self,
        thread_usage: int,
        turn_usage: int,
        thread_limit: int | None,
        turn_limit: int | None,
    ) -> None:
        """Initialize the exception with token usage information.

        Args:
            thread_usage: Tokens used so far in the thread.
            turn_usage: Tokens used so far in the turn.
            thread_limit: Thread token budget (if set).
            turn_limit: Turn token budget (if set).
        """
        self.thread_usage = thread_usage
        self.turn_usage = turn_usage
        self.thread_limit = thread_limit
        self.turn_limit = turn_limit

        msg = _build_limit_exceeded_message(thread_usage, turn_usage, thread_limit, turn_limit)
        super().__init__(msg)


class TokenBudgetMiddleware(AgentMiddleware[TokenBudgetState[ResponseT], ContextT, ResponseT]):
    """Tracks token usage and keeps each turn of the conversation within a budget.

    Token usage is read from the `usage_metadata` reported by the model on each call
    and accumulated at two scopes:

    - Turn-level: tokens used during a single run (invocation) of the agent, i.e. one
        turn of the conversation. Reset on every turn.
    - Thread-level: tokens used across all turns of the agent on a thread. Persisted
        with the checkpointer.

    When a budget is exceeded and the agent is about to call the model again, the
    configured `exit_behavior` decides what happens. The default, `'check_in'`, injects
    a synthetic `HumanMessage` asking the model to stop, summarize what it has done and
    what is left, and ask the user how to proceed. Tools are removed from the following
    model calls, so the run ends with the model's reply and control returns to the user
    instead of the agent silently looping past its budget. The injected message carries
    `additional_kwargs={"lc_source": "token_budget"}` so applications can recognise it.

    Tools that must stay available after the budget is exceeded (for example a tool
    that asks the user a question, or one that saves progress) can be declared with
    `exempt_tools`, or everything except a few tools can be kept with
    `restricted_tools`. A model reply that only calls exempt tools keeps the run going;
    a call to a restricted tool ends it.

    If the model does not report `usage_metadata`, usage is estimated with
    `token_counter` over the request messages and the model response.

    Example:
        ```python
        from langchain.agents.middleware import TokenBudgetMiddleware
        from langchain.agents import create_agent

        budget = TokenBudgetMiddleware(turn_token_limit=50_000, exempt_tools=["ask_user"])

        agent = create_agent("openai:gpt-5.5", tools=tools, middleware=[budget])

        # Once a turn has consumed 50k tokens, the agent stops calling tools other
        # than `ask_user`, summarizes its progress, and asks the user for direction.
        result = agent.invoke({"messages": [HumanMessage("Refactor the billing module")]})
        ```
    """

    state_schema = TokenBudgetState  # type: ignore[assignment]

    def __init__(
        self,
        *,
        turn_token_limit: int | None = None,
        thread_token_limit: int | None = None,
        exit_behavior: ExitBehavior = "check_in",
        check_in_prompt: str = DEFAULT_CHECK_IN_PROMPT,
        exempt_tools: ToolFilter | None = None,
        restricted_tools: ToolFilter | None = None,
        token_counter: TokenCounter = count_tokens_approximately,
    ) -> None:
        """Initialize the token budget middleware.

        Args:
            turn_token_limit: Maximum number of tokens allowed per turn (one run).

                `None` means no limit.
            thread_token_limit: Maximum number of tokens allowed per thread.

                `None` means no limit.
            exit_behavior: What to do when a budget is exceeded.

                - `'check_in'`: Inject a synthetic `HumanMessage` (`check_in_prompt`)
                    asking the model to summarize progress and ask the user for
                    direction, remove restricted tools from the following model
                    calls, and end the run with the model's reply.
                - `'end'`: Jump to the end of the agent execution and inject an
                    artificial `AIMessage` indicating that the budget was exceeded.
                - `'error'`: Raise a `TokenBudgetExceededError`.
            check_in_prompt: Content of the synthetic `HumanMessage` used by
                `'check_in'`.
            exempt_tools: Tools that are never limited by the budget, as a list of
                tool names or a predicate over the tool. They stay available after
                the check-in, and a model reply that only calls exempt tools does not
                end the run. Mutually exclusive with `restricted_tools`.
            restricted_tools: Tools that are removed after the check-in, as a list of
                tool names or a predicate over the tool. All other tools are exempt.
                Mutually exclusive with `exempt_tools`.
            token_counter: Fallback used to estimate token usage when the model
                response carries no `usage_metadata`. Called with the system message,
                request messages and response messages of a single model call.

        Raises:
            ValueError: If both limits are `None`, a limit is not positive,
                `exit_behavior` is invalid, or both `exempt_tools` and
                `restricted_tools` are given.
        """
        super().__init__()

        if turn_token_limit is None and thread_token_limit is None:
            msg = "At least one limit must be specified (turn_token_limit or thread_token_limit)"
            raise ValueError(msg)

        for name, limit in (
            ("turn_token_limit", turn_token_limit),
            ("thread_token_limit", thread_token_limit),
        ):
            if limit is not None and limit <= 0:
                msg = f"{name} must be a positive integer, got {limit}"
                raise ValueError(msg)

        if exit_behavior not in {"check_in", "end", "error"}:
            msg = f"Invalid exit_behavior: {exit_behavior}. Must be 'check_in', 'end' or 'error'"
            raise ValueError(msg)

        if exempt_tools is not None and restricted_tools is not None:
            msg = "Specify either exempt_tools or restricted_tools, not both"
            raise ValueError(msg)

        self.turn_token_limit = turn_token_limit
        self.thread_token_limit = thread_token_limit
        self.exit_behavior = exit_behavior
        self.check_in_prompt = check_in_prompt
        self.exempt_tools = exempt_tools
        self.restricted_tools = restricted_tools
        self.token_counter = token_counter

    @hook_config(can_jump_to=["end"])
    @override
    def before_model(
        self, state: TokenBudgetState[ResponseT], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Check token budgets before making a model call.

        Args:
            state: The current agent state containing token usage.
            runtime: The langgraph runtime.

        Returns:
            `None` if the budgets are respected, or if a check-in was already sent and
                the model only called exempt tools. Otherwise, depending on
                `exit_behavior`: a state update injecting the check-in `HumanMessage`
                (`'check_in'`, first time), or a jump to the end with an `AIMessage`
                (`'end'`, or `'check_in'` when the model called a restricted tool
                after the check-in).

        Raises:
            TokenBudgetExceededError: If a budget is exceeded and `exit_behavior`
                is `'error'`.
        """
        thread_usage = state.get("thread_token_usage", 0)
        turn_usage = state.get("turn_token_usage", 0)

        thread_exceeded = (
            self.thread_token_limit is not None and thread_usage >= self.thread_token_limit
        )
        turn_exceeded = self.turn_token_limit is not None and turn_usage >= self.turn_token_limit

        if not (thread_exceeded or turn_exceeded):
            return None

        if self.exit_behavior == "error":
            raise TokenBudgetExceededError(
                thread_usage=thread_usage,
                turn_usage=turn_usage,
                thread_limit=self.thread_token_limit,
                turn_limit=self.turn_token_limit,
            )

        if self.exit_behavior == "check_in":
            if not state.get("token_budget_check_in_sent"):
                check_in_message = HumanMessage(
                    content=self.check_in_prompt,
                    additional_kwargs={"lc_source": CHECK_IN_SOURCE},
                )
                return {"messages": [check_in_message], "token_budget_check_in_sent": True}

            if self._last_call_used_only_exempt_tools(state):
                return None

        # Either exit_behavior is 'end', or the model called a restricted tool after
        # the check-in. All pending tool calls have been resolved at this point, so
        # ending the run here leaves the message history consistent.
        limit_message = _build_limit_exceeded_message(
            thread_usage, turn_usage, self.thread_token_limit, self.turn_token_limit
        )
        return {"jump_to": "end", "messages": [AIMessage(content=limit_message)]}

    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self, state: TokenBudgetState[ResponseT], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async check token budgets before making a model call.

        Args:
            state: The current agent state containing token usage.
            runtime: The langgraph runtime.

        Returns:
            See `before_model`.

        Raises:
            TokenBudgetExceededError: If a budget is exceeded and `exit_behavior`
                is `'error'`.
        """
        return self.before_model(state, runtime)

    @override
    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ExtendedModelResponse[ResponseT]:
        """Execute the model call and record its token usage.

        After a check-in has been sent, restricted tools are removed from the request
        so the model can only reply to the user or call exempt tools.

        Args:
            request: The model request.
            handler: Callback that executes the model request.

        Returns:
            The model response together with a state update carrying the new
                token usage.
        """
        request = self._prepare_request(request)
        response = handler(request)
        return self._record_usage(request, response)

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ExtendedModelResponse[ResponseT]:
        """Async execute the model call and record its token usage.

        Args:
            request: The model request.
            handler: Async callback that executes the model request.

        Returns:
            See `wrap_model_call`.
        """
        request = self._prepare_request(request)
        response = await handler(request)
        return self._record_usage(request, response)

    def _is_exempt(self, tool: BaseTool | dict[str, Any]) -> bool:
        """Whether a tool stays available after the budget is exceeded."""
        if self.exempt_tools is not None:
            if callable(self.exempt_tools):
                return self.exempt_tools(tool)
            return _tool_name(tool) in set(self.exempt_tools)
        if self.restricted_tools is not None:
            if callable(self.restricted_tools):
                return not self.restricted_tools(tool)
            return _tool_name(tool) not in set(self.restricted_tools)
        return False

    def _prepare_request(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        """Remove restricted tools from model calls made after a check-in."""
        if not request.state.get("token_budget_check_in_sent"):
            return request
        exempt = [tool for tool in request.tools if self._is_exempt(tool)]
        return request.override(tools=exempt, tool_choice=None)

    def _record_usage(
        self, request: ModelRequest[ContextT], response: ModelResponse[ResponseT]
    ) -> ExtendedModelResponse[ResponseT]:
        """Attach a state update with the token usage of this model call."""
        tokens = self._count_tokens(request, response)
        state = cast("TokenBudgetState[ResponseT]", request.state)
        update: dict[str, Any] = {
            "thread_token_usage": state.get("thread_token_usage", 0) + tokens,
            "turn_token_usage": state.get("turn_token_usage", 0) + tokens,
        }
        if state.get("token_budget_check_in_sent"):
            # `request` has already been narrowed to the exempt tools. Remember their
            # names so `before_model` can tell exempt calls from restricted ones.
            update["token_budget_exempt_tools"] = [
                name for tool in request.tools if (name := _tool_name(tool)) is not None
            ]
        return ExtendedModelResponse(model_response=response, command=Command(update=update))

    @staticmethod
    def _last_call_used_only_exempt_tools(state: TokenBudgetState[ResponseT]) -> bool:
        """Whether the last model reply called tools and all of them were exempt."""
        last_ai_message = next(
            (m for m in reversed(state.get("messages", [])) if isinstance(m, AIMessage)),
            None,
        )
        if last_ai_message is None or not last_ai_message.tool_calls:
            return False
        exempt = set(state.get("token_budget_exempt_tools", []))
        return all(tool_call["name"] in exempt for tool_call in last_ai_message.tool_calls)

    def _count_tokens(
        self, request: ModelRequest[ContextT], response: ModelResponse[ResponseT]
    ) -> int:
        """Count the tokens used by a single model call.

        Uses the `usage_metadata` reported on the response messages when available,
        otherwise falls back to `token_counter`.
        """
        reported = 0
        has_usage_metadata = False
        for message in response.result:
            if isinstance(message, AIMessage) and message.usage_metadata:
                has_usage_metadata = True
                usage = message.usage_metadata
                reported += usage.get("total_tokens") or (
                    usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                )
        if has_usage_metadata:
            return reported

        messages: list[MessageLikeRepresentation] = []
        if request.system_message is not None:
            messages.append(request.system_message)
        messages.extend(request.messages)
        messages.extend(response.result)
        return self.token_counter(messages)

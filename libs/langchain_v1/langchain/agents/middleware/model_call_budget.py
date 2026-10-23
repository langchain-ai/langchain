"""Agent-visible model-turn budgets with a reserved final response."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from langchain_core.messages import AIMessage, SystemMessage
from typing_extensions import NotRequired, override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    hook_config,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.runtime import Runtime


class ModelCallBudgetState(AgentState):
    """Per-invocation model-turn consumption."""

    model_budget_used: NotRequired[int]


class ModelCallBudgetMiddleware(AgentMiddleware[ModelCallBudgetState, Any]):
    """Allow `run_limit` working model turns and one extra tool-free final turn."""

    state_schema = ModelCallBudgetState

    def __init__(self, *, run_limit: int) -> None:
        """Configure a nonnegative working-turn budget (not a token or cost limit)."""
        if isinstance(run_limit, bool) or not isinstance(run_limit, int) or run_limit < 0:
            msg = "run_limit must be a nonnegative integer"
            raise ValueError(msg)
        self.run_limit = run_limit

    @override
    def before_agent(self, state: ModelCallBudgetState, runtime: Runtime[Any]) -> dict[str, Any]:
        """Reset consumption for a new invocation, not a checkpoint resume."""
        return {"model_budget_used": 0}

    async def abefore_agent(
        self, state: ModelCallBudgetState, runtime: Runtime[Any]
    ) -> dict[str, Any]:
        """Reset consumption for a new invocation."""
        return self.before_agent(state, runtime)

    def _request(self, request: ModelRequest) -> ModelRequest:
        used = cast("ModelCallBudgetState", request.state).get("model_budget_used", 0)
        remaining = max(0, self.run_limit - used)
        notice = (
            f"Model-call budget: {used}/{self.run_limit} working turns used; "
            f"{remaining} remaining. One additional tool-free final response is reserved."
        )
        if remaining == 0:
            notice += (
                " The working budget is exhausted. Provide your final answer now using "
                "the evidence already collected. State any limitations. Do not call tools."
            )
        system = request.system_message
        content = system.content if system else ""
        updated_content = (
            f"{content}\n\n{notice}"
            if isinstance(content, str)
            else [*content, {"type": "text", "text": notice}]
        )
        system = (
            system.model_copy(update={"content": updated_content})
            if system
            else SystemMessage(content=updated_content)
        )
        if remaining:
            return request.override(system_message=system)
        settings = {
            key: value
            for key, value in request.model_settings.items()
            if key not in {"tools", "tool_choice", "response_format"}
        }
        return request.override(
            system_message=system,
            tools=[],
            tool_choice=None,
            response_format=None,
            model_settings=settings,
        )

    def _validate(self, request: ModelRequest, response: ModelResponse) -> ModelResponse:
        if cast("ModelCallBudgetState", request.state).get(
            "model_budget_used", 0
        ) >= self.run_limit and any(
            isinstance(message, AIMessage) and (message.tool_calls or message.invalid_tool_calls)
            for message in response.result
        ):
            msg = "The budget final response must not contain tool calls"
            raise ValueError(msg)
        return response

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Expose consumption and remove tools on the reserved final turn."""
        return self._validate(request, handler(self._request(request)))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Expose consumption and remove tools on the reserved final turn."""
        return self._validate(request, await handler(self._request(request)))

    @hook_config(can_jump_to=["end"])
    @override
    def after_model(self, state: ModelCallBudgetState, runtime: Runtime[Any]) -> dict[str, Any]:
        """Count model turns and terminate after the reserved final response."""
        used = state.get("model_budget_used", 0) + 1
        update: dict[str, Any] = {"model_budget_used": used}
        if used > self.run_limit:
            update["jump_to"] = "end"
        return update

    @hook_config(can_jump_to=["end"])
    async def aafter_model(
        self, state: ModelCallBudgetState, runtime: Runtime[Any]
    ) -> dict[str, Any]:
        """Count model turns and terminate after the reserved final response."""
        return self.after_model(state, runtime)

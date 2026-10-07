"""Experimental model-routing middleware powered by the OpenAI Decisions API."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
    TracePolicy,
    omit_payload,
)
from langchain.chat_models import init_chat_model
from langchain_core._api import beta
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from typing_extensions import NotRequired, override

from langchain_openai.decisions import (
    Choice,
    ChoiceAnswer,
    DecisionRequest,
    DecisionResponse,
    OpenAIDecisions,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.runtime import Runtime

_QUESTION_NAME = "model_route"


@dataclass(frozen=True)
class ModelChoice:
    """A model available to the router and the criterion for selecting it.

    Args:
        model: LangChain model instance or model string accepted by `init_chat_model`.
        criteria: Description of the tasks suited to the model.
    """

    model: str | BaseChatModel
    criteria: str


class _ModelRouterState(AgentState):
    """Agent state used to persist the routing answer."""

    model_route: NotRequired[ChoiceAnswer | None]


@beta()
class OpenAIModelRouterMiddleware(AgentMiddleware[_ModelRouterState]):
    """Select an agent's model with an OpenAI Decisions `Choice` question.

    Classifies the latest human message once before each agent run, stores the
    complete `ChoiceAnswer` in agent state under `model_route`, and uses the selected
    route for every model call in the run. Keeping the complete answer makes
    probabilities and confidence available in state and traces.

    When no route is selected, because the model refused or the state has no human
    message, `model_route` is set to `None` and model calls use the agent's own
    model. API errors propagate and terminate the run.

    !!! warning

        This middleware is experimental. Its API may change without notice.

    Args:
        choices: Named model choices, each containing a LangChain model or model
            string and the criterion for selecting it.
        instructions: Routing question sent to the Decisions API.
        model: Decisions model name, or a configured `OpenAIDecisions` instance.

    ??? example "Route agent calls by task"

        ```python
        from langchain.agents import create_agent
        from langchain_openai.middleware import (
            ModelChoice,
            OpenAIModelRouterMiddleware,
        )

        router = OpenAIModelRouterMiddleware(
            choices={
                "fast": ModelChoice(
                    model="openai:gpt-5.4-mini",
                    criteria="Simple, well-scoped tasks.",
                ),
                "powerful": ModelChoice(
                    model="openai:gpt-6-sol",
                    criteria="Complex tasks requiring deeper reasoning.",
                ),
            },
            instructions="Choose the least costly model suited to the task.",
            model="gpt-6-luna",
        )
        agent = create_agent("openai:gpt-6-sol", middleware=[router])
        ```
    """

    state_schema = _ModelRouterState
    trace_policy = TracePolicy(process_inputs=omit_payload)
    """Exclude user messages from middleware traces."""

    def __init__(
        self,
        *,
        choices: Mapping[str, ModelChoice],
        instructions: str,
        model: str | OpenAIDecisions,
    ) -> None:
        """Initialize the model router.

        Args:
            choices: Named model choices and the criterion for selecting each.
            instructions: Routing question sent to the Decisions API.
            model: Decisions model name, or a configured `OpenAIDecisions` instance.

        Raises:
            ValueError: If `choices` is empty or `instructions` is blank.
        """
        if not choices:
            msg = "`choices` must contain at least one model choice."
            raise ValueError(msg)
        if not instructions.strip():
            msg = "`instructions` must not be empty."
            raise ValueError(msg)
        self.question = Choice(
            instructions=instructions,
            choices={route: choice.criteria for route, choice in choices.items()},
        )
        self.models = {
            route: init_chat_model(choice.model)
            if isinstance(choice.model, str)
            else choice.model
            for route, choice in choices.items()
        }
        self.decisions = (
            OpenAIDecisions(model=model) if isinstance(model, str) else model
        )

    @override
    def before_agent(
        self, state: _ModelRouterState, runtime: Runtime[ContextT]
    ) -> dict[str, Any]:
        """Classify the latest task and store the routing answer."""
        request = self._decision_request(state)
        if request is None:
            return {"model_route": None}
        return {"model_route": _route(self.decisions.invoke(request))}

    @override
    async def abefore_agent(
        self, state: _ModelRouterState, runtime: Runtime[ContextT]
    ) -> dict[str, Any]:
        """Classify the latest task asynchronously and store the routing answer."""
        request = self._decision_request(state)
        if request is None:
            return {"model_route": None}
        return {"model_route": _route(await self.decisions.ainvoke(request))}

    @override
    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Route a synchronous model call to the selected model."""
        return handler(self._routed(request))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]
        ],
    ) -> ModelResponse[ResponseT]:
        """Route an asynchronous model call to the selected model."""
        return await handler(self._routed(request))

    def _decision_request(self, state: _ModelRouterState) -> DecisionRequest | None:
        message = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        if message is None:
            return None
        return {"input": message, "questions": {_QUESTION_NAME: self.question}}

    def _routed(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        answer = cast("ChoiceAnswer | None", request.state.get("model_route"))
        if answer is None:
            return request
        return request.override(model=self.models[str(answer.choice)])


def _route(response: DecisionResponse) -> ChoiceAnswer | None:
    return response.choices.get(_QUESTION_NAME)


__all__ = ["ModelChoice", "OpenAIModelRouterMiddleware"]

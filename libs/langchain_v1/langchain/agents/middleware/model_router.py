"""Model-routing middleware driven by a decision model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NotRequired

from langchain_core._api import beta
from langchain_core.decisions import BaseDecisionModel, Choice, ChoiceAnswer, Content, Question
from langchain_core.language_models import BaseChatModel  # noqa: TC002  # used by pydantic
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, JsonValue
from typing_extensions import override

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

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from langgraph.runtime import Runtime

_QUESTION_ID = "model_route"


@dataclass(frozen=True)
class ModelChoice:
    """A model available to the router and the criterion for selecting it.

    Args:
        model: LangChain model instance or model string accepted by `init_chat_model`.
        criteria: Description of the tasks suited to the model.
    """

    model: str | BaseChatModel
    criteria: JsonValue


class _ModelRouterConfig(BaseModel):
    """Validated model-router configuration."""

    choices: dict[str, ModelChoice] = Field(min_length=1)
    instructions: Content


def _routing_questions(config: _ModelRouterConfig) -> dict[str, Question]:
    """Build the routing question from validated middleware configuration."""
    return {
        _QUESTION_ID: Choice(
            instructions=config.instructions,
            criteria={route: choice.criteria for route, choice in config.choices.items()},
        )
    }


class _ModelRouterState(AgentState):
    """Agent state used to persist the routing answer."""

    model_route: NotRequired[ChoiceAnswer]


@beta(obj_type="middleware")
class ModelRouterMiddleware(AgentMiddleware[_ModelRouterState]):
    """Select an agent's model with a decision model's `Choice` answer.

    The middleware classifies the latest human message once before an agent run,
    stores the complete `ChoiceAnswer` in agent state, and uses its selected label for
    every model call in the run. Keeping the complete answer makes probabilities and
    confidence available in state and traces. Classifier failures propagate and
    terminate the run rather than silently selecting a different model.

    !!! warning "Beta"

        This middleware is in beta. Its API may change without notice.

    Args:
        choices: Named model choices, each containing a LangChain model or model string
            and the criterion for selecting it.
        instructions: Instructions the classifier should follow when selecting a route.
        classifier: Decision model used to select a route.

    Raises:
        pydantic.ValidationError: If no model choices are provided.

    ??? example "Route agent calls by task"

        ```python
        from langchain.agents import create_agent
        from langchain.agents.middleware import ModelChoice, ModelRouterMiddleware

        router = ModelRouterMiddleware(
            choices={
                "fast": ModelChoice(
                    model="openai:gpt-5-mini",
                    criteria="Simple, well-scoped tasks.",
                ),
                "powerful": ModelChoice(
                    model=powerful_model,
                    criteria="Complex tasks requiring deeper reasoning.",
                ),
            },
            instructions="Choose the least costly model suited to the task.",
            classifier=classifier,
        )
        agent = create_agent("openai:gpt-5-mini", middleware=[router])
        ```
    """

    state_schema = _ModelRouterState
    trace_policy = TracePolicy(process_inputs=omit_payload)

    def __init__(
        self,
        *,
        choices: Mapping[str, ModelChoice],
        instructions: Content,
        classifier: BaseDecisionModel,
    ) -> None:
        """Initialize the model router."""
        self.config = _ModelRouterConfig.model_validate(
            {"choices": choices, "instructions": instructions}
        )
        self.models = {
            route: init_chat_model(choice.model) if isinstance(choice.model, str) else choice.model
            for route, choice in self.config.choices.items()
        }
        self.classifier = classifier

    @staticmethod
    def _latest_human_message(state: _ModelRouterState) -> HumanMessage:
        """Return the latest human message from agent state.

        Raises:
            ValueError: If the agent state contains no human message.
        """
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage):
                return message
        msg = "ModelRouterMiddleware requires a HumanMessage in the agent state to route on."
        raise ValueError(msg)

    @override
    def before_agent(
        self, state: _ModelRouterState, runtime: Runtime[ContextT]
    ) -> dict[str, ChoiceAnswer]:
        """Classify the latest task and store the complete routing answer."""
        response = self.classifier.invoke(
            {
                "state": self._latest_human_message(state),
                "questions": _routing_questions(self.config),
            }
        )
        return {"model_route": response.choices[_QUESTION_ID]}

    @override
    async def abefore_agent(
        self, state: _ModelRouterState, runtime: Runtime[ContextT]
    ) -> dict[str, ChoiceAnswer]:
        """Classify the latest task asynchronously and store the routing answer."""
        response = await self.classifier.ainvoke(
            {
                "state": self._latest_human_message(state),
                "questions": _routing_questions(self.config),
            }
        )
        return {"model_route": response.choices[_QUESTION_ID]}

    @override
    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Route a synchronous model call to the selected model."""
        answer: ChoiceAnswer = request.state["model_route"]  # type: ignore[typeddict-item]
        return handler(request.override(model=self.models[answer.choice]))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Route an asynchronous model call to the selected model."""
        answer: ChoiceAnswer = request.state["model_route"]  # type: ignore[typeddict-item]
        return await handler(request.override(model=self.models[answer.choice]))


__all__ = ["ModelChoice", "ModelRouterMiddleware"]

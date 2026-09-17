"""Experimental model-routing middleware powered by TypeSafe."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

from langchain.agents.middleware import Runtime
from langchain.agents.middleware.types import ContextT

try:
    from langchain.agents.middleware.types import (
        AgentMiddleware,
        AgentState,
        ModelRequest,
        ModelResponse,
        ResponseT,
        TracePolicy,
        omit_payload,
    )
    from langchain.chat_models import init_chat_model
except ImportError as error:
    msg = (
        "ModelRouterMiddleware requires the LangChain agent framework. "
        "Install it with `pip install 'langchain-typesafe[experimental]'`."
    )
    raise ImportError(msg) from error

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, JsonValue
from typing_extensions import NotRequired, override

from langchain_typesafe.classifier import TypeSafeClassifier
from langchain_typesafe.types import Choice, ChoiceAnswer

_QUESTION_ID = "model_route"
_QuestionContent = str | dict[str, JsonValue] | list[JsonValue]


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
    instructions: _QuestionContent


class _ModelRouterState(AgentState):
    """Agent state used to persist the TypeSafe routing answer."""

    model_route: NotRequired[ChoiceAnswer]


class ModelRouterMiddleware(AgentMiddleware[_ModelRouterState]):
    """Select an agent's model with a TypeSafe `Choice` classification.

    The middleware classifies the latest human message once before an agent run,
    stores the complete `ChoiceAnswer` in agent state, and uses its selected label
    for every model call in the run. Keeping the complete answer makes probabilities
    and confidence available in state and traces. Classifier failures propagate and
    terminate the run rather than silently selecting a different model.

    !!! warning

        This middleware is experimental. Its API may change without notice.

    Install the experimental extra to use this class:

    ```bash
    pip install "langchain-typesafe[experimental]"
    ```

    Args:
        choices: Named model choices, each containing a LangChain model or model
            string and the criterion for selecting it.
        instructions: Additional instructions TypeSafe should follow when selecting a
            route.

    Raises:
        pydantic.ValidationError: If no model choices are provided.

    ??? example "Route agent calls by task"

        ```python
        from langchain.agents import create_agent
        from langchain_typesafe.experimental.middleware import (
            ModelChoice,
            ModelRouterMiddleware,
        )

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
        instructions: _QuestionContent,
    ) -> None:
        """Initialize the model router."""
        self.config = _ModelRouterConfig.model_validate(
            {"choices": choices, "instructions": instructions}
        )
        self.models = {
            route: init_chat_model(choice.model)
            if isinstance(choice.model, str)
            else choice.model
            for route, choice in self.config.choices.items()
        }
        self.classifier = TypeSafeClassifier(
            questions={
                _QUESTION_ID: Choice(
                    instructions=self.config.instructions,
                    criteria={
                        route: choice.criteria
                        for route, choice in self.config.choices.items()
                    },
                )
            }
        )

    @staticmethod
    def _latest_human_message(state: _ModelRouterState) -> HumanMessage:
        """Return the latest human message from agent state."""
        return next(
            message
            for message in reversed(state["messages"])
            if isinstance(message, HumanMessage)
        )

    @override
    def before_agent(
        self, state: _ModelRouterState, runtime: Runtime[ContextT]
    ) -> dict[str, ChoiceAnswer]:
        """Classify the latest task and store the complete routing answer."""
        response = self.classifier.invoke(self._latest_human_message(state))
        return {"model_route": response.choices[_QUESTION_ID]}

    @override
    async def abefore_agent(
        self, state: _ModelRouterState, runtime: Runtime[ContextT]
    ) -> dict[str, ChoiceAnswer]:
        """Classify the latest task asynchronously and store the routing answer."""
        response = await self.classifier.ainvoke(self._latest_human_message(state))
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
        handler: Callable[
            [ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]
        ],
    ) -> ModelResponse[ResponseT]:
        """Route an asynchronous model call to the selected model."""
        answer: ChoiceAnswer = request.state["model_route"]  # type: ignore[typeddict-item]
        return await handler(request.override(model=self.models[answer.choice]))


__all__ = ["ModelChoice", "ModelRouterMiddleware"]

"""Configurable model routing for agents."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, cast

from langchain_core._api import beta
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from typing_extensions import NotRequired, TypedDict, override

from langchain.agents.middleware.internal_call_transformer import (
    InternalCallTransformer,
    internal_call_metadata,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain.chat_models import init_chat_model

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

    from langchain_core.runnables import Runnable
    from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTIONS = "Choose the least expensive model likely to complete the user's task."


class ModelRoutingConfig(TypedDict):
    """Beta, experimental model and routing criteria; no compatibility guarantees."""

    model: str | BaseChatModel
    criteria: str


class ModelRoutingInput(TypedDict):
    """Beta, experimental classifier input; no compatibility guarantees."""

    messages: list[BaseMessage]
    instructions: str
    criteria: dict[str, str]


class ModelRoutingOutput(TypedDict):
    """Select a model route for the user's task.

    Args:
        route: The selected model route.
    """

    route: str


class ModelRoutingState(AgentState):
    """Experimental checkpointed route; clear `model_route` to select again."""

    model_route: NotRequired[str | None]


@beta(
    addendum=(
        "Experimental API: may change or be removed without notice; no compatibility guarantees."
    )
)
class ModelRoutingMiddleware(AgentMiddleware[ModelRoutingState, ContextT, ResponseT]):
    """Route agent model calls using a structured-output LLM or classification runnable.

    !!! warning "Beta / Experimental"
        This middleware and its input schema may change or be removed without notice.
        No compatibility guarantees are provided.

    Routes are selected before the first model call and persisted in `model_route`,
    keeping the same model throughout tool loops and checkpoint resumes. Clear this
    state field (set it to `None`) to route a new task. Selection is never cached on
    the middleware instance. Plug this middleware into `create_agent`; applications
    can customize routing input and backends without invoking selection directly.
    Place this middleware before model fallback middleware so fallbacks receive the
    selected model. Candidate models must support the agent's tools and output format.

    ??? example "Route agent calls by task"

        ```python
        middleware = ModelRoutingMiddleware(
            models={
                "fast": {"model": fast_model, "criteria": "Direct lookups"},
                "reasoning": {
                    "model": reasoning_model,
                    "criteria": "Architectural tradeoffs",
                },
            },
            decision_model=selector_model,
            instructions="Use the least expensive model that can complete the task safely.",
        )
        agent = create_agent(model=fast_model, middleware=[middleware])
        ```
    """

    state_schema = ModelRoutingState
    transformers = (InternalCallTransformer,)

    def __init__(
        self,
        *,
        models: Mapping[str, ModelRoutingConfig],
        decision_model: str | BaseChatModel | Runnable[ModelRoutingInput, ModelRoutingOutput],
        instructions: str = DEFAULT_INSTRUCTIONS,
        input_extractor: Callable[[ModelRoutingState], Sequence[BaseMessage]] | None = None,
        fallback_route: str | None = None,
    ) -> None:
        """Initialize routing with a chat model or custom decision runnable.

        Args:
            models: Route names mapped to configurations containing a `model` instance
                or identifier string and its selection `criteria`.
            decision_model: Chat model or model string supporting `with_structured_output`,
                or a custom runnable accepting `ModelRoutingInput` and returning
                `ModelRoutingOutput`. Chat models use the built-in routing prompt and
                output schema.
            instructions: Base instructions for selection, separate from the agent prompt.
            input_extractor: Routing messages extracted from agent state. By default,
                uses the latest human message. Customize to filter application metadata.
            fallback_route: Route used for malformed or unknown selections. Without it,
                invalid selections raise `ValueError`. Backend and extractor exceptions
                always propagate; configure retries or fallbacks on the backend runnable.

        Raises:
            ValueError: If routes are empty or the fallback is unknown.
        """
        super().__init__()
        if not models or any(not route for route in models):
            msg = "models must contain non-empty route names"
            raise ValueError(msg)
        if fallback_route is not None and fallback_route not in models:
            msg = "fallback_route must be a configured route name"
            raise ValueError(msg)
        self.models = {
            route: (
                init_chat_model(config["model"])
                if isinstance(config["model"], str)
                else config["model"]
            )
            for route, config in models.items()
        }
        self.criteria = {route: config["criteria"] for route, config in models.items()}
        self.instructions = instructions
        self.input_extractor = input_extractor
        self.fallback_route = fallback_route
        self.decision_model: Runnable[ModelRoutingInput, ModelRoutingOutput]
        if isinstance(decision_model, (str, BaseChatModel)):
            self.decision_model = self._create_decision_model(decision_model)
        else:
            self.decision_model = decision_model

    def _create_decision_model(
        self, model: str | BaseChatModel
    ) -> Runnable[ModelRoutingInput, ModelRoutingOutput]:
        """Adapt a structured-output chat model to the decision runnable interface."""
        if isinstance(model, str):
            model = init_chat_model(model)
        return cast(
            "Runnable[ModelRoutingInput, ModelRoutingOutput]",
            self._llm_input
            | model.with_structured_output(
                {
                    "title": "ModelRoutingResponse",
                    "description": "Select a model route for the user's task.",
                    "type": "object",
                    "properties": {
                        "route": {
                            "type": "string",
                            "enum": list(self.models),
                            "description": "The selected model route.",
                        }
                    },
                    "required": ["route"],
                    "additionalProperties": False,
                }
            ),
        )

    def _extract_route(self, response: object) -> str | None:
        """Extract a string route, or return `None` for malformed output."""
        route = response.get("route") if isinstance(response, dict) else None
        return route if isinstance(route, str) else None

    def _routing_input(self, state: ModelRoutingState) -> ModelRoutingInput:
        """Extract application input without modifying the agent request."""
        if self.input_extractor is not None:
            messages = list(self.input_extractor(state))
        else:
            messages = next(
                (
                    [message]
                    for message in reversed(state["messages"])
                    if isinstance(message, HumanMessage)
                ),
                [],
            )
        if not messages:
            msg = "No routing messages found; provide input_extractor for non-human inputs"
            raise ValueError(msg)
        return {
            "messages": messages,
            "instructions": self.instructions,
            "criteria": dict(self.criteria),
        }

    def _llm_input(self, inputs: ModelRoutingInput) -> list[BaseMessage]:
        """Present configured criteria as data alongside base routing instructions."""
        prompt = (
            f"{inputs['instructions']}\n\nModel routing criteria:\n{json.dumps(inputs['criteria'])}"
        )
        return [SystemMessage(content=prompt), *inputs["messages"]]

    def _validate_route(self, route: object) -> str:
        """Resolve only configured routes, with an explicit fallback for invalid output."""
        if isinstance(route, str) and route in self.models:
            return route
        if self.fallback_route is not None:
            logger.warning("Invalid model routing selection; using configured fallback route")
            return self.fallback_route
        msg = "Model routing selection must be a configured route name"
        raise ValueError(msg)

    @override
    def before_model(self, state: ModelRoutingState, runtime: Runtime[ContextT]) -> dict[str, str]:
        """Select or reuse a route before the model runs."""
        route = state.get("model_route")
        if route is None:
            response = self.decision_model.invoke(
                self._routing_input(state), config={"metadata": internal_call_metadata()}
            )
            route = self._extract_route(response)
        return {"model_route": self._validate_route(route)}

    @override
    async def abefore_model(
        self, state: ModelRoutingState, runtime: Runtime[ContextT]
    ) -> dict[str, str]:
        """Select or reuse a route asynchronously before the model runs."""
        route = state.get("model_route")
        if route is None:
            response = await self.decision_model.ainvoke(
                self._routing_input(state), config={"metadata": internal_call_metadata()}
            )
            route = self._extract_route(response)
        return {"model_route": self._validate_route(route)}

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Call the handler with the selected model."""
        route = self._validate_route(request.state.get("model_route"))
        return handler(request.override(model=self.models[route]))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Call the async handler with the selected model."""
        route = self._validate_route(request.state.get("model_route"))
        return await handler(request.override(model=self.models[route]))

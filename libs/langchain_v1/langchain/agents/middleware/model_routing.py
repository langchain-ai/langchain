"""Configurable model routing for agents."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from langchain_core._api import beta
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from typing_extensions import TypedDict

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

    from langchain_core.language_models import BaseChatModel, LanguageModelInput
    from langchain_core.runnables import Runnable, RunnableConfig

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = "Choose the least expensive model likely to complete the user's task."


class ModelRoutingConfig(TypedDict):
    """Beta, experimental model and routing criteria; no compatibility guarantees."""

    model: str | BaseChatModel
    criteria: str


class ModelRoutingInput(TypedDict):
    """Beta, experimental classifier input; no compatibility guarantees."""

    messages: list[BaseMessage]
    system_prompt: str
    criteria: dict[str, str]


@beta(
    addendum=(
        "Experimental API: may change or be removed without notice; no compatibility guarantees."
    )
)
class ModelRoutingMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """Route agent model calls using a structured-output LLM or classification runnable.

    !!! warning "Beta / Experimental"
        This middleware and its input schema may change or be removed without notice.
        No compatibility guarantees are provided.

    Routes are selected for each model call, without shared or persisted selection state.
    Applications requiring one selection per turn can call `select_route` or
    `aselect_route` during preparation and persist the result in their own state.
    Place this middleware before model fallback middleware so fallbacks receive the
    selected model. Candidate models must support the agent's tools and output format.

    Example:
        ```python
        middleware = ModelRoutingMiddleware(
            models={
                "fast": {"model": fast_model, "criteria": "Direct lookups"},
                "reasoning": {
                    "model": reasoning_model,
                    "criteria": "Architectural tradeoffs",
                },
            },
            routing_model=selector_model,
            system_prompt="Use the least expensive model that can complete the task safely.",
        )
        agent = create_agent(model=fast_model, middleware=[middleware])
        ```
    """

    transformers = (InternalCallTransformer,)

    def __init__(
        self,
        *,
        models: Mapping[str, ModelRoutingConfig],
        routing_model: str | BaseChatModel | None = None,
        decision_model: Runnable[ModelRoutingInput, str] | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        input_extractor: Callable[[ModelRequest[ContextT]], Sequence[BaseMessage]] | None = None,
        fallback_route: str | None = None,
    ) -> None:
        """Initialize routing with exactly one selection backend.

        Args:
            models: Route names mapped to configurations containing a `model` instance
                or identifier string and its selection `criteria`.
            routing_model: Chat model supporting `with_structured_output`.
            decision_model: Classification runnable accepting `ModelRoutingInput` and
                returning a route name. Adapt provider-specific classifiers with a
                runnable; no classifier dependency is required.
            system_prompt: Base instructions for selection, separate from the agent prompt.
            input_extractor: Routing messages extracted from the request. By default,
                uses the latest human message. Customize to filter application metadata.
            fallback_route: Route used for malformed or unknown selections. Without it,
                invalid selections raise `ValueError`. Backend and extractor exceptions
                always propagate; configure retries or fallbacks on the backend runnable.

        Raises:
            ValueError: If routes are empty, the fallback is unknown, or exactly one
                selection backend is not supplied.
        """
        super().__init__()
        if not models or any(not route for route in models):
            msg = "models must contain non-empty route names"
            raise ValueError(msg)
        if (routing_model is None) == (decision_model is None):
            msg = "Provide exactly one of routing_model or decision_model"
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
        self.system_prompt = system_prompt
        self.input_extractor = input_extractor
        self.fallback_route = fallback_route
        self.decision_model = decision_model
        self._routing_model: Runnable[LanguageModelInput, object] | None = None
        if routing_model is not None:
            model = (
                init_chat_model(routing_model) if isinstance(routing_model, str) else routing_model
            )
            self._routing_model = model.with_structured_output(
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
            )

    def _routing_input(self, request: ModelRequest[ContextT]) -> ModelRoutingInput:
        """Extract application input without modifying the agent request."""
        if self.input_extractor is not None:
            messages = list(self.input_extractor(request))
        else:
            messages = next(
                (
                    [message]
                    for message in reversed(request.messages)
                    if isinstance(message, HumanMessage)
                ),
                [],
            )
        if not messages:
            msg = "No routing messages found; provide input_extractor for non-human inputs"
            raise ValueError(msg)
        return {
            "messages": messages,
            "system_prompt": self.system_prompt,
            "criteria": dict(self.criteria),
        }

    def _llm_input(self, inputs: ModelRoutingInput) -> list[BaseMessage]:
        """Present configured criteria as data alongside base routing instructions."""
        prompt = (
            f"{inputs['system_prompt']}\n\nModel routing criteria:\n"
            f"{json.dumps(inputs['criteria'])}"
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

    def select_route(self, request: ModelRequest[ContextT]) -> str:
        """Select a route without changing the request.

        Args:
            request: Agent request supplying routing input.

        Returns:
            A configured route name.
        """
        inputs = self._routing_input(request)
        config: RunnableConfig = {"metadata": internal_call_metadata()}
        if self.decision_model is not None:
            return self._validate_route(self.decision_model.invoke(inputs, config=config))
        if self._routing_model is None:
            msg = "No routing backend configured"
            raise AssertionError(msg)
        response = self._routing_model.invoke(self._llm_input(inputs), config=config)
        return self._validate_route(response.get("route") if isinstance(response, dict) else None)

    async def aselect_route(self, request: ModelRequest[ContextT]) -> str:
        """Select a route asynchronously without changing the request.

        Args:
            request: Agent request supplying routing input.

        Returns:
            A configured route name.
        """
        inputs = self._routing_input(request)
        config: RunnableConfig = {"metadata": internal_call_metadata()}
        if self.decision_model is not None:
            return self._validate_route(await self.decision_model.ainvoke(inputs, config=config))
        if self._routing_model is None:
            msg = "No routing backend configured"
            raise AssertionError(msg)
        response = await self._routing_model.ainvoke(self._llm_input(inputs), config=config)
        return self._validate_route(response.get("route") if isinstance(response, dict) else None)

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Call the handler with the selected model, preserving all other request fields.

        Args:
            request: Agent model request.
            handler: Handler for the selected model request.

        Returns:
            The selected model's response.
        """
        route = self.select_route(request)
        return handler(request.override(model=self.models[route]))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Call the async handler with the selected model, preserving other request fields.

        Args:
            request: Agent model request.
            handler: Async handler for the selected model request.

        Returns:
            The selected model's response.
        """
        route = await self.aselect_route(request)
        return await handler(request.override(model=self.models[route]))

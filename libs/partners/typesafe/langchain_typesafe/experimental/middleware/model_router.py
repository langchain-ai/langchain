"""Experimental model-routing middleware powered by TypeSafe."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

try:
    from langchain.agents.middleware.types import (
        AgentMiddleware,
        AgentState,
        ModelRequest,
        ModelResponse,
        TracePolicy,
        omit_payload,
    )
    from langgraph.runtime import Runtime
except ImportError as error:
    msg = (
        "ModelRouterMiddleware requires the LangChain agent framework. "
        "Install it with `pip install 'langchain-typesafe[experimental]'`."
    )
    raise ImportError(msg) from error

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import JsonValue
from typing_extensions import NotRequired, override

from langchain_typesafe.classifier import TypeSafeClassifier
from langchain_typesafe.types import Choice, ClassificationResponse

logger = logging.getLogger(__name__)

_QUESTION_ID = "model_route"
_QuestionContent = str | dict[str, JsonValue] | list[JsonValue]


@dataclass(frozen=True)
class ModelChoice:
    """A model available to the router and the criterion for selecting it.

    Args:
        model: LangChain model used when this choice is selected.
        criteria: Description of the tasks suited to the model.
    """

    model: BaseChatModel
    criteria: JsonValue


class _ModelRouterState(AgentState):
    """Agent state used to persist the selected model route."""

    model_route: NotRequired[str]


class ModelRouterMiddleware(AgentMiddleware[_ModelRouterState]):
    """Select an agent's model with a TypeSafe `Choice` classification.

    The middleware classifies the latest human message once before an agent run,
    stores the selected route in agent state, and uses that route for every model
    call in the run. Classification failures and unrecognized routes fall back to
    `default_route`.

    !!! warning

        This middleware is experimental. Its API may change without notice.

    Install the `middleware` extra to use this class:

    ```bash
    pip install "langchain-typesafe[experimental]"
    ```

    Args:
        choices: Named model choices, each containing a LangChain model and the
            criterion for selecting it.
        instructions: Additional instructions TypeSafe should follow when selecting a
            route.
        default_route: Route used when classification fails or does not select a
            configured model.

    Raises:
        ValueError: If the model mapping and criteria do not define valid routes.

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain_typesafe.experimental.middleware import (
            ModelChoice,
            ModelRouterMiddleware,
        )

        router = ModelRouterMiddleware(
            choices={
                "fast": ModelChoice(
                    model=fast_model,
                    criteria="Simple, well-scoped tasks.",
                ),
                "powerful": ModelChoice(
                    model=powerful_model,
                    criteria="Complex tasks requiring deeper reasoning.",
                ),
            },
            instructions="Choose the least costly model suited to the task.",
            default_route="powerful",
        )
        agent = create_agent(fast_model, middleware=[router])
        ```
    """

    state_schema = _ModelRouterState  # type: ignore[assignment]
    trace_policy = TracePolicy(process_inputs=omit_payload)

    def __init__(
        self,
        *,
        choices: Mapping[str, ModelChoice],
        instructions: _QuestionContent,
        default_route: str,
    ) -> None:
        """Initialize the model router."""
        super().__init__()
        self.choices = dict(choices)
        self.default_route = default_route
        self._validate_configuration()
        self.classifier = TypeSafeClassifier(
            questions={
                _QUESTION_ID: Choice(
                    instructions=instructions,
                    criteria={
                        route: choice.criteria for route, choice in self.choices.items()
                    },
                )
            }
        )

    def _validate_configuration(self) -> None:
        """Validate that the choices define a default route."""
        if not self.choices:
            msg = "At least one model choice is required."
            raise ValueError(msg)
        if self.default_route not in self.choices:
            msg = f"Default route {self.default_route!r} is not present in `choices`."
            raise ValueError(msg)

    def _classification_input(self, state: _ModelRouterState) -> HumanMessage | None:
        """Return the latest human message from agent state."""
        return next(
            (
                message
                for message in reversed(state.get("messages", []))
                if isinstance(message, HumanMessage)
            ),
            None,
        )

    def _resolve_route(self, response: ClassificationResponse) -> str:
        """Resolve a configured route from a classification response."""
        answer = response.choices.get(_QUESTION_ID)
        if answer is not None and answer.choice in self.choices:
            return answer.choice
        logger.warning(
            "TypeSafe model router received no configured route for question %r; "
            "using default route %r",
            _QUESTION_ID,
            self.default_route,
        )
        return self.default_route

    def _classification_config(self) -> RunnableConfig:
        """Return tracing metadata for the internal classification call."""
        return {"metadata": {"lc_source": "typesafe_model_router"}}

    @override
    def before_agent(
        self,
        state: _ModelRouterState,
        runtime: Runtime[Any],
    ) -> dict[str, str]:
        """Classify the latest task and store its model route."""
        del runtime
        classifier_input = self._classification_input(state)
        if classifier_input is None:
            return {"model_route": self.default_route}
        try:
            response = self.classifier.invoke(
                classifier_input,
                config=self._classification_config(),
            )
            route = self._resolve_route(response)
        except Exception:
            logger.exception(
                "TypeSafe model routing classification failed; using default route %r",
                self.default_route,
            )
            route = self.default_route
        return {"model_route": route}

    @override
    async def abefore_agent(
        self,
        state: _ModelRouterState,
        runtime: Runtime[Any],
    ) -> dict[str, str]:
        """Classify the latest task asynchronously and store its model route."""
        del runtime
        classifier_input = self._classification_input(state)
        if classifier_input is None:
            return {"model_route": self.default_route}
        try:
            response = await self.classifier.ainvoke(
                classifier_input,
                config=self._classification_config(),
            )
            route = self._resolve_route(response)
        except Exception:
            logger.exception(
                "TypeSafe model routing classification failed; using default route %r",
                self.default_route,
            )
            route = self.default_route
        return {"model_route": route}

    def _route_request(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        """Return a model request overridden with the selected routed model."""
        route = request.state.get("model_route", self.default_route)
        choice = self.choices.get(route) if isinstance(route, str) else None
        selected_choice = (
            choice if choice is not None else self.choices[self.default_route]
        )
        return request.override(model=selected_choice.model)

    @override
    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """Route a synchronous model call to the selected model."""
        return handler(self._route_request(request))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        """Route an asynchronous model call to the selected model."""
        return await handler(self._route_request(request))


__all__ = ["ModelChoice", "ModelRouterMiddleware"]

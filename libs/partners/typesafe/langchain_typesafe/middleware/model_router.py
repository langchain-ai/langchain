"""Model-routing middleware powered by TypeSafe."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

import typesafe_sdk as ts
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    TracePolicy,
    omit_payload,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, convert_to_openai_messages
from langgraph.runtime import Runtime
from pydantic import SecretStr
from typing_extensions import NotRequired, override

from langchain_typesafe._classify import TypeSafeClassifier, log_classification_failure

logger = logging.getLogger(__name__)

_QUESTION_ID = "model_route"


@dataclass(frozen=True)
class ModelChoice:
    """A model available to the router and the criterion for selecting it.

    Args:
        model: LangChain model used when this choice is selected.
        criteria: Description of the tasks suited to the model.
    """

    model: BaseChatModel
    criteria: ts.JSONContent | None


class _ModelRouterState(AgentState):
    """Agent state used to persist the selected model route."""

    model_route: NotRequired[str]


class ModelRouterMiddleware(AgentMiddleware[_ModelRouterState]):
    """Select an agent's model with a TypeSafe `Choice` classification.

    The middleware classifies the latest human message once before an agent run,
    stores the selected route in agent state, and uses that route for every model
    call in the run. Classification failures and unrecognized routes fall back to
    `default_route`, so routing never blocks an agent from running.

    !!! warning

        This middleware is experimental. Its API may change without notice.

    Args:
        choices: Named model choices, each containing a LangChain model and the
            criterion for selecting it.
        instructions: Instructions TypeSafe should follow when selecting a route.
        default_route: Route used when classification fails or does not select a
            configured model.
        api_key: TypeSafe API key. If omitted, reads `TYPESAFE_API_KEY`.
        base_url: Root URL for the TypeSafe API.
        model: TypeSafe model used for the routing decision.
        timeout: Timeout in seconds for the routing request.
        retry: Retry policy for the routing request.
        client: Optional synchronous TypeSafe client.
        async_client: Optional asynchronous TypeSafe client.

    Raises:
        ValueError: If `choices` is empty or `default_route` is not one of them.

    ??? example "Route between a fast and a powerful model"

        ```python
        from langchain.agents import create_agent
        from langchain_typesafe import ModelChoice, ModelRouterMiddleware

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
    """Keep the classified request payload out of middleware traces."""

    def __init__(
        self,
        *,
        choices: Mapping[str, ModelChoice],
        instructions: ts.JSONContent,
        default_route: str,
        api_key: SecretStr | str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        retry: ts.RetryPolicy | None = None,
        client: ts.TypeSafeClient | None = None,
        async_client: ts.AsyncTypeSafeClient | None = None,
    ) -> None:
        """Initialize the model router."""
        super().__init__()
        self.choices = dict(choices)
        self.default_route = default_route
        if not self.choices:
            msg = "At least one model choice is required."
            raise ValueError(msg)
        if self.default_route not in self.choices:
            msg = f"Default route {self.default_route!r} is not present in `choices`."
            raise ValueError(msg)
        self._classifier = TypeSafeClassifier(
            {
                _QUESTION_ID: ts.Choice(
                    instructions=instructions,
                    criteria={
                        route: choice.criteria for route, choice in self.choices.items()
                    },
                )
            },
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            retry=retry,
            client=client,
            async_client=async_client,
        )

    def _classification_state(self, state: _ModelRouterState) -> dict[str, Any] | None:
        """Return the latest human message as classifiable state."""
        message = next(
            (
                message
                for message in reversed(state.get("messages", []))
                if isinstance(message, HumanMessage)
            ),
            None,
        )
        if message is None:
            return None
        return convert_to_openai_messages(message)

    def _resolve_route(self, response: ts.SystemOneResponse) -> str:
        """Resolve a configured route from a classification response."""
        answer = response.choices.get(_QUESTION_ID)
        if answer is not None and answer.choice in self.choices:
            return answer.choice
        logger.warning(
            "TypeSafe model router did not return a configured route; using %r",
            self.default_route,
        )
        return self.default_route

    @override
    def before_agent(
        self,
        state: _ModelRouterState,
        runtime: Runtime[Any],
    ) -> dict[str, str]:
        """Classify the latest task and store its model route."""
        del runtime
        classifier_state = self._classification_state(state)
        if classifier_state is None:
            return {"model_route": self.default_route}
        try:
            route = self._resolve_route(self._classifier.classify(classifier_state))
        except Exception as error:  # noqa: BLE001 - routing must not break the agent
            log_classification_failure(
                logger,
                error,
                f"using default route {self.default_route!r}",
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
        classifier_state = self._classification_state(state)
        if classifier_state is None:
            return {"model_route": self.default_route}
        try:
            response = await self._classifier.aclassify(classifier_state)
            route = self._resolve_route(response)
        except Exception as error:  # noqa: BLE001 - routing must not break the agent
            log_classification_failure(
                logger,
                error,
                f"using default route {self.default_route!r}",
            )
            route = self.default_route
        return {"model_route": route}

    def _route_request(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        """Return a model request overridden with the selected routed model."""
        route = request.state.get("model_route", self.default_route)
        choice = self.choices.get(route) if isinstance(route, str) else None
        selected = choice if choice is not None else self.choices[self.default_route]
        return request.override(model=selected.model)

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

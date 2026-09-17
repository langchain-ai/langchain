"""Experimental model-routing middleware powered by TypeSafe."""

from __future__ import annotations

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
    from langchain.chat_models import init_chat_model
    from langgraph.runtime import Runtime
except ImportError as error:
    msg = (
        "ModelRouterMiddleware requires the LangChain agent framework. "
        "Install it with `pip install 'langchain-typesafe[experimental]'`."
    )
    raise ImportError(msg) from error

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import JsonValue
from typing_extensions import NotRequired, override

from langchain_typesafe.classifier import TypeSafeClassifier
from langchain_typesafe.types import Choice, ClassificationResponse

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


class _ModelRouterState(AgentState):
    """Agent state used to persist the selected model route."""

    model_route: NotRequired[str]


class ModelRouterMiddleware(AgentMiddleware[_ModelRouterState]):
    """Select an agent's model with a TypeSafe `Choice` classification.

    The middleware classifies the latest human message once before an agent run,
    stores the selected route in agent state, and uses that route for every model
    call in the run. Classification and routing failures terminate the run rather
    than silently selecting a different model.

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
        ValueError: If no model choices are provided, no human message is available,
            or TypeSafe does not select a configured route.

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

    state_schema = _ModelRouterState  # type: ignore[assignment]
    trace_policy = TracePolicy(process_inputs=omit_payload)

    def __init__(
        self,
        *,
        choices: Mapping[str, ModelChoice],
        instructions: _QuestionContent,
    ) -> None:
        """Initialize the model router."""
        super().__init__()
        self.choices = dict(choices)
        if not self.choices:
            msg = "At least one model choice is required."
            raise ValueError(msg)

        self.models = {
            route: init_chat_model(choice.model)
            if isinstance(choice.model, str)
            else choice.model
            for route, choice in self.choices.items()
        }
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

    def _classification_input(self, state: _ModelRouterState) -> HumanMessage:
        """Return the latest human message from agent state.

        Raises:
            ValueError: If the state does not contain a human message to classify.
        """
        message = next(
            (
                message
                for message in reversed(state.get("messages", []))
                if isinstance(message, HumanMessage)
            ),
            None,
        )
        if message is None:
            msg = "Model routing requires at least one human message."
            raise ValueError(msg)
        return message

    def _resolve_route(self, response: ClassificationResponse) -> str:
        """Resolve a configured route from a classification response.

        Raises:
            ValueError: If the response does not select a configured route.
        """
        answer = response.choices.get(_QUESTION_ID)
        if answer is None:
            msg = f"TypeSafe response did not answer {_QUESTION_ID!r}."
            raise ValueError(msg)
        if answer.choice not in self.models:
            msg = f"TypeSafe selected unknown model route {answer.choice!r}."
            raise ValueError(msg)
        return answer.choice

    @override
    def before_agent(
        self,
        state: _ModelRouterState,
        runtime: Runtime[Any],
    ) -> dict[str, str]:
        """Classify the latest task and store its model route."""
        response = self.classifier.invoke(self._classification_input(state))
        return {"model_route": self._resolve_route(response)}

    @override
    async def abefore_agent(
        self,
        state: _ModelRouterState,
        runtime: Runtime[Any],
    ) -> dict[str, str]:
        """Classify the latest task asynchronously and store its model route."""
        response = await self.classifier.ainvoke(self._classification_input(state))
        return {"model_route": self._resolve_route(response)}

    def _route_request(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        """Return a model request overridden with the selected routed model.

        Raises:
            ValueError: If agent state does not contain a configured route.
        """
        route = request.state.get("model_route")
        if not isinstance(route, str) or route not in self.models:
            msg = f"Agent state contains unknown model route {route!r}."
            raise ValueError(msg)
        return request.override(model=self.models[route])

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

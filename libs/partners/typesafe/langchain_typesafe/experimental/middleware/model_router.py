"""Experimental model-routing middleware powered by TypeSafe."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from threading import Lock
from typing import Any
from weakref import ReferenceType, ref

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
from langchain_typesafe.types import Choice, ChoiceAnswer, Question

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


def _routing_questions(config: _ModelRouterConfig) -> dict[str, Question]:
    """Build the routing question from validated middleware configuration."""
    return {
        _QUESTION_ID: Choice(
            instructions=config.instructions,
            criteria={
                route: choice.criteria for route, choice in config.choices.items()
            },
        )
    }


class _ModelRouterState(AgentState):
    """Agent state used to persist the TypeSafe routing answer."""

    model_route: NotRequired[ChoiceAnswer]


_ExecutionKey = tuple[str, str, str, int]


class ModelRouterMiddleware(AgentMiddleware[_ModelRouterState]):
    """Select an agent's model with a TypeSafe `Choice` classification.

    The middleware classifies the latest human message once before an agent run,
    stores the complete `ChoiceAnswer` in agent state, and uses its selected label
    for every model call in the run. Keeping the complete answer makes probabilities
    and confidence available in state and traces. Classifier failures propagate and
    terminate the run rather than silently selecting a different model.

    When an enclosing `ModelRetryMiddleware` retries the original request, the
    selected route is applied again. When an enclosing `ModelFallbackMiddleware`
    supplies a replacement request, its fallback model is preserved.

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
        self.classifier = TypeSafeClassifier()
        self._routed_requests: dict[
            _ExecutionKey, ReferenceType[ModelRequest[Any]]
        ] = {}
        self._routed_requests_lock = Lock()

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

    @staticmethod
    def _execution_key(request: ModelRequest[Any]) -> _ExecutionKey | None:
        """Return the stable identity of the current model-node execution."""
        execution_info = request.runtime.execution_info
        if execution_info is None:
            return None
        return (
            execution_info.checkpoint_id,
            execution_info.checkpoint_ns,
            execution_info.task_id,
            execution_info.node_attempt,
        )

    def _forget_request(
        self,
        key: _ExecutionKey,
        request_ref: ReferenceType[ModelRequest[Any]],
    ) -> None:
        """Remove a completed node without deleting a newer record for the same key."""
        with self._routed_requests_lock:
            current = self._routed_requests.get(key)
            if current is request_ref:
                self._routed_requests.pop(key)

    def _route_request(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        """Route original and retry requests while preserving fallback replacements."""
        answer: ChoiceAnswer = request.state["model_route"]  # type: ignore[typeddict-item]
        routed_model = self.models[answer.choice]
        key = self._execution_key(request)
        if key is None:
            return request.override(model=routed_model)

        with self._routed_requests_lock:
            request_ref = self._routed_requests.get(key)
            original_request = request_ref() if request_ref is not None else None
            if original_request is None:
                request_ref = ref(
                    request,
                    lambda dead_ref: self._forget_request(key, dead_ref),
                )
                self._routed_requests[key] = request_ref
            elif request is not original_request:
                return request

        return request.override(model=routed_model)

    @override
    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Route a synchronous model call to the selected model."""
        return handler(self._route_request(request))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]
        ],
    ) -> ModelResponse[ResponseT]:
        """Route an asynchronous model call to the selected model."""
        return await handler(self._route_request(request))


__all__ = ["ModelChoice", "ModelRouterMiddleware"]

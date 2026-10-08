"""Experimental model-routing middleware powered by the OpenAI Decisions API."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import openai
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
from langchain_core.messages import HumanMessage, convert_to_openai_messages
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
_RETRY_STATUS_CODES = frozenset({400, 413})

logger = logging.getLogger(__name__)


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

    The message text and base64 images are classified. Hosted image URLs, files,
    and audio are replaced with placeholders such as `[image omitted]`. If a request
    with images is rejected (HTTP 400 or 413), routing is retried once with every
    image replaced.

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
                    model="openai:gpt-6-luna",
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
        agent = create_agent("openai:gpt-6-luna", middleware=[router])
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
        requests = self._decision_requests(state)
        if not requests:
            return {"model_route": None}
        for request in requests[:-1]:
            try:
                return {"model_route": _route(self.decisions.invoke(request))}
            except openai.APIStatusError as e:
                if e.status_code not in _RETRY_STATUS_CODES:
                    raise
                logger.debug("Routing with images failed (HTTP %s).", e.status_code)
        return {"model_route": _route(self.decisions.invoke(requests[-1]))}

    @override
    async def abefore_agent(
        self, state: _ModelRouterState, runtime: Runtime[ContextT]
    ) -> dict[str, Any]:
        """Classify the latest task asynchronously and store the routing answer."""
        requests = self._decision_requests(state)
        if not requests:
            return {"model_route": None}
        for request in requests[:-1]:
            try:
                return {"model_route": _route(await self.decisions.ainvoke(request))}
            except openai.APIStatusError as e:
                if e.status_code not in _RETRY_STATUS_CODES:
                    raise
                logger.debug("Routing with images failed (HTTP %s).", e.status_code)
        return {"model_route": _route(await self.decisions.ainvoke(requests[-1]))}

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

    def _decision_requests(self, state: _ModelRouterState) -> list[DecisionRequest]:
        """Return requests to try in order: with images (if any), then text only."""
        message = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        if message is None:
            return []
        text_only = _routable(message, keep_images=False)
        with_images = _routable(message, keep_images=True)
        inputs = (
            [text_only]
            if with_images.content == text_only.content
            else [with_images, text_only]
        )
        return [
            {"input": m, "questions": {_QUESTION_NAME: self.question}} for m in inputs
        ]

    def _routed(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        answer = cast("ChoiceAnswer | None", request.state.get("model_route"))
        if answer is None:
            return request
        return request.override(model=self.models[str(answer.choice)])


def _routable(message: HumanMessage, *, keep_images: bool) -> HumanMessage:
    """Keep text (and base64 images if requested); replace other blocks."""
    if isinstance(message.content, str):
        return message
    content = [
        block
        if isinstance(block, str)
        or block.get("type") == "text"
        or (keep_images and _is_base64_image(block))
        else {"type": "text", "text": f"[{block.get('type', 'content')} omitted]"}
        for block in message.content
    ]
    return message.model_copy(update={"content": content})


def _is_base64_image(block: dict[str, Any]) -> bool:
    if block.get("type") not in {"image", "image_url"}:
        return False
    try:
        part = convert_to_openai_messages(HumanMessage(content=[block]))["content"][0]
    except (ValueError, KeyError, TypeError):
        return False
    url = part.get("image_url", {}).get("url", "") if isinstance(part, dict) else ""
    return url.startswith("data:")


def _route(response: DecisionResponse) -> ChoiceAnswer | None:
    return response.choices.get(_QUESTION_NAME)


__all__ = ["ModelChoice", "OpenAIModelRouterMiddleware"]

"""Tests for `ModelRouterMiddleware`."""

import gc
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware, ModelRetryMiddleware
from langchain.agents.middleware.types import AgentState, InputAgentState, ModelRequest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatResult
from langgraph.runtime import ExecutionInfo, Runtime
from pydantic import ValidationError
from typing_extensions import override

from langchain_typesafe import Choice, ChoiceAnswer
from langchain_typesafe.classifier import TypeSafeClassifier
from langchain_typesafe.experimental.middleware import (
    ModelChoice,
    ModelRouterMiddleware,
)
from langchain_typesafe.experimental.middleware import (
    __all__ as middleware_all,
)
from langchain_typesafe.experimental.middleware.model_router import _routing_questions
from langchain_typesafe.types import ClassifierResponse


class _FailingModel(GenericFakeChatModel):
    """Chat model that always fails and counts attempts."""

    attempts: int = 0

    def __init__(self) -> None:
        """Initialize the model without scripted messages."""
        super().__init__(messages=iter([]))

    @override
    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        """Fail every model invocation."""
        _ = (args, kwargs)
        self.attempts += 1
        msg = "model unavailable"
        raise RuntimeError(msg)


class _FailOnceModel(GenericFakeChatModel):
    """Chat model that succeeds after one transient failure."""

    attempts: int = 0

    @override
    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        """Fail the first invocation and then return the scripted response."""
        self.attempts += 1
        if self.attempts == 1:
            msg = "transient model error"
            raise RuntimeError(msg)
        return super()._generate(*args, **kwargs)


def _response(route: str) -> ClassifierResponse:
    return ClassifierResponse(
        model="jev-latest",
        answers={
            "model_route": ChoiceAnswer(
                type="choice",
                choice=route,
                probabilities={route: 1.0},
                confidence=1.0,
            )
        },
    )


def _router(
    route: str = "fast",
) -> tuple[
    ModelRouterMiddleware,
    dict[str, GenericFakeChatModel],
    MagicMock,
    MagicMock,
]:
    models = {
        "fast": GenericFakeChatModel(messages=iter([AIMessage("fast response")])),
        "powerful": GenericFakeChatModel(
            messages=iter([AIMessage("powerful response")])
        ),
    }
    classifier = MagicMock(spec=TypeSafeClassifier)
    classifier.invoke.return_value = _response(route)
    classifier.ainvoke = AsyncMock(return_value=_response(route))
    with patch(
        "langchain_typesafe.experimental.middleware.model_router.TypeSafeClassifier",
        return_value=classifier,
    ) as classifier_class:
        middleware = ModelRouterMiddleware(
            choices={
                "fast": ModelChoice(model=models["fast"], criteria="Simple tasks."),
                "powerful": ModelChoice(
                    model=models["powerful"], criteria="Complex tasks."
                ),
            },
            instructions="Choose the least costly model suited to the task.",
        )
    return middleware, models, classifier, classifier_class


def test_middleware_constructs_classifier_from_routing_configuration() -> None:
    """Construct a TypeSafe Choice and expose validated configuration fields."""
    middleware, _, classifier, classifier_class = _router()

    classifier_class.assert_called_once_with()
    assert _routing_questions(middleware.config) == {
        "model_route": Choice(
            instructions="Choose the least costly model suited to the task.",
            criteria={"fast": "Simple tasks.", "powerful": "Complex tasks."},
        )
    }
    assert middleware.classifier is classifier
    assert middleware.config.instructions == (
        "Choose the least costly model suited to the task."
    )
    assert set(middleware.config.choices) == {"fast", "powerful"}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_agent_routes_using_latest_human_message(*, asynchronous: bool) -> None:
    """Route sync and async agent runs while preserving the complete answer."""
    middleware, models, classifier, _ = _router()
    agent = create_agent(models["powerful"], middleware=[middleware])
    latest_message = HumanMessage("Update the README")
    inputs: InputAgentState = {
        "messages": [
            HumanMessage("Earlier task"),
            AIMessage("Ready"),
            latest_message,
        ]
    }

    if asynchronous:
        result = await agent.ainvoke(inputs)
        classifier.ainvoke.assert_awaited_once_with(
            {
                "state": latest_message,
                "questions": _routing_questions(middleware.config),
            }
        )
    else:
        result = agent.invoke(inputs)
        classifier.invoke.assert_called_once_with(
            {
                "state": latest_message,
                "questions": _routing_questions(middleware.config),
            }
        )

    assert result["messages"][-1].text == "fast response"
    assert result["model_route"] == _response("fast").choices["model_route"]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_classifier_failure_terminates_agent_run(*, asynchronous: bool) -> None:
    """Propagate classifier failures through sync and async agent execution."""
    middleware, models, classifier, _ = _router()
    classifier.invoke.side_effect = RuntimeError("unavailable")
    classifier.ainvoke.side_effect = RuntimeError("unavailable")
    agent = create_agent(models["fast"], middleware=[middleware])
    inputs: InputAgentState = {"messages": [HumanMessage("Do the task")]}

    if asynchronous:
        with pytest.raises(RuntimeError, match="unavailable"):
            await agent.ainvoke(inputs)
    else:
        with pytest.raises(RuntimeError, match="unavailable"):
            agent.invoke(inputs)


@pytest.mark.parametrize("router_is_outer", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_model_fallback_preserves_its_replacement(
    *, router_is_outer: bool, asynchronous: bool
) -> None:
    """Use the fallback model regardless of router/fallback middleware order."""
    router, _, _, _ = _router()
    routed_model = _FailingModel()
    router.models["fast"] = routed_model
    fallback_model = GenericFakeChatModel(
        messages=iter([AIMessage("fallback response")])
    )
    fallback = ModelFallbackMiddleware(fallback_model)
    middleware: list[Any] = (
        [router, fallback] if router_is_outer else [fallback, router]
    )
    agent = create_agent(fallback_model, middleware=middleware)
    inputs: InputAgentState = {"messages": [HumanMessage("Do the task")]}

    result = await agent.ainvoke(inputs) if asynchronous else agent.invoke(inputs)

    assert result["messages"][-1].text == "fallback response"
    assert routed_model.attempts == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_outer_model_retry_reuses_the_routed_model(*, asynchronous: bool) -> None:
    """Retry the selected model when retry middleware reuses its request."""
    router, _, _, _ = _router()
    routed_model = _FailOnceModel(messages=iter([AIMessage("routed response")]))
    router.models["fast"] = routed_model
    retry = ModelRetryMiddleware(
        max_retries=1,
        initial_delay=0,
        jitter=False,
        on_failure="error",
    )
    agent = create_agent(
        GenericFakeChatModel(messages=iter([AIMessage("base response")])),
        middleware=[retry, cast("Any", router)],
    )
    inputs: InputAgentState = {"messages": [HumanMessage("Do the task")]}

    result = await agent.ainvoke(inputs) if asynchronous else agent.invoke(inputs)

    assert result["messages"][-1].text == "routed response"
    assert routed_model.attempts == 2


def test_active_node_records_are_not_evicted_and_are_cleaned_up() -> None:
    """Keep every live node record and discard records after requests are released."""
    router, models, _, _ = _router()
    originals: list[ModelRequest[Any]] = []
    for index in range(64):
        request = ModelRequest(
            model=models["powerful"],
            messages=[HumanMessage(f"Task {index}")],
            state=cast(
                "AgentState[Any]",
                {
                    "messages": [HumanMessage(f"Task {index}")],
                    "model_route": _response("fast").choices["model_route"],
                },
            ),
            runtime=Runtime(
                execution_info=ExecutionInfo(
                    checkpoint_id=f"checkpoint-{index}",
                    checkpoint_ns="agent",
                    task_id=f"task-{index}",
                )
            ),
        )
        originals.append(request)
        assert router._route_request(request).model is models["fast"]

    fallback_model = GenericFakeChatModel(messages=iter([]))
    fallback_request = originals[0].override(model=fallback_model)

    assert len(router._routed_requests) == 64
    assert router._route_request(fallback_request) is fallback_request

    originals.clear()
    del request
    gc.collect()

    assert router._routed_requests == {}


def test_new_node_attempt_routes_a_fresh_request() -> None:
    """Route a new request when LangGraph retries the same task-level node."""
    router, models, _, _ = _router()
    state = cast(
        "AgentState[Any]",
        {
            "messages": [HumanMessage("Do the task")],
            "model_route": _response("fast").choices["model_route"],
        },
    )

    def request_for_attempt(node_attempt: int) -> ModelRequest[Any]:
        return ModelRequest(
            model=models["powerful"],
            messages=[HumanMessage("Do the task")],
            state=state,
            runtime=Runtime(
                execution_info=ExecutionInfo(
                    checkpoint_id="checkpoint",
                    checkpoint_ns="agent",
                    task_id="task",
                    node_attempt=node_attempt,
                )
            ),
        )

    first_request = request_for_attempt(1)
    second_request = request_for_attempt(2)

    assert router._route_request(first_request).model is models["fast"]
    assert router._route_request(second_request).model is models["fast"]


def test_choices_are_required() -> None:
    """Reject an empty choice mapping through validated configuration fields."""
    with pytest.raises(ValidationError):
        ModelRouterMiddleware(
            choices={},
            instructions="Choose a route.",
        )


def test_model_string_is_initialized_once() -> None:
    """Resolve model strings through `init_chat_model` during construction."""
    initialized_model = GenericFakeChatModel(
        messages=iter([AIMessage("initialized response")])
    )
    classifier = MagicMock(spec=TypeSafeClassifier)
    with (
        patch(
            "langchain_typesafe.experimental.middleware.model_router.init_chat_model",
            return_value=initialized_model,
        ) as init_model,
        patch(
            "langchain_typesafe.experimental.middleware.model_router.TypeSafeClassifier",
            return_value=classifier,
        ),
    ):
        middleware = ModelRouterMiddleware(
            choices={
                "fast": ModelChoice(
                    model="openai:gpt-5-mini",
                    criteria="Simple tasks.",
                )
            },
            instructions="Choose a route.",
        )

    init_model.assert_called_once_with("openai:gpt-5-mini")
    assert middleware.models == {"fast": initialized_model}


def test_experimental_public_interface() -> None:
    """Expose the model router from the experimental middleware namespace."""
    assert middleware_all == [
        "AutoModeMiddleware",
        "ModelChoice",
        "ModelRouterMiddleware",
    ]

"""Tests for `ModelRouterMiddleware`."""

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from langchain_typesafe import Choice, ChoiceAnswer
from langchain_typesafe.classifier import TypeSafeClassifier
from langchain_typesafe.experimental.middleware import (
    ModelChoice,
    ModelRouterMiddleware,
)
from langchain_typesafe.experimental.middleware import (
    __all__ as middleware_all,
)
from langchain_typesafe.types import ClassificationResponse


def _response(route: str) -> ClassificationResponse:
    return ClassificationResponse(
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
    dict[str, BaseChatModel],
    MagicMock,
    MagicMock,
]:
    models = {
        "fast": cast("BaseChatModel", MagicMock(name="fast")),
        "powerful": cast("BaseChatModel", MagicMock(name="powerful")),
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


def _request(state: dict[str, Any]) -> ModelRequest[Any]:
    return ModelRequest(
        model=cast("BaseChatModel", MagicMock(name="original")),
        messages=state.get("messages", []),
        state=cast("Any", state),
    )


def test_middleware_constructs_classifier_from_routing_configuration() -> None:
    """Construct a TypeSafe Choice from the supplied criteria and instructions."""
    middleware, _, classifier, classifier_class = _router()

    classifier_class.assert_called_once()
    questions = classifier_class.call_args.kwargs["questions"]
    assert questions == {
        "model_route": Choice(
            instructions="Choose the least costly model suited to the task.",
            criteria={"fast": "Simple tasks.", "powerful": "Complex tasks."},
        )
    }
    assert middleware.classifier is classifier


def test_sync_route_is_stored_and_used_for_model_calls() -> None:
    """Classify the latest human message and route synchronous model calls."""
    middleware, models, classifier, _ = _router()
    old_message = HumanMessage("Earlier task")
    latest_message = HumanMessage("Update the README")
    state: dict[str, Any] = {
        "messages": [old_message, AIMessage("Ready"), latest_message]
    }

    state.update(middleware.before_agent(cast("Any", state), MagicMock()))
    request = _request(state)
    seen: list[ModelRequest[Any]] = []

    def handler(routed: ModelRequest[Any]) -> ModelResponse[Any]:
        seen.append(routed)
        return MagicMock()

    middleware.wrap_model_call(request, handler)

    assert state["model_route"] == "fast"
    assert seen[0].model is models["fast"]
    assert request.model is not models["fast"]
    classifier.invoke.assert_called_once()
    assert classifier.invoke.call_args.args[0] is latest_message
    assert classifier.invoke.call_args.kwargs == {}


@pytest.mark.asyncio
async def test_async_route_is_stored_and_used_for_model_calls() -> None:
    """Classify the latest human message and route asynchronous model calls."""
    middleware, models, classifier, _ = _router()
    latest_message = HumanMessage("Investigate a race condition")
    state: dict[str, Any] = {"messages": [latest_message]}

    state.update(await middleware.abefore_agent(cast("Any", state), MagicMock()))
    request = _request(state)
    seen: list[ModelRequest[Any]] = []

    async def handler(routed: ModelRequest[Any]) -> ModelResponse[Any]:
        seen.append(routed)
        return MagicMock()

    await middleware.awrap_model_call(request, handler)

    assert state["model_route"] == "fast"
    assert seen[0].model is models["fast"]
    classifier.ainvoke.assert_awaited_once()
    assert classifier.ainvoke.call_args.args[0] is latest_message


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_classifier_failure_terminates_run(*, asynchronous: bool) -> None:
    """Propagate classifier failures instead of silently selecting a model."""
    middleware, _, classifier, _ = _router()
    classifier.invoke.side_effect = RuntimeError("unavailable")
    classifier.ainvoke.side_effect = RuntimeError("unavailable")
    state = cast("Any", {"messages": [HumanMessage("Do the task")]})

    if asynchronous:
        with pytest.raises(RuntimeError, match="unavailable"):
            await middleware.abefore_agent(state, MagicMock())
    else:
        with pytest.raises(RuntimeError, match="unavailable"):
            middleware.before_agent(state, MagicMock())


def test_missing_human_message_terminates_run() -> None:
    """Reject agent state without a human task to classify."""
    middleware, _, classifier, _ = _router()
    state = cast("Any", {"messages": [AIMessage("No task yet")]})

    with pytest.raises(ValueError, match="at least one human message"):
        middleware.before_agent(state, MagicMock())

    classifier.invoke.assert_not_called()


def test_unknown_choice_terminates_run() -> None:
    """Reject a TypeSafe choice that has no configured model."""
    middleware, _, classifier, _ = _router()
    classifier.invoke.return_value = _response("unknown")

    with pytest.raises(ValueError, match="unknown model route 'unknown'"):
        middleware.before_agent(
            cast("Any", {"messages": [HumanMessage("Do the task")]}),
            MagicMock(),
        )


def test_unknown_route_in_state_terminates_run() -> None:
    """Reject model calls whose state has no configured route."""
    middleware, _, _, _ = _router()
    request = _request({"messages": [], "model_route": "unknown"})

    with pytest.raises(ValueError, match="unknown model route 'unknown'"):
        middleware.wrap_model_call(request, MagicMock())


def test_configuration_validation() -> None:
    """Reject an empty choice mapping."""
    with pytest.raises(ValueError, match="At least one model choice"):
        ModelRouterMiddleware(
            choices={},
            instructions="Choose a route.",
        )


def test_model_string_is_initialized_once() -> None:
    """Resolve model strings through `init_chat_model` during construction."""
    initialized_model = cast("BaseChatModel", MagicMock())
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


def test_create_agent_routes_the_run_to_the_selected_model() -> None:
    """Compose routing state and model overrides through a compiled agent graph."""
    fast_model = GenericFakeChatModel(messages=iter([AIMessage("fast response")]))
    powerful_model = GenericFakeChatModel(
        messages=iter([AIMessage("powerful response")])
    )
    classifier = MagicMock(spec=TypeSafeClassifier)
    classifier.invoke.return_value = _response("fast")
    with patch(
        "langchain_typesafe.experimental.middleware.model_router.TypeSafeClassifier",
        return_value=classifier,
    ):
        middleware = ModelRouterMiddleware(
            choices={
                "fast": ModelChoice(model=fast_model, criteria="Simple."),
                "powerful": ModelChoice(model=powerful_model, criteria="Complex."),
            },
            instructions="Choose the least costly suitable route.",
        )
    agent = create_agent(powerful_model, middleware=[middleware])

    result = agent.invoke({"messages": [HumanMessage("Update one line")]})

    assert result["messages"][-1].text == "fast response"
    classifier.invoke.assert_called_once()


def test_experimental_public_interface() -> None:
    """Expose the model router from the experimental middleware namespace."""
    assert middleware_all == ["ModelChoice", "ModelRouterMiddleware"]

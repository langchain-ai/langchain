"""Tests for `ModelRouterMiddleware`."""

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.agents import create_agent
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


def test_sync_agent_routes_using_latest_human_message() -> None:
    """Route a synchronous agent run using the latest human task."""
    middleware, models, classifier, _ = _router()
    agent = create_agent(models["powerful"], middleware=[middleware])
    latest_message = HumanMessage("Update the README")

    result = agent.invoke(
        {
            "messages": [
                HumanMessage("Earlier task"),
                AIMessage("Ready"),
                latest_message,
            ]
        }
    )

    assert result["messages"][-1].text == "fast response"
    classifier.invoke.assert_called_once_with(latest_message)


@pytest.mark.asyncio
async def test_async_agent_routes_using_latest_human_message() -> None:
    """Route an asynchronous agent run using the latest human task."""
    middleware, models, classifier, _ = _router()
    agent = create_agent(models["powerful"], middleware=[middleware])
    latest_message = HumanMessage("Investigate a race condition")

    result = await agent.ainvoke({"messages": [latest_message]})

    assert result["messages"][-1].text == "fast response"
    classifier.ainvoke.assert_awaited_once_with(latest_message)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_classifier_failure_terminates_agent_run(*, asynchronous: bool) -> None:
    """Propagate classifier failures through sync and async agent execution."""
    middleware, models, classifier, _ = _router()
    classifier.invoke.side_effect = RuntimeError("unavailable")
    classifier.ainvoke.side_effect = RuntimeError("unavailable")
    agent = create_agent(models["fast"], middleware=[middleware])
    inputs = cast("Any", {"messages": [HumanMessage("Do the task")]})

    if asynchronous:
        with pytest.raises(RuntimeError, match="unavailable"):
            await agent.ainvoke(inputs)
    else:
        with pytest.raises(RuntimeError, match="unavailable"):
            agent.invoke(inputs)


def test_missing_human_message_terminates_agent_run() -> None:
    """Reject an agent run without a human task to classify."""
    middleware, models, classifier, _ = _router()
    agent = create_agent(models["fast"], middleware=[middleware])

    with pytest.raises(ValueError, match="at least one human message"):
        agent.invoke({"messages": [AIMessage("No task yet")]})

    classifier.invoke.assert_not_called()


def test_unknown_choice_terminates_agent_run() -> None:
    """Reject a TypeSafe choice that has no configured model."""
    middleware, models, classifier, _ = _router()
    classifier.invoke.return_value = _response("unknown")
    agent = create_agent(models["fast"], middleware=[middleware])

    with pytest.raises(ValueError, match="unknown model route 'unknown'"):
        agent.invoke({"messages": [HumanMessage("Do the task")]})


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


def test_experimental_public_interface() -> None:
    """Expose the model router from the experimental middleware namespace."""
    assert middleware_all == ["ModelChoice", "ModelRouterMiddleware"]

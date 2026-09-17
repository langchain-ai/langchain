"""Tests for `ModelRouterMiddleware`."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import InputAgentState
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

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
    """Construct a TypeSafe Choice and expose validated configuration fields."""
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
        classifier.ainvoke.assert_awaited_once_with(latest_message)
    else:
        result = agent.invoke(inputs)
        classifier.invoke.assert_called_once_with(latest_message)

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

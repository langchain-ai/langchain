"""Tests for `ModelRouterMiddleware`."""

import asyncio
import inspect
import runpy
import warnings
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core._api import LangChainBetaWarning
from langchain_core.decisions import (
    BaseDecisionModel,
    Choice,
    ChoiceAnswer,
    ClassifierResponse,
    Question,
    State,
)
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import Field, ValidationError
from typing_extensions import override

from langchain.agents import create_agent
from langchain.agents.middleware import ModelChoice, ModelRouterMiddleware
from langchain.agents.middleware.model_router import _routing_questions
from langchain.agents.middleware.types import InputAgentState


def _answer(route: str) -> ChoiceAnswer:
    return ChoiceAnswer(type="choice", choice=route, probabilities={route: 1.0}, confidence=1.0)


class _RouteClassifier(BaseDecisionModel):
    """Decision model that records requests and always selects one route."""

    model: str = "fake-model"
    route: str = "fast"
    error: str | None = None
    requests: list[Any] = Field(default_factory=list)

    @property
    @override
    def _provider(self) -> str:
        return "fake"

    @override
    def _classify(self, state: State, questions: dict[str, Question]) -> ClassifierResponse:
        self.requests.append((state, questions))
        if self.error is not None:
            raise RuntimeError(self.error)
        return ClassifierResponse(model=self.model, answers={"model_route": _answer(self.route)})


def _router(
    classifier: _RouteClassifier,
) -> tuple[ModelRouterMiddleware, dict[str, GenericFakeChatModel]]:
    models = {
        "fast": GenericFakeChatModel(messages=iter([AIMessage("fast response")])),
        "powerful": GenericFakeChatModel(messages=iter([AIMessage("powerful response")])),
    }
    middleware = ModelRouterMiddleware(
        choices={
            "fast": ModelChoice(model=models["fast"], criteria="Simple tasks."),
            "powerful": ModelChoice(model=models["powerful"], criteria="Complex tasks."),
        },
        instructions="Choose the least costly model suited to the task.",
        classifier=classifier,
    )
    return middleware, models


async def _run_agent(
    middleware: ModelRouterMiddleware,
    model: GenericFakeChatModel,
    inputs: InputAgentState,
    *,
    async_: bool,
) -> dict[str, Any]:
    agent = create_agent(model, middleware=[middleware])
    if async_:
        return await agent.ainvoke(inputs)
    return await asyncio.to_thread(agent.invoke, inputs)


def test_beta_warning_is_emitted_once() -> None:
    namespace = runpy.run_path(inspect.getfile(ModelRouterMiddleware))
    router_class = namespace["ModelRouterMiddleware"]
    model_choice = namespace["ModelChoice"]
    model = GenericFakeChatModel(messages=iter([AIMessage("response")]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", LangChainBetaWarning)
        for _ in range(2):
            router_class(
                choices={"fast": model_choice(model=model, criteria="Simple tasks.")},
                instructions="Choose a route.",
                classifier=_RouteClassifier(),
            )

    beta_warnings = [warning for warning in caught if warning.category is LangChainBetaWarning]
    assert len(beta_warnings) == 1
    assert str(beta_warnings[0].message) == (
        "The middleware `ModelRouterMiddleware` is in beta. "
        "It is actively being worked on, so the API may change."
    )


def test_routing_question_is_built_from_choices() -> None:
    classifier = _RouteClassifier()
    middleware, _ = _router(classifier)

    assert middleware.classifier is classifier
    assert _routing_questions(middleware.config) == {
        "model_route": Choice(
            instructions="Choose the least costly model suited to the task.",
            criteria={"fast": "Simple tasks.", "powerful": "Complex tasks."},
        )
    }


@pytest.mark.parametrize("async_", [False, True])
async def test_agent_routes_using_latest_human_message(*, async_: bool) -> None:
    classifier = _RouteClassifier()
    middleware, models = _router(classifier)
    latest_message = HumanMessage("Update the README")
    inputs: InputAgentState = {
        "messages": [HumanMessage("Earlier task"), AIMessage("Ready"), latest_message]
    }

    result = await _run_agent(middleware, models["powerful"], inputs, async_=async_)

    [(state, questions)] = classifier.requests
    assert state == latest_message
    assert questions == _routing_questions(middleware.config)
    assert result["messages"][-1].text == "fast response"
    assert result["model_route"] == _answer("fast")


@pytest.mark.parametrize("async_", [False, True])
async def test_classifier_failure_terminates_agent_run(*, async_: bool) -> None:
    middleware, models = _router(_RouteClassifier(error="unavailable"))
    inputs: InputAgentState = {"messages": [HumanMessage("Do the task")]}

    with pytest.raises(RuntimeError, match="unavailable"):
        await _run_agent(middleware, models["fast"], inputs, async_=async_)


@pytest.mark.parametrize("async_", [False, True])
async def test_missing_human_message_is_rejected(*, async_: bool) -> None:
    classifier = _RouteClassifier()
    middleware, models = _router(classifier)
    inputs: InputAgentState = {"messages": [AIMessage("Ready")]}

    with pytest.raises(ValueError, match="requires a HumanMessage"):
        await _run_agent(middleware, models["fast"], inputs, async_=async_)

    assert classifier.requests == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"choices": {}, "classifier": _RouteClassifier()},
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ModelRouterMiddleware(instructions="Choose a route.", **kwargs)


def test_model_string_is_initialized_once() -> None:
    initialized_model = GenericFakeChatModel(messages=iter([AIMessage("initialized")]))
    with patch(
        "langchain.agents.middleware.model_router.init_chat_model",
        return_value=initialized_model,
    ) as init_model:
        middleware = ModelRouterMiddleware(
            choices={"fast": ModelChoice(model="openai:gpt-5-mini", criteria="Simple tasks.")},
            instructions="Choose a route.",
            classifier=_RouteClassifier(),
        )

    init_model.assert_called_once_with("openai:gpt-5-mini")
    assert middleware.models == {"fast": initialized_model}

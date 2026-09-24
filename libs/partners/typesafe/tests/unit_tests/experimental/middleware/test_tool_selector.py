"""Tests for `TsToolSelectorMiddleware`."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from langchain_typesafe import ChoiceAnswer, NoulAnswer, TypeSafeClassifier
from langchain_typesafe.experimental.middleware import (
    TsChoiceToolSelectorMiddleware,
    TsToolSelectorMiddleware,
)
from langchain_typesafe.experimental.middleware import __all__ as middleware_all
from langchain_typesafe.types import ClassificationResponse


@tool
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    return f"Weather in {location}: 72F, sunny"


@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"


@tool
def send_email(to: str, subject: str) -> str:
    """Send an email to someone."""
    return f"Email with subject {subject} sent to {to}"


def _response(scores: dict[str, float]) -> ClassificationResponse:
    return ClassificationResponse(
        model="jev-latest",
        answers={
            f"tool::{name}": NoulAnswer(type="noul", noul=score)
            for name, score in scores.items()
        },
    )


def _choice_response(choice: str) -> ClassificationResponse:
    return ClassificationResponse(
        model="jev-latest",
        answers={
            "tool": ChoiceAnswer(
                type="choice",
                choice=choice,
                probabilities={choice: 1.0},
                confidence=1.0,
            )
        },
    )


def _classifier(response: ClassificationResponse) -> MagicMock:
    classifier = MagicMock(spec=TypeSafeClassifier)
    classifier.invoke.return_value = response
    classifier.ainvoke = AsyncMock(return_value=response)
    return classifier


def _tool_names(tools: list[Any]) -> list[str]:
    return [tool.name for tool in tools if not isinstance(tool, dict)]


def _request(tools: list[Any], messages: list[Any]) -> ModelRequest[Any]:
    return ModelRequest(
        model=cast("BaseChatModel", MagicMock()),
        messages=messages,
        tools=tools,
        state=cast("Any", {"messages": messages}),
    )


def test_relevance_threshold_is_validated() -> None:
    """Reject an out-of-range threshold at construction time."""
    with pytest.raises(ValueError, match="relevance_threshold"):
        TsToolSelectorMiddleware(relevance_threshold=1.1)
    with pytest.raises(ValueError, match="relevance_threshold"):
        TsToolSelectorMiddleware(relevance_threshold=-0.1)


def test_middleware_constructs_classifier_per_call() -> None:
    """Build one independent `Noul` question per candidate tool, per call."""
    classifier = _classifier(_response({"get_weather": 0.9, "search_web": 0.1}))
    request = _request([get_weather, search_web], [HumanMessage("What's the weather?")])
    middleware = TsToolSelectorMiddleware()

    with patch(
        "langchain_typesafe.experimental.middleware.tool_selector.TypeSafeClassifier",
        return_value=classifier,
    ) as classifier_class:
        middleware.wrap_model_call(request, lambda _req: MagicMock())

    classifier_class.assert_called_once()
    questions = classifier_class.call_args.kwargs["questions"]
    assert set(questions) == {"tool::get_weather", "tool::search_web"}
    assert "get_weather" in questions["tool::get_weather"].instructions
    assert "current weather" in questions["tool::get_weather"].instructions


@pytest.mark.parametrize("model", ["jev-latest", "semif-qwen3.5-4b"])
@pytest.mark.parametrize(
    "selector", [TsToolSelectorMiddleware, TsChoiceToolSelectorMiddleware]
)
def test_classifier_uses_requested_model(model: str, selector: type) -> None:
    """Route both tool selection modes to the configured classifier model."""
    response = (
        _choice_response("get_weather")
        if selector is TsChoiceToolSelectorMiddleware
        else _response({"get_weather": 0.9})
    )
    classifier = _classifier(response)
    request = _request([get_weather], [HumanMessage("What's the weather?")])
    middleware = selector(classifier_model=model)

    with patch(
        "langchain_typesafe.experimental.middleware.tool_selector.TypeSafeClassifier",
        return_value=classifier,
    ) as classifier_class:
        middleware.wrap_model_call(request, lambda _req: MagicMock())

    assert classifier_class.call_args.kwargs["model"] == model


def test_sync_selection_filters_tools_above_threshold() -> None:
    """Keep only tools whose `Noul` probability clears the threshold."""
    classifier = _classifier(
        _response({"get_weather": 0.9, "search_web": 0.1, "send_email": 0.05})
    )
    request = _request(
        [get_weather, search_web, send_email],
        [HumanMessage("What's the weather in Boston?")],
    )
    middleware = TsToolSelectorMiddleware()
    seen: list[ModelRequest[Any]] = []

    def handler(modified: ModelRequest[Any]) -> ModelResponse[Any]:
        seen.append(modified)
        return cast("ModelResponse[Any]", MagicMock())

    with patch(
        "langchain_typesafe.experimental.middleware.tool_selector.TypeSafeClassifier",
        return_value=classifier,
    ):
        middleware.wrap_model_call(request, handler)

    classifier.invoke.assert_called_once()
    assert classifier.invoke.call_args.args[0] is request.messages[0]
    assert classifier.invoke.call_args.kwargs["config"]["metadata"] == {
        "lc_source": "ts_tool_selector"
    }
    assert _tool_names(seen[0].tools) == ["get_weather"]


def test_max_tools_orders_by_probability_then_truncates() -> None:
    """Rank kept tools by probability and cap at `max_tools`."""
    classifier = _classifier(
        _response({"get_weather": 0.6, "search_web": 0.9, "send_email": 0.4})
    )
    request = _request(
        [get_weather, search_web, send_email], [HumanMessage("Help me with tasks")]
    )
    middleware = TsToolSelectorMiddleware(max_tools=2)
    seen: list[ModelRequest[Any]] = []

    def handler(modified: ModelRequest[Any]) -> ModelResponse[Any]:
        seen.append(modified)
        return cast("ModelResponse[Any]", MagicMock())

    with patch(
        "langchain_typesafe.experimental.middleware.tool_selector.TypeSafeClassifier",
        return_value=classifier,
    ):
        middleware.wrap_model_call(request, handler)

    assert _tool_names(seen[0].tools) == ["search_web", "get_weather"]


def test_always_include_bypasses_classification_and_max_tools() -> None:
    """Always-included tools skip TypeSafe entirely and don't count toward max_tools."""
    classifier = _classifier(_response({"search_web": 0.9}))
    request = _request([get_weather, search_web], [HumanMessage("What's the weather?")])
    middleware = TsToolSelectorMiddleware(max_tools=1, always_include=["get_weather"])
    seen: list[ModelRequest[Any]] = []

    def handler(modified: ModelRequest[Any]) -> ModelResponse[Any]:
        seen.append(modified)
        return cast("ModelResponse[Any]", MagicMock())

    with patch(
        "langchain_typesafe.experimental.middleware.tool_selector.TypeSafeClassifier",
        return_value=classifier,
    ) as classifier_class:
        middleware.wrap_model_call(request, handler)

    questions = classifier_class.call_args.kwargs["questions"]
    assert set(questions) == {"tool::search_web"}
    assert set(_tool_names(seen[0].tools)) == {"get_weather", "search_web"}


def test_always_include_missing_tool_raises() -> None:
    """Fail fast when an always-included tool isn't bound to the agent."""
    request = _request([get_weather], [HumanMessage("Hi")])
    middleware = TsToolSelectorMiddleware(always_include=["send_email"])

    with pytest.raises(ValueError, match="not found in request"):
        middleware.wrap_model_call(request, lambda _req: MagicMock())


def test_provider_tool_dicts_pass_through_untouched() -> None:
    """Preserve provider-specific tool dicts that aren't `BaseTool` instances."""
    classifier = _classifier(_response({"get_weather": 0.9}))
    provider_tool = {"type": "web_search"}
    request = _request(
        [get_weather, provider_tool], [HumanMessage("What's the weather?")]
    )
    middleware = TsToolSelectorMiddleware()
    seen: list[ModelRequest[Any]] = []

    def handler(modified: ModelRequest[Any]) -> ModelResponse[Any]:
        seen.append(modified)
        return cast("ModelResponse[Any]", MagicMock())

    with patch(
        "langchain_typesafe.experimental.middleware.tool_selector.TypeSafeClassifier",
        return_value=classifier,
    ):
        middleware.wrap_model_call(request, handler)

    assert provider_tool in seen[0].tools


def test_no_tools_is_noop() -> None:
    """Skip classification entirely when the request has no tools."""
    request = _request([], [HumanMessage("Hi")])
    middleware = TsToolSelectorMiddleware()

    result_request: list[ModelRequest[Any]] = []

    def handler(modified: ModelRequest[Any]) -> ModelResponse[Any]:
        result_request.append(modified)
        return cast("ModelResponse[Any]", MagicMock())

    with patch(
        "langchain_typesafe.experimental.middleware.tool_selector.TypeSafeClassifier"
    ) as classifier_class:
        middleware.wrap_model_call(request, handler)

    classifier_class.assert_not_called()
    assert result_request[0] is request


def test_relevance_threshold_is_inclusive() -> None:
    """Keep a tool exactly at the threshold boundary."""
    classifier = _classifier(_response({"get_weather": 0.3, "search_web": 0.29}))
    request = _request([get_weather, search_web], [HumanMessage("Help")])
    middleware = TsToolSelectorMiddleware(relevance_threshold=0.3)
    seen: list[ModelRequest[Any]] = []

    def handler(modified: ModelRequest[Any]) -> ModelResponse[Any]:
        seen.append(modified)
        return cast("ModelResponse[Any]", MagicMock())

    with patch(
        "langchain_typesafe.experimental.middleware.tool_selector.TypeSafeClassifier",
        return_value=classifier,
    ):
        middleware.wrap_model_call(request, handler)

    assert _tool_names(seen[0].tools) == ["get_weather"]


def test_missing_human_message_raises() -> None:
    """Fail fast rather than silently classify against no request."""
    request = _request([get_weather], [AIMessage("No request yet")])
    middleware = TsToolSelectorMiddleware()

    with pytest.raises(AssertionError, match="No user message found"):
        middleware.wrap_model_call(request, lambda _req: MagicMock())


def test_sync_classifier_failure_propagates() -> None:
    """Surface TypeSafe errors rather than silently running unfiltered or empty."""
    classifier = MagicMock(spec=TypeSafeClassifier)
    classifier.invoke.side_effect = RuntimeError("unavailable")
    request = _request([get_weather], [HumanMessage("What's the weather?")])
    middleware = TsToolSelectorMiddleware()

    with (
        patch(
            "langchain_typesafe.experimental.middleware.tool_selector.TypeSafeClassifier",
            return_value=classifier,
        ),
        pytest.raises(RuntimeError, match="unavailable"),
    ):
        middleware.wrap_model_call(request, lambda _req: MagicMock())


@pytest.mark.asyncio
async def test_async_classifier_failure_propagates() -> None:
    """Surface TypeSafe errors rather than silently running unfiltered or empty."""
    classifier = MagicMock(spec=TypeSafeClassifier)
    classifier.ainvoke = AsyncMock(side_effect=RuntimeError("unavailable"))
    request = _request([get_weather], [HumanMessage("What's the weather?")])
    middleware = TsToolSelectorMiddleware()

    with (
        patch(
            "langchain_typesafe.experimental.middleware.tool_selector.TypeSafeClassifier",
            return_value=classifier,
        ),
        pytest.raises(RuntimeError, match="unavailable"),
    ):
        await middleware.awrap_model_call(request, AsyncMock())


@pytest.mark.asyncio
async def test_async_selection_filters_tools() -> None:
    """Use asynchronous classification and request handling."""
    classifier = _classifier(_response({"get_weather": 0.9, "search_web": 0.1}))
    request = _request([get_weather, search_web], [HumanMessage("What's the weather?")])
    middleware = TsToolSelectorMiddleware()
    seen: list[ModelRequest[Any]] = []

    async def handler(modified: ModelRequest[Any]) -> ModelResponse[Any]:
        seen.append(modified)
        return cast("ModelResponse[Any]", MagicMock())

    with patch(
        "langchain_typesafe.experimental.middleware.tool_selector.TypeSafeClassifier",
        return_value=classifier,
    ):
        await middleware.awrap_model_call(request, handler)

    classifier.ainvoke.assert_awaited_once()
    assert _tool_names(seen[0].tools) == ["get_weather"]


def test_choice_selects_one_tool_per_model_call() -> None:
    """Choose one candidate afresh for every model call, not the entire task."""
    classifier = _classifier(_choice_response("get_weather"))
    middleware = TsChoiceToolSelectorMiddleware()
    first = _request([get_weather, search_web], [HumanMessage("Find the weather")])
    second = _request(
        [get_weather, search_web],
        [HumanMessage("Find the weather"), AIMessage("Next search the web")],
    )
    seen: list[ModelRequest[Any]] = []

    def handler(modified: ModelRequest[Any]) -> ModelResponse[Any]:
        seen.append(modified)
        return cast("ModelResponse[Any]", MagicMock())

    with patch(
        "langchain_typesafe.experimental.middleware.tool_selector.TypeSafeClassifier",
        return_value=classifier,
    ) as classifier_class:
        middleware.wrap_model_call(first, handler)
        classifier.invoke.return_value = _choice_response("search_web")
        middleware.wrap_model_call(second, handler)

    assert classifier_class.call_count == 2
    question = classifier_class.call_args.kwargs["questions"]["tool"]
    assert question.criteria == {
        "get_weather": get_weather.description,
        "search_web": search_web.description,
    }
    assert "next" in question.instructions
    assert classifier.invoke.call_args.kwargs["config"]["metadata"] == {
        "lc_source": "ts_choice_tool_selector"
    }
    assert _tool_names(seen[0].tools) == ["get_weather"]
    assert _tool_names(seen[1].tools) == ["search_web"]
    assert first.tools == [get_weather, search_web]


@pytest.mark.asyncio
async def test_choice_async_preserves_always_included_and_provider_tools() -> None:
    """Keep bypassed tools while asynchronously choosing one candidate."""
    classifier = _classifier(_choice_response("search_web"))
    provider_tool = {"type": "web_search"}
    request = _request(
        [get_weather, search_web, send_email, provider_tool],
        [HumanMessage("Search for the latest weather")],
    )
    middleware = TsChoiceToolSelectorMiddleware(always_include=["get_weather"])
    seen: list[ModelRequest[Any]] = []

    async def handler(modified: ModelRequest[Any]) -> ModelResponse[Any]:
        seen.append(modified)
        return cast("ModelResponse[Any]", MagicMock())

    with patch(
        "langchain_typesafe.experimental.middleware.tool_selector.TypeSafeClassifier",
        return_value=classifier,
    ) as classifier_class:
        await middleware.awrap_model_call(request, handler)

    assert classifier.ainvoke.await_count == 1
    assert classifier_class.call_args.kwargs["questions"]["tool"].criteria == {
        "search_web": search_web.description,
        "send_email": send_email.description,
    }
    assert _tool_names(seen[0].tools) == ["search_web", "get_weather"]
    assert provider_tool in seen[0].tools


@pytest.mark.parametrize("response", [_choice_response("unknown"), _response({})])
def test_choice_rejects_invalid_response(response: ClassificationResponse) -> None:
    """Never substitute an unavailable or missing tool silently."""
    classifier = _classifier(response)
    request = _request([get_weather], [HumanMessage("Get the weather")])

    with (
        patch(
            "langchain_typesafe.experimental.middleware.tool_selector.TypeSafeClassifier",
            return_value=classifier,
        ),
        pytest.raises(ValueError, match="no valid tool choice"),
    ):
        TsChoiceToolSelectorMiddleware().wrap_model_call(
            request, lambda _req: MagicMock()
        )


def test_choice_only_always_included_tools_skips_classifier() -> None:
    """Skip classification if no candidate tools remain."""
    request = _request([get_weather], [HumanMessage("Get the weather")])
    with patch(
        "langchain_typesafe.experimental.middleware.tool_selector.TypeSafeClassifier"
    ) as classifier_class:
        TsChoiceToolSelectorMiddleware(always_include=["get_weather"]).wrap_model_call(
            request, lambda _req: MagicMock()
        )
    classifier_class.assert_not_called()


def test_experimental_public_interface() -> None:
    """Expose the tool selector from the experimental middleware namespace."""
    assert middleware_all == [
        "Skill",
        "SkillSource",
        "SkillsMiddleware",
        "TsChoiceToolSelectorMiddleware",
        "TsToolSelectorMiddleware",
    ]

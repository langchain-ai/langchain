"""Tests for `TsToolSelectorMiddleware`."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from langchain_typesafe import NoulAnswer, TypeSafeClassifier
from langchain_typesafe.experimental.middleware import TsToolSelectorMiddleware
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


def _classifier(response: ClassificationResponse) -> MagicMock:
    classifier = MagicMock(spec=TypeSafeClassifier)
    classifier.invoke.return_value = response
    classifier.ainvoke = AsyncMock(return_value=response)
    return classifier


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
    assert [t.name for t in seen[0].tools] == ["get_weather"]


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

    assert [t.name for t in seen[0].tools] == ["search_web", "get_weather"]


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
    assert {t.name for t in seen[0].tools} == {"get_weather", "search_web"}


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

    with patch(
        "langchain_typesafe.experimental.middleware.tool_selector.TypeSafeClassifier"
    ) as classifier_class:
        result_request: list[ModelRequest[Any]] = []
        middleware.wrap_model_call(
            request, lambda req: result_request.append(req) or MagicMock()
        )

    classifier_class.assert_not_called()
    assert result_request[0] is request


def test_relevance_threshold_is_inclusive() -> None:
    """Keep a tool exactly at the threshold boundary."""
    classifier = _classifier(_response({"get_weather": 0.3, "search_web": 0.29}))
    request = _request([get_weather, search_web], [HumanMessage("Help")])
    middleware = TsToolSelectorMiddleware(relevance_threshold=0.3)
    seen: list[ModelRequest[Any]] = []

    with patch(
        "langchain_typesafe.experimental.middleware.tool_selector.TypeSafeClassifier",
        return_value=classifier,
    ):
        middleware.wrap_model_call(request, lambda req: seen.append(req) or MagicMock())

    assert [t.name for t in seen[0].tools] == ["get_weather"]


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
    assert [t.name for t in seen[0].tools] == ["get_weather"]


def test_experimental_public_interface() -> None:
    """Expose the tool selector from the experimental middleware namespace."""
    assert middleware_all == [
        "Skill",
        "SkillSource",
        "SkillsMiddleware",
        "TsToolSelectorMiddleware",
    ]

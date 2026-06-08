"""Unit tests for provider tool search middleware."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool, tool

from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    ProviderToolSearchMiddleware,
)

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}"


@tool
def send_email(to: str) -> str:
    """Send an email."""
    return f"Sent to {to}"


@tool(extras={"defer_loading": True})
def lookup_order(order_id: str) -> str:
    """Look up an order."""
    return f"Order {order_id} shipped"


class FakeModel:
    def __init__(self, provider: str) -> None:
        self.provider = provider

    def _get_ls_params(self) -> dict[str, str]:
        return {"ls_provider": self.provider}


class FakeConfigurableModel:
    def __init__(self, default_config: dict[str, Any]) -> None:
        self._default_config = default_config


def _request(provider: str, tools: list[BaseTool | dict[str, Any]]) -> ModelRequest:
    return ModelRequest(
        model=cast("BaseChatModel", FakeModel(provider)),
        messages=[HumanMessage("hi")],
        tools=tools,
    )


def _invoke(middleware: ProviderToolSearchMiddleware, request: ModelRequest) -> ModelRequest:
    captured_request = None

    def handler(model_request: ModelRequest) -> ModelResponse:
        nonlocal captured_request
        captured_request = model_request
        return ModelResponse(result=[AIMessage("ok")])

    middleware.wrap_model_call(request, handler)
    assert captured_request is not None
    return captured_request


def test_passes_through_when_no_tools_are_deferred() -> None:
    request = _request("anthropic", [get_weather, send_email])
    middleware = ProviderToolSearchMiddleware()

    modified_request = _invoke(middleware, request)

    assert modified_request is request


def test_defers_tools_named_in_searchable_tools() -> None:
    request = _request("anthropic", [get_weather, send_email])
    middleware = ProviderToolSearchMiddleware(searchable_tools=["send_email"])

    modified_request = _invoke(middleware, request)

    email_tool = next(
        tool
        for tool in modified_request.tools
        if isinstance(tool, BaseTool) and tool.name == "send_email"
    )
    weather_tool = next(
        tool
        for tool in modified_request.tools
        if isinstance(tool, BaseTool) and tool.name == "get_weather"
    )
    assert email_tool.extras == {"defer_loading": True}
    assert weather_tool.extras is None
    assert {
        "type": "tool_search_tool_bm25_20251119",
        "name": "tool_search_tool_bm25",
    } in modified_request.tools


def test_accepts_tool_instances_in_searchable_tools() -> None:
    request = _request("openai", [get_weather, send_email])
    middleware = ProviderToolSearchMiddleware(searchable_tools=[send_email])

    modified_request = _invoke(middleware, request)

    email_tool = next(
        tool
        for tool in modified_request.tools
        if isinstance(tool, BaseTool) and tool.name == "send_email"
    )
    assert email_tool.extras == {"defer_loading": True}
    assert {"type": "tool_search"} in modified_request.tools


def test_honors_tools_pre_marked_with_defer_loading() -> None:
    request = _request("anthropic", [get_weather, lookup_order])
    middleware = ProviderToolSearchMiddleware()

    modified_request = _invoke(middleware, request)

    order_tool = next(
        tool
        for tool in modified_request.tools
        if isinstance(tool, BaseTool) and tool.name == "lookup_order"
    )
    assert order_tool.extras == {"defer_loading": True}
    assert {
        "type": "tool_search_tool_bm25_20251119",
        "name": "tool_search_tool_bm25",
    } in modified_request.tools


def test_raises_when_searchable_tool_is_not_bound() -> None:
    request = _request("anthropic", [get_weather])
    middleware = ProviderToolSearchMiddleware(searchable_tools=["missing_tool"])

    with pytest.raises(ValueError, match="missing_tool"):
        _invoke(middleware, request)


def test_raises_for_unsupported_provider_even_without_deferred_tools() -> None:
    request = _request("mistralai", [get_weather])
    middleware = ProviderToolSearchMiddleware()

    with pytest.raises(ValueError, match="server-side tool search"):
        _invoke(middleware, request)


def test_detects_provider_from_configurable_model() -> None:
    request = ModelRequest(
        model=cast("BaseChatModel", FakeConfigurableModel({"model": "openai:gpt-5.4"})),
        messages=[HumanMessage("hi")],
        tools=[send_email],
    )
    middleware = ProviderToolSearchMiddleware(searchable_tools=["send_email"])

    modified_request = _invoke(middleware, request)

    assert {"type": "tool_search"} in modified_request.tools


async def test_async_wrap_model_call_defers_tools() -> None:
    request = _request("openai", [send_email])
    middleware = ProviderToolSearchMiddleware(searchable_tools=["send_email"])
    captured_request = None

    async def handler(model_request: ModelRequest) -> ModelResponse:
        nonlocal captured_request
        captured_request = model_request
        return ModelResponse(result=[AIMessage("ok")])

    await middleware.awrap_model_call(request, handler)

    assert captured_request is not None
    assert {"type": "tool_search"} in captured_request.tools

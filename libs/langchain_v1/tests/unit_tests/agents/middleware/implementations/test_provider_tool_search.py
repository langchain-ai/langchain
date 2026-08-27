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

ANTHROPIC_SEARCH_TOOL = {
    "type": "tool_search_tool_bm25_20251119",
    "name": "tool_search_tool_bm25",
}
OPENAI_SEARCH_TOOL = {"type": "tool_search"}
OPENAI_TEST_MODEL = "gpt-5.5"
OPENAI_REASONING_TEST_MODEL = "o3"


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


@tool(extras={"category": "billing"})
def refund_order(order_id: str) -> str:
    """Refund an order."""
    return f"Refunded {order_id}"


class FakeModel:
    def __init__(self, provider: str) -> None:
        self.provider = provider

    def _get_ls_params(self) -> dict[str, str]:
        return {"ls_provider": self.provider}


class FakeConfigurableModel:
    def __init__(
        self,
        default_config: dict[str, Any] | None = None,
        model_params: dict[str, Any] | None = None,
    ) -> None:
        self._default_config = default_config or {}
        self.model_params = model_params or {}

    def _model_params(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.model_params if config is None else config.get("configurable", {})


class ChatAnthropic:
    """Bare model whose class name is the only provider signal."""


class ChatOpenAI:
    """Bare model whose class name is the only provider signal."""


class MysteryModel:
    """Model that exposes no provider signal at all."""


class FakeRuntime:
    def __init__(self, config: Any) -> None:
        self.config = config


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
    """Without deferral the request must be returned untouched (no needless copy)."""
    request = _request("anthropic", [get_weather, send_email])
    middleware = ProviderToolSearchMiddleware()

    modified_request = _invoke(middleware, request)

    assert modified_request is request


def test_defers_tools_named_in_searchable_tools() -> None:
    """Only named tools get `defer_loading`; others stay intact and the search tool is added."""
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
    assert ANTHROPIC_SEARCH_TOOL in modified_request.tools


def test_accepts_tool_instances_in_searchable_tools() -> None:
    """`searchable_tools` accepts `BaseTool` instances, not just names (the other input form)."""
    request = _request("openai", [get_weather, send_email])
    middleware = ProviderToolSearchMiddleware(searchable_tools=[send_email])

    modified_request = _invoke(middleware, request)

    email_tool = next(
        tool
        for tool in modified_request.tools
        if isinstance(tool, BaseTool) and tool.name == "send_email"
    )
    assert email_tool.extras == {"defer_loading": True}
    assert OPENAI_SEARCH_TOOL in modified_request.tools


def test_honors_tools_pre_marked_with_defer_loading() -> None:
    """A pre-marked `defer_loading` tool triggers deferral even with no `searchable_tools`."""
    request = _request("anthropic", [get_weather, lookup_order])
    middleware = ProviderToolSearchMiddleware()

    modified_request = _invoke(middleware, request)

    order_tool = next(
        tool
        for tool in modified_request.tools
        if isinstance(tool, BaseTool) and tool.name == "lookup_order"
    )
    assert order_tool.extras == {"defer_loading": True}
    assert ANTHROPIC_SEARCH_TOOL in modified_request.tools


def test_preserves_existing_extras_when_deferring() -> None:
    """Deferral merges into existing `extras` rather than overwriting a user's other keys."""
    request = _request("anthropic", [refund_order])
    middleware = ProviderToolSearchMiddleware(searchable_tools=["refund_order"])

    modified_request = _invoke(middleware, request)

    refund_tool = next(
        tool
        for tool in modified_request.tools
        if isinstance(tool, BaseTool) and tool.name == "refund_order"
    )
    assert refund_tool.extras == {"category": "billing", "defer_loading": True}


def test_passes_dict_tools_through_untouched() -> None:
    """Dict-form tools have no name/extras and must pass through without an `AttributeError`."""
    web_search = {"type": "web_search"}
    request = _request("anthropic", [get_weather, web_search])
    middleware = ProviderToolSearchMiddleware(searchable_tools=["get_weather"])

    modified_request = _invoke(middleware, request)

    assert web_search in modified_request.tools
    assert ANTHROPIC_SEARCH_TOOL in modified_request.tools


def test_raises_when_searchable_tool_is_not_bound() -> None:
    """A typo'd `searchable_tools` name fails loudly instead of silently never deferring."""
    request = _request("anthropic", [get_weather])
    middleware = ProviderToolSearchMiddleware(searchable_tools=["missing_tool"])

    with pytest.raises(ValueError, match="missing_tool"):
        _invoke(middleware, request)


def test_unbound_tool_error_is_sorted() -> None:
    """The error lists unknown tools in sorted order so the message is deterministic."""
    request = _request("anthropic", [get_weather])
    middleware = ProviderToolSearchMiddleware(searchable_tools=["zzz", "aaa"])

    with pytest.raises(ValueError, match="aaa, zzz"):
        _invoke(middleware, request)


def test_unbound_tool_check_precedes_provider_check() -> None:
    """Config errors (unbound tool) surface before provider errors, pinning the check order."""
    request = _request("mistralai", [get_weather])
    middleware = ProviderToolSearchMiddleware(searchable_tools=["missing_tool"])

    with pytest.raises(ValueError, match="not bound to the model"):
        _invoke(middleware, request)


def test_passes_through_unsupported_provider_when_nothing_deferred() -> None:
    """An unsupported provider must not raise when no tool is actually deferred."""
    request = _request("mistralai", [get_weather])
    middleware = ProviderToolSearchMiddleware()

    modified_request = _invoke(middleware, request)

    assert modified_request is request


def test_passes_through_unsupported_provider_with_empty_tools() -> None:
    """Empty tool list is a clean no-op and never trips the provider guard."""
    request = _request("mistralai", [])
    middleware = ProviderToolSearchMiddleware()

    modified_request = _invoke(middleware, request)

    assert modified_request is request


def test_raises_for_unsupported_provider_when_tool_deferred() -> None:
    """Deferring against a provider without tool search is a hard error, not a silent drop."""
    request = _request("mistralai", [send_email])
    middleware = ProviderToolSearchMiddleware(searchable_tools=["send_email"])

    with pytest.raises(ValueError, match="server-side tool search"):
        _invoke(middleware, request)


def test_raises_when_provider_cannot_be_determined() -> None:
    """Detection failure raises a distinct, actionable error rather than a misleading one."""
    request = ModelRequest(
        model=cast("BaseChatModel", MysteryModel()),
        messages=[HumanMessage("hi")],
        tools=[lookup_order],
    )
    middleware = ProviderToolSearchMiddleware()

    with pytest.raises(ValueError, match="could not determine the provider"):
        _invoke(middleware, request)


@pytest.mark.parametrize(
    ("model_factory", "expected_tool"),
    [
        (ChatAnthropic, ANTHROPIC_SEARCH_TOOL),
        (ChatOpenAI, OPENAI_SEARCH_TOOL),
    ],
)
def test_detects_provider_from_class_name(
    model_factory: type, expected_tool: dict[str, str]
) -> None:
    """Class-name fallback identifies the provider when no params/ls_provider are exposed."""
    request = ModelRequest(
        model=cast("BaseChatModel", model_factory()),
        messages=[HumanMessage("hi")],
        tools=[send_email],
    )
    middleware = ProviderToolSearchMiddleware(searchable_tools=["send_email"])

    modified_request = _invoke(middleware, request)

    assert expected_tool in modified_request.tools


def test_normalizes_provider_casing_and_hyphens() -> None:
    """Provider identifiers are normalized so mixed casing still matches the registry."""
    request = _request("Anthropic", [send_email])
    middleware = ProviderToolSearchMiddleware(searchable_tools=["send_email"])

    modified_request = _invoke(middleware, request)

    assert ANTHROPIC_SEARCH_TOOL in modified_request.tools


@pytest.mark.parametrize(
    ("model_name", "expected_tool"),
    [
        ("claude-sonnet-4-5", ANTHROPIC_SEARCH_TOOL),
        ("gpt-5.4", OPENAI_SEARCH_TOOL),
        (OPENAI_REASONING_TEST_MODEL, OPENAI_SEARCH_TOOL),
        (OPENAI_TEST_MODEL, OPENAI_SEARCH_TOOL),
    ],
)
def test_detects_provider_from_bare_model_name(
    model_name: str, expected_tool: dict[str, str]
) -> None:
    """Bare model names (no `provider:` prefix) are mapped via the name heuristics."""
    request = ModelRequest(
        model=cast("BaseChatModel", FakeConfigurableModel(model_params={"model": model_name})),
        messages=[HumanMessage("hi")],
        tools=[send_email],
    )
    middleware = ProviderToolSearchMiddleware(searchable_tools=["send_email"])

    modified_request = _invoke(middleware, request)

    assert expected_tool in modified_request.tools


def test_raises_for_unrecognized_bare_model_name() -> None:
    """An unrecognized bare name yields detection failure, not a wrong-provider guess."""
    request = ModelRequest(
        model=cast("BaseChatModel", FakeConfigurableModel(model_params={"model": "llama-3.1"})),
        messages=[HumanMessage("hi")],
        tools=[send_email],
    )
    middleware = ProviderToolSearchMiddleware(searchable_tools=["send_email"])

    with pytest.raises(ValueError, match="could not determine the provider"):
        _invoke(middleware, request)


def test_detects_provider_from_configurable_model() -> None:
    """A configurable model's `_default_config` `provider:model` string resolves the provider."""
    request = ModelRequest(
        model=cast("BaseChatModel", FakeConfigurableModel({"model": "openai:gpt-5.4"})),
        messages=[HumanMessage("hi")],
        tools=[send_email],
    )
    middleware = ProviderToolSearchMiddleware(searchable_tools=["send_email"])

    modified_request = _invoke(middleware, request)

    assert OPENAI_SEARCH_TOOL in modified_request.tools


def test_detects_provider_from_runtime_configurable_model() -> None:
    """Provider set only via runtime `configurable` is read through `_model_params`."""
    request = ModelRequest(
        model=cast("BaseChatModel", FakeConfigurableModel()),
        messages=[HumanMessage("hi")],
        tools=[send_email],
        runtime=cast("Any", FakeRuntime({"configurable": {"model_provider": "openai"}})),
    )
    middleware = ProviderToolSearchMiddleware(searchable_tools=["send_email"])

    modified_request = _invoke(middleware, request)

    assert OPENAI_SEARCH_TOOL in modified_request.tools


def test_runtime_model_override_uses_default_configurable_model_provider() -> None:
    """`model_provider` in the merged params wins over a runtime `model` of another provider."""
    request = ModelRequest(
        model=cast("BaseChatModel", FakeConfigurableModel({"model_provider": "openai"})),
        messages=[HumanMessage("hi")],
        tools=[send_email],
        runtime=cast("Any", FakeRuntime({"configurable": {"model": "claude-sonnet-4-5"}})),
    )
    middleware = ProviderToolSearchMiddleware(searchable_tools=["send_email"])

    modified_request = _invoke(middleware, request)

    assert OPENAI_SEARCH_TOOL in modified_request.tools


def test_runtime_provider_override_uses_runtime_configurable_model_provider() -> None:
    """Runtime `model_provider` overrides the default config's, confirming the merge precedence."""
    request = ModelRequest(
        model=cast("BaseChatModel", FakeConfigurableModel({"model_provider": "openai"})),
        messages=[HumanMessage("hi")],
        tools=[send_email],
        runtime=cast("Any", FakeRuntime({"configurable": {"model_provider": "anthropic"}})),
    )
    middleware = ProviderToolSearchMiddleware(searchable_tools=["send_email"])

    modified_request = _invoke(middleware, request)

    assert ANTHROPIC_SEARCH_TOOL in modified_request.tools


def test_ignores_non_mapping_runtime_config() -> None:
    """A malformed (non-mapping) runtime config is treated as absent, not propagated to a crash."""
    request = ModelRequest(
        model=cast(
            "BaseChatModel",
            FakeConfigurableModel(model_params={"model_provider": "openai"}),
        ),
        messages=[HumanMessage("hi")],
        tools=[send_email],
        runtime=cast("Any", FakeRuntime("not-a-mapping")),
    )
    middleware = ProviderToolSearchMiddleware(searchable_tools=["send_email"])

    modified_request = _invoke(middleware, request)

    assert OPENAI_SEARCH_TOOL in modified_request.tools


async def test_async_wrap_model_call_defers_tools() -> None:
    """The async path applies the same deferral as the sync path through shared logic."""
    request = _request("openai", [send_email])
    middleware = ProviderToolSearchMiddleware(searchable_tools=["send_email"])
    captured_request = None

    async def handler(model_request: ModelRequest) -> ModelResponse:
        nonlocal captured_request
        captured_request = model_request
        return ModelResponse(result=[AIMessage("ok")])

    await middleware.awrap_model_call(request, handler)

    assert captured_request is not None
    assert OPENAI_SEARCH_TOOL in captured_request.tools

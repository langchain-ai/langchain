"""Provider-side tool search middleware."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeAlias

from langchain_core.tools import BaseTool
from typing_extensions import NotRequired, TypedDict

from langchain.chat_models.base import _attempt_infer_model_provider

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)

ToolIdentifier: TypeAlias = str | BaseTool
"""Tool name or tool instance that can be deferred behind provider tool search."""


class _ServerToolSearchSpec(TypedDict):
    """Provider-native tool search tool descriptor sent to the model as a tool."""

    type: str
    name: NotRequired[str]


# Provider-native tool search descriptors keyed by normalized provider name (see
# `_normalize_provider`). This mapping is the single source of truth for which
# providers support server-side tool search.
#
# The identifiers below are version-stamped by the providers and can go stale;
# re-verify against provider docs when updating:
# - Anthropic: https://docs.langchain.com/oss/python/integrations/chat/anthropic#tool-search
# - OpenAI: server-side `tool_search` tool.
_SERVER_TOOL_SEARCH_TOOLS: dict[str, _ServerToolSearchSpec] = {
    "anthropic": {
        "type": "tool_search_tool_bm25_20251119",
        "name": "tool_search_tool_bm25",
    },
    "openai": {"type": "tool_search"},
}


class ProviderToolSearchMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """Defer selected tools behind provider-native tool search.

    Instead of sending every tool schema on every turn, this middleware marks
    selected tools as deferred (via `extras["defer_loading"]`) and injects the
    provider's server-side tool search tool. The provider then retrieves the
    full schema of a deferred tool only when the model needs it, which keeps the
    request payload small when many tools are bound.

    A tool is deferred when its name (or instance) is passed in `searchable_tools`,
    or when it already carries `extras["defer_loading"] is True`.

    Only providers with server-side tool search are supported (currently
    Anthropic and OpenAI). The provider is inferred from the bound model.

    !!! warning

        This relies on provider-native tool search and only takes effect for
        supported providers. If a tool is deferred but the model's provider
        cannot be identified or does not support tool search, the model call
        raises `ValueError`. When no tool is deferred, the middleware passes the
        request through unchanged regardless of provider.

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain.agents.middleware import ProviderToolSearchMiddleware

        agent = create_agent(
            "anthropic:claude-opus-4-8",
            tools=[get_weather, send_email, lookup_order],
            middleware=[ProviderToolSearchMiddleware(searchable_tools=["lookup_order"])],
        )
        ```
    """

    def __init__(self, *, searchable_tools: list[ToolIdentifier] | None = None) -> None:
        """Initialize provider-side tool search.

        Args:
            searchable_tools: Tools or tool names to defer behind provider-native
                tool search.
        """
        super().__init__()
        self.searchable_tool_names = _to_tool_names(searchable_tools)

    def _prepare_request(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        """Prepare a model request with deferred tools and provider search.

        Validates that every name in `searchable_tools` is bound to the model,
        then (only when at least one tool is deferred) resolves the model's
        provider and injects the provider-native tool search tool. Requests with
        no deferred tools pass through unchanged.

        Args:
            request: Model request to prepare.

        Returns:
            The original request when nothing is deferred, otherwise a new
            request with deferred tools and the provider search tool appended.

        Raises:
            ValueError: If `searchable_tools` references a tool not bound to the
                model, or if a tool is deferred but the model's provider cannot
                be identified or does not support server-side tool search.
        """
        tools = request.tools
        if self.searchable_tool_names:
            available = {tool.name for tool in tools if isinstance(tool, BaseTool)}
            unknown = sorted(self.searchable_tool_names - available)
            if unknown:
                msg = (
                    "ProviderToolSearchMiddleware: searchable_tools references "
                    f"tool(s) not bound to the model: {', '.join(unknown)}"
                )
                raise ValueError(msg)

        if not any(_is_deferred_tool(tool, self.searchable_tool_names) for tool in tools):
            return request

        provider = _get_model_provider(request.model, request.runtime)
        if provider is None:
            msg = (
                "ProviderToolSearchMiddleware could not determine the provider for "
                f"model {request.model.__class__.__name__!r}; server-side tool search "
                f"supports: {', '.join(sorted(_SERVER_TOOL_SEARCH_TOOLS))}"
            )
            raise ValueError(msg)
        if provider not in _SERVER_TOOL_SEARCH_TOOLS:
            msg = (
                "ProviderToolSearchMiddleware requires a provider with server-side "
                f"tool search, but got {provider!r}; supported providers: "
                f"{', '.join(sorted(_SERVER_TOOL_SEARCH_TOOLS))}"
            )
            raise ValueError(msg)

        bound_tools = [_defer_tool_if_needed(tool, self.searchable_tool_names) for tool in tools]
        return request.override(tools=[*bound_tools, dict(_SERVER_TOOL_SEARCH_TOOLS[provider])])

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """Defer tools before invoking the model.

        Args:
            request: Model request to execute.
            handler: Callback that executes the model request.

        Returns:
            The model call result.

        Raises:
            ValueError: If `searchable_tools` references a tool not bound to the
                model, or if a tool is deferred but the model's provider cannot
                be identified or does not support server-side tool search.
        """
        return handler(self._prepare_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """Defer tools before asynchronously invoking the model.

        Args:
            request: Model request to execute.
            handler: Callback that executes the model request.

        Returns:
            The model call result.

        Raises:
            ValueError: If `searchable_tools` references a tool not bound to the
                model, or if a tool is deferred but the model's provider cannot
                be identified or does not support server-side tool search.
        """
        return await handler(self._prepare_request(request))


def _to_tool_names(tools: list[ToolIdentifier] | None) -> set[str]:
    """Convert tool identifiers to names."""
    if tools is None:
        return set()
    return {tool if isinstance(tool, str) else tool.name for tool in tools}


def _is_deferred_tool(tool: BaseTool | dict[str, Any], tool_names: set[str]) -> bool:
    """Return whether a tool should be deferred.

    Only `BaseTool` instances can be deferred; dict-form tools (e.g. provider
    tool specs) have no `extras` or name to match and are never deferred.
    """
    if not isinstance(tool, BaseTool):
        return False
    extras = tool.extras if isinstance(tool.extras, dict) else {}
    return extras.get("defer_loading") is True or tool.name in tool_names


def _defer_tool_if_needed(
    tool: BaseTool | dict[str, Any], tool_names: set[str]
) -> BaseTool | dict[str, Any]:
    """Return the tool with `defer_loading` set, or unchanged if not deferred.

    Returns the input unchanged when the tool should not be deferred or is not a
    `BaseTool` (only `BaseTool` instances carry the `extras` that flags deferral).
    """
    if not _is_deferred_tool(tool, tool_names):
        return tool
    if not isinstance(tool, BaseTool):
        return tool
    extras = {**(tool.extras or {}), "defer_loading": True}
    return tool.model_copy(update={"extras": extras})


def _get_model_provider(model: BaseChatModel, runtime: Any) -> str | None:
    """Infer the normalized provider name for server-side tool search.

    Returns `None` when no provider can be identified, so callers can
    distinguish a detection failure from a provider that is simply unsupported.
    """
    default_config = getattr(model, "_default_config", None)
    model_params_fn = getattr(model, "_model_params", None)
    if callable(model_params_fn):
        config = getattr(runtime, "config", None)
        # `_model_params` expects a config mapping (or None); coerce a malformed
        # non-mapping config to None so it is treated as "no config" rather than
        # raising deep inside the configurable model.
        if config is not None and not isinstance(config, Mapping):
            config = None
        model_params = model_params_fn(config)
        if isinstance(model_params, dict):
            params = (
                {**default_config, **model_params}
                if isinstance(default_config, dict)
                else model_params
            )
            if provider := _provider_from_params(params):
                return provider

    if isinstance(default_config, dict) and (provider := _provider_from_params(default_config)):
        return provider

    get_ls_params = getattr(model, "_get_ls_params", None)
    if callable(get_ls_params):
        ls_params = get_ls_params()
        if isinstance(ls_params, dict) and isinstance(ls_params.get("ls_provider"), str):
            return _normalize_provider(ls_params["ls_provider"])

    return _provider_from_class_name(model.__class__.__name__)


def _provider_from_params(params: dict[str, Any]) -> str | None:
    """Infer the provider from model parameters, or `None` if absent."""
    provider = params.get("model_provider")
    if isinstance(provider, str):
        return _normalize_provider(provider)
    model_name = params.get("model")
    if isinstance(model_name, str):
        return _provider_from_model_name(model_name)
    return None


def _provider_from_model_name(model_name: str) -> str | None:
    """Infer the provider from a model name, or `None` if unrecognized."""
    provider, _, rest = model_name.partition(":")
    if rest:
        return _normalize_provider(provider)
    # The inferred provider is only used for a registry lookup here, so suppress the
    # `model_provider` inference deprecation warning that `_attempt_infer_model_provider`
    # emits for some names (e.g. `gemini*`); its guidance is irrelevant to routing.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        inferred = _attempt_infer_model_provider(model_name)
    return _normalize_provider(inferred) if inferred else None


def _provider_from_class_name(class_name: str) -> str | None:
    """Infer the provider from a model class name, or `None` if unrecognized."""
    if class_name in {"ChatAnthropic", "AnthropicChat"}:
        return "anthropic"
    if class_name in {"ChatOpenAI", "OpenAIChat"}:
        return "openai"
    return None


def _normalize_provider(provider: str) -> str:
    """Normalize a provider identifier by lowercasing and mapping `-` to `_`."""
    return provider.replace("-", "_").lower()

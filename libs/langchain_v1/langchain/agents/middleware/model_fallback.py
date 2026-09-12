"""Model fallback middleware for agents.

When a caching middleware such as `AnthropicPromptCachingMiddleware` wraps this
middleware from the outside, it applies Anthropic `cache_control` markers to the
request *before* the fallback loop runs. Those markers are provider-specific and
cause API errors on non-Anthropic fallback models, so this middleware strips them
from fallback attempts — but only when the fallback model itself cannot accept
Anthropic cache markers. When the fallback is another Anthropic model the markers
are valid and preserve prompt caching, so they are left intact.

The knowledge of the `cache_control` marker is duplicated here (rather than owned
solely by the Anthropic partner package) because an outer caching middleware
never re-runs during fallback and therefore cannot clean up after itself.

Provider built-in tools (OpenAI's `{"type": "web_search"}`, Anthropic's server
tools, Gemini's `{"google_search": {}}`) are provider-scoped in the same way, and
are dropped from fallback attempts that target a different provider.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import _WellKnownOpenAITools
from langgraph.errors import GraphBubbleUp

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain.chat_models import init_chat_model

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, AnyMessage, SystemMessage

logger = logging.getLogger(__name__)


def _sanitize_content_blocks(
    content: str | list[str | dict[str, Any]],
) -> str | list[str | dict[str, Any]]:
    """Remove Anthropic cache markers from message content blocks."""
    if not isinstance(content, list):
        return content

    sanitized_content: list[str | dict[str, Any]] = []
    changed = False

    for block in content:
        if not isinstance(block, dict):
            sanitized_content.append(block)
            continue

        sanitized_block, block_changed = _without_cache_control_from_content_block(block)
        changed = changed or block_changed
        sanitized_content.append(sanitized_block)

    return sanitized_content if changed else content


def _sanitize_system_message(
    system_message: SystemMessage | None,
) -> SystemMessage | None:
    """Remove Anthropic cache markers from a system message."""
    if system_message is None:
        return None

    sanitized_content = _sanitize_content_blocks(system_message.content)
    if sanitized_content is system_message.content:
        return system_message

    return system_message.model_copy(update={"content": sanitized_content})


def _sanitize_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Remove Anthropic cache markers from request messages."""
    sanitized_messages: list[AnyMessage] = []
    changed = False

    for message in messages:
        sanitized_message, message_changed = _sanitize_message(message)
        changed = changed or message_changed
        sanitized_messages.append(sanitized_message)

    return sanitized_messages if changed else messages


def _sanitize_tools(
    tools: list[BaseTool | dict[str, Any]],
) -> list[BaseTool | dict[str, Any]]:
    """Remove Anthropic cache markers from tool payloads."""
    sanitized_tools: list[BaseTool | dict[str, Any]] = []
    changed = False

    for tool in tools:
        sanitized_tool: BaseTool | dict[str, Any]
        if isinstance(tool, BaseTool):
            sanitized_tool, tool_changed = _sanitize_base_tool(tool)
        else:
            sanitized_tool, tool_changed = _sanitize_dict_tool(tool)

        changed = changed or tool_changed
        sanitized_tools.append(sanitized_tool)

    return sanitized_tools if changed else tools


def _sanitize_request_for_fallback(request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
    """Sanitize provider-specific Anthropic cache markers before fallback attempts."""
    overrides: dict[str, Any] = {}

    model_settings, model_settings_changed = _without_cache_control(request.model_settings)
    if model_settings_changed:
        overrides["model_settings"] = model_settings

    system_message = _sanitize_system_message(request.system_message)
    if system_message is not request.system_message:
        overrides["system_message"] = system_message

    messages = _sanitize_messages(request.messages)
    if messages is not request.messages:
        overrides["messages"] = messages

    tools = _sanitize_tools(request.tools)
    if tools is not request.tools:
        overrides["tools"] = tools

    if not overrides:
        return request

    # Log only the field names that changed, never request content (may contain
    # prompt data or PII).
    logger.debug(
        "Stripped Anthropic cache_control markers from %s before fallback attempt",
        sorted(overrides),
    )

    return request.override(**overrides)


def _sanitize_message(message: AnyMessage) -> tuple[AnyMessage, bool]:
    """Remove Anthropic cache markers from a single message.

    Returns:
        The sanitized message (the original instance when unchanged) and whether
            any marker was removed.
    """
    sanitized_content = _sanitize_content_blocks(message.content)
    if sanitized_content is message.content:
        return message, False

    return message.model_copy(update={"content": sanitized_content}), True


def _sanitize_base_tool(tool: BaseTool) -> tuple[BaseTool, bool]:
    """Remove Anthropic cache markers from a `BaseTool` payload.

    Returns:
        The sanitized tool (the original instance when unchanged) and whether any
            marker was removed.

            Emptied `extras` collapse back to `None`.
    """
    if not tool.extras:
        return tool, False

    sanitized_extras, changed = _without_cache_control(tool.extras)
    if not changed:
        return tool, False

    return tool.model_copy(update={"extras": sanitized_extras or None}), True


def _sanitize_dict_tool(tool: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Remove Anthropic cache markers from a dict-style tool payload.

    Returns:
        The sanitized tool (the original instance when unchanged) and whether any
            marker was removed.

            Emptied `extras` collapse back to `None`.
    """
    sanitized_tool, changed = _without_cache_control(tool)

    extras = sanitized_tool.get("extras")
    if not isinstance(extras, dict):
        return sanitized_tool, changed

    sanitized_extras, extras_changed = _without_cache_control(extras)
    if not extras_changed:
        return sanitized_tool, changed

    return {**sanitized_tool, "extras": sanitized_extras or None}, True


def _without_cache_control(payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Return payload without `cache_control`, plus whether anything changed."""
    if "cache_control" not in payload:
        return payload, False

    return (
        {key: value for key, value in payload.items() if key != "cache_control"},
        True,
    )


def _without_cache_control_from_content_block(
    block: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Return content block without Anthropic cache markers.

    Strips `cache_control` from the block itself and from its nested `extras` and
    `metadata` payloads.

    Returns:
        The sanitized block (the original instance when unchanged) and whether any
            marker was removed.
    """
    sanitized_block, changed = _without_cache_control(block)

    for nested_key in ("extras", "metadata"):
        nested_payload = sanitized_block.get(nested_key)
        if not isinstance(nested_payload, dict):
            continue

        sanitized_payload, nested_changed = _without_cache_control(nested_payload)
        if not nested_changed:
            continue

        if sanitized_block is block:
            sanitized_block = dict(block)
        sanitized_block[nested_key] = sanitized_payload
        changed = True

    return sanitized_block, changed


# `_llm_type` values that indicate a model speaks an Anthropic-compatible API
# and therefore accepts `cache_control` markers. Direct Anthropic models
# (`ChatAnthropic`) report `"anthropic-chat"`; Bedrock-hosted Claude
# (`ChatAnthropicBedrock`, a `ChatAnthropic` subclass in `langchain-aws`) reports
# `"anthropic-bedrock-chat"` and translates the top-level kwarg into block-level
# breakpoints inside the inherited `ChatAnthropic._get_request_payload`, while
# content-block and tool `cache_control` markers pass through unchanged.
# Vertex-hosted Claude (`ChatAnthropicVertex` in `langchain-google`) reports
# `"anthropic-chat-vertexai"` and nests the same marker shape through its own
# request builder — not the shared `ChatAnthropic` method. All three keep prompt
# caching intact on fallback.
#
# Keep this set in sync with those classes' `_llm_type` values, which live in
# separate repositories. If a value drifts or a new Anthropic transport ships,
# the failure mode is silent loss of prompt caching (markers stripped from a
# model that supports them), not a hard error — so CI here will not catch it.
_ANTHROPIC_LLM_TYPES: frozenset[str] = frozenset(
    {
        "anthropic-chat",
        "anthropic-bedrock-chat",
        "anthropic-chat-vertexai",
    }
)


def _supports_anthropic_cache_control(model: BaseChatModel) -> bool:
    """Return whether `model` accepts Anthropic `cache_control` markers.

    Checked via `_llm_type` so the decision is provider-based rather than
    model-name-based: any Anthropic-compatible model (including future model IDs
    we have not seen) keeps its cache markers on fallback, while OpenAI, Gemini,
    and other non-Anthropic providers get a sanitized request.
    """
    llm_type = getattr(model, "_llm_type", None)
    return isinstance(llm_type, str) and llm_type in _ANTHROPIC_LLM_TYPES


# Provider built-in tools are passed as dicts and are only accepted by the provider
# that defines them. OpenAI Responses built-ins are `{"type": <name>}` entries, with
# dated variants (`web_search_preview_2025_03_11`); Anthropic tools carry a dated
# `type` (`web_search_20250305`) plus a few undated ones; Gemini built-ins are keyed
# payloads (`{"google_search": {}}`) in either snake or camel case. Provider-neutral
# function tools — `BaseTool`, `{"type": "function", ...}`, OpenAI's `namespace`
# grouping — match none of these shapes and are never dropped.
# Sourced from core's own allowlist so the two cannot drift, minus the two entries
# that carry client tools rather than a provider capability (`function` wraps a
# function schema, `namespace` groups them) — dropping those would delete real tools.
# `local_shell` and `shell` are Responses tools that core does not list.
_OPENAI_BUILTIN_TOOL_TYPES: frozenset[str] = (
    frozenset(_WellKnownOpenAITools) - {"function", "namespace"}
) | {"local_shell", "shell"}
_ANTHROPIC_BUILTIN_TOOL_TYPES: frozenset[str] = frozenset(
    {
        "mcp_toolset",
        "tool_search_tool_bm25",
        "tool_search_tool_regex",
    }
)
_ANTHROPIC_DATED_BUILTIN_TOOL_TYPE = re.compile(r"_\d{8}$")
# Every field of the Gemini `Tool` payload, `functionDeclarations` included: a Gemini
# tool object is Gemini-shaped in all of its parts, so no other provider can read one
# even when it carries function declarations.
_GOOGLE_TOOL_KEYS: frozenset[str] = frozenset(
    {
        "codeexecution",
        "computeruse",
        "enterprisewebsearch",
        "filesearch",
        "functiondeclarations",
        "googlemaps",
        "googlesearch",
        "googlesearchretrieval",
        "mcpservers",
        "retrieval",
        "urlcontext",
    }
)

# Substring matched against `_llm_type`, ordered so `anthropic-chat-vertexai`
# resolves to Anthropic rather than Google.
_LLM_TYPE_PROVIDERS: tuple[tuple[str, str], ...] = (
    ("anthropic", "anthropic"),
    ("openai", "openai"),
    ("google", "google"),
    ("gemini", "google"),
    ("vertex", "google"),
)


def _model_provider(model: BaseChatModel) -> str | None:
    """Return the provider whose built-in tools `model` accepts, if recognized."""
    llm_type = getattr(model, "_llm_type", None)
    if not isinstance(llm_type, str):
        return None
    return next(
        (provider for marker, provider in _LLM_TYPE_PROVIDERS if marker in llm_type),
        None,
    )


def _builtin_tool_provider(tool: dict[str, Any]) -> str | None:
    """Return the provider that defines this built-in tool, or `None` for plain tools.

    Anthropic is matched before OpenAI: `tool_search_tool_bm25_20251119` would also
    match the `tool_search` prefix, and `web_search_20250305` the `web_search` one.
    """
    tool_type = tool.get("type")
    if isinstance(tool_type, str):
        if tool_type in _ANTHROPIC_BUILTIN_TOOL_TYPES or _ANTHROPIC_DATED_BUILTIN_TOOL_TYPE.search(
            tool_type
        ):
            return "anthropic"
        if any(
            tool_type == name or tool_type.startswith(f"{name}_")
            for name in _OPENAI_BUILTIN_TOOL_TYPES
        ):
            return "openai"
        return None
    if tool and all(_normalized_key(key) in _GOOGLE_TOOL_KEYS for key in tool):
        return "google"
    return None


def _normalized_key(key: str) -> str:
    """Fold a Gemini tool key so snake and camel spellings compare equal."""
    return key.replace("_", "").lower()


def _is_foreign_builtin_tool(tool: BaseTool | dict[str, Any], provider: str | None) -> bool:
    if not isinstance(tool, dict):
        return False
    owner = _builtin_tool_provider(tool)
    return owner is not None and owner != provider


def _without_foreign_builtin_tools(
    request: ModelRequest[ContextT], fallback_model: BaseChatModel
) -> ModelRequest[ContextT]:
    """Drop built-in tools the fallback model's provider does not define.

    An unrecognized fallback provider drops them too: a built-in tool only exists
    on its own provider's API, so losing the capability on a retry beats failing
    the retry outright on an unknown tool type.
    """
    provider = _model_provider(fallback_model)
    tools = [tool for tool in request.tools if not _is_foreign_builtin_tool(tool, provider)]
    if len(tools) == len(request.tools):
        return request

    logger.warning(
        "Dropped %d provider built-in tool(s) not supported by the fallback model",
        len(request.tools) - len(tools),
    )
    return request.override(tools=tools)


def _prepare_request_for_fallback(
    request: ModelRequest[ContextT], fallback_model: BaseChatModel
) -> ModelRequest[ContextT]:
    """Strip everything the fallback model cannot accept from `request`."""
    prepared = (
        request
        if _supports_anthropic_cache_control(fallback_model)
        else _sanitize_request_for_fallback(request)
    )
    return _without_foreign_builtin_tools(prepared, fallback_model)


def _log_fallback_attempt(exception: Exception, fallback_model: BaseChatModel) -> None:
    """Report a fallback attempt without logging request content (may hold PII)."""
    logger.warning(
        "Model call failed with %s; retrying with fallback model %s",
        type(exception).__name__,
        _model_label(fallback_model),
    )
    logger.debug("Model call failure preceding fallback", exc_info=exception)


def _model_label(model: BaseChatModel) -> str:
    name = getattr(model, "model_name", None) or getattr(model, "model", None)
    return name if isinstance(name, str) and name else type(model).__name__


class ModelFallbackMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """Automatic fallback to alternative models on errors.

    Retries failed model calls with alternative models in sequence until
    success or all models exhausted. Primary model specified in `create_agent`.

    Example:
        ```python
        from langchain.agents.middleware import ModelFallbackMiddleware
        from langchain.agents import create_agent

        fallback = ModelFallbackMiddleware(
            "openai:gpt-5.5",  # Try first on error
            "anthropic:claude-sonnet-4-5-20250929",  # Then this
        )

        agent = create_agent(
            model="openai:gpt-5.5",  # Primary model
            middleware=[fallback],
        )

        # If primary fails: tries gpt-5.5, then claude-sonnet-4-5-20250929
        result = await agent.invoke({"messages": [HumanMessage("Hello")]})
        ```
    """

    def __init__(
        self,
        first_model: str | BaseChatModel,
        *additional_models: str | BaseChatModel,
    ) -> None:
        """Initialize model fallback middleware.

        Args:
            first_model: First fallback model (string name or instance).
            *additional_models: Additional fallbacks in order.
        """
        super().__init__()

        # Initialize all fallback models
        all_models = (first_model, *additional_models)
        self.models: list[BaseChatModel] = []
        for model in all_models:
            if isinstance(model, str):
                self.models.append(init_chat_model(model))
            else:
                self.models.append(model)

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """Try fallback models in sequence on errors.

        Args:
            request: Initial model request.
            handler: Callback to execute the model.

        Returns:
            AIMessage from successful model call.

        Raises:
            Exception: If all models fail, re-raises last exception.
        """
        # Try primary model first
        last_exception: Exception
        try:
            return handler(request)
        except GraphBubbleUp:
            raise
        except Exception as e:
            last_exception = e

        # Try fallback models — the request is prepared outside the try so a
        # sanitizer or `_llm_type` bug surfaces directly instead of being masked
        # as a model failure.
        for fallback_model in self.models:
            fallback_request = _prepare_request_for_fallback(request, fallback_model)
            _log_fallback_attempt(last_exception, fallback_model)
            try:
                return handler(fallback_request.override(model=fallback_model))
            except GraphBubbleUp:
                raise
            except Exception as e:
                last_exception = e
                continue

        raise last_exception

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """Try fallback models in sequence on errors (async version).

        Args:
            request: Initial model request.
            handler: Async callback to execute the model.

        Returns:
            AIMessage from successful model call.

        Raises:
            Exception: If all models fail, re-raises last exception.
        """
        # Try primary model first
        last_exception: Exception
        try:
            return await handler(request)
        except GraphBubbleUp:
            raise
        except Exception as e:
            last_exception = e

        # Try fallback models — the request is prepared outside the try so a
        # sanitizer or `_llm_type` bug surfaces directly instead of being masked
        # as a model failure.
        for fallback_model in self.models:
            fallback_request = _prepare_request_for_fallback(request, fallback_model)
            _log_fallback_attempt(last_exception, fallback_model)
            try:
                return await handler(fallback_request.override(model=fallback_model))
            except GraphBubbleUp:
                raise
            except Exception as e:
                last_exception = e
                continue

        raise last_exception

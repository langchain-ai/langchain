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
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool

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
        except Exception as e:
            last_exception = e

        # Try fallback models — sanitize cache markers only when the fallback
        # model cannot accept them (i.e. is not an Anthropic-compatible model).
        # The request is derived outside the try so a sanitizer or `_llm_type`
        # bug surfaces directly instead of being masked as a model failure.
        for fallback_model in self.models:
            fallback_request = (
                request
                if _supports_anthropic_cache_control(fallback_model)
                else _sanitize_request_for_fallback(request)
            )
            try:
                return handler(fallback_request.override(model=fallback_model))
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
        except Exception as e:
            last_exception = e

        # Try fallback models — sanitize cache markers only when the fallback
        # model cannot accept them (i.e. is not an Anthropic-compatible model).
        # The request is derived outside the try so a sanitizer or `_llm_type`
        # bug surfaces directly instead of being masked as a model failure.
        for fallback_model in self.models:
            fallback_request = (
                request
                if _supports_anthropic_cache_control(fallback_model)
                else _sanitize_request_for_fallback(request)
            )
            try:
                return await handler(fallback_request.override(model=fallback_model))
            except Exception as e:
                last_exception = e
                continue

        raise last_exception

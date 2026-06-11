"""Model fallback middleware for agents."""

from __future__ import annotations

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

        sanitized_block, block_changed = _without_cache_control(block)
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

    return request.override(**overrides)


def _sanitize_message(message: AnyMessage) -> tuple[AnyMessage, bool]:
    """Remove Anthropic cache markers from a single message."""
    sanitized_content = _sanitize_content_blocks(message.content)
    if sanitized_content is message.content:
        return message, False

    return message.model_copy(update={"content": sanitized_content}), True


def _sanitize_base_tool(tool: BaseTool) -> tuple[BaseTool, bool]:
    """Remove Anthropic cache markers from a `BaseTool` payload."""
    if not tool.extras:
        return tool, False

    sanitized_extras, changed = _without_cache_control(tool.extras)
    if not changed:
        return tool, False

    return tool.model_copy(update={"extras": sanitized_extras or None}), True


def _sanitize_dict_tool(tool: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Remove Anthropic cache markers from a dict-style tool payload."""
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

        # Try fallback models
        for fallback_model in self.models:
            try:
                fallback_request = _sanitize_request_for_fallback(
                    request.override(model=fallback_model)
                )
                return handler(fallback_request)
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

        # Try fallback models
        for fallback_model in self.models:
            try:
                fallback_request = _sanitize_request_for_fallback(
                    request.override(model=fallback_model)
                )
                return await handler(fallback_request)
            except Exception as e:
                last_exception = e
                continue

        raise last_exception

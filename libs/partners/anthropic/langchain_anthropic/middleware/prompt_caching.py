"""Anthropic prompt caching middleware.

Requires:
    - `langchain`: For agent middleware framework
    - `langchain-anthropic`: For `ChatAnthropic` model (already a dependency)
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal
from warnings import warn

from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool

from langchain_anthropic.chat_models import ChatAnthropic

try:
    from langchain.agents.middleware.types import (
        AgentMiddleware,
        ModelCallResult,
        ModelRequest,
        ModelResponse,
    )
except ImportError as e:
    msg = (
        "AnthropicPromptCachingMiddleware requires 'langchain' to be installed. "
        "This middleware is designed for use with LangChain agents. "
        "Install it with: pip install langchain"
    )
    raise ImportError(msg) from e


class AnthropicPromptCachingMiddleware(AgentMiddleware):
    """Prompt Caching Middleware.

    Optimizes API usage by caching conversation prefixes for Anthropic models.

    Requires both `langchain` and `langchain-anthropic` packages to be installed.

    Applies cache control breakpoints to:

    - **System message**: Tags the last content block of the system message
        with `cache_control` so static system prompt content is cached.
    - **Tools**: Tags all tool definitions with `cache_control` so tool
        schemas are cached across turns.
    - **Last cacheable block**: Tags last cacheable block of message sequence using
        Anthropic's automatic caching feature.

    Learn more about Anthropic prompt caching
    [here](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).
    """

    def __init__(
        self,
        type: Literal["ephemeral"] = "ephemeral",  # noqa: A002
        ttl: Literal["5m", "1h"] = "5m",
        min_messages_to_cache: int = 0,
        unsupported_model_behavior: Literal["ignore", "warn", "raise"] = "warn",
    ) -> None:
        """Initialize the middleware with cache control settings.

        Args:
            type: The type of cache to use, only `'ephemeral'` is supported.
            ttl: The time to live for the cache, only `'5m'` and `'1h'` are
                supported.
            min_messages_to_cache: The minimum number of messages until the
                cache is used.
            unsupported_model_behavior: The behavior to take when an
                unsupported model is used.

                `'ignore'` will ignore the unsupported model and continue without
                caching.

                `'warn'` will warn the user and continue without caching.

                `'raise'` will raise an error and stop the agent.
        """
        self.type = type
        self.ttl = ttl
        self.min_messages_to_cache = min_messages_to_cache
        self.unsupported_model_behavior = unsupported_model_behavior

    @property
    def _cache_control(self) -> dict[str, str]:
        return {"type": self.type, "ttl": self.ttl}

    def _should_apply_caching(self, request: ModelRequest) -> bool:
        """Check if caching should be applied to the request.

        Args:
            request: The model request to check.

        Returns:
            `True` if caching should be applied, `False` otherwise.

        Raises:
            ValueError: If model is unsupported and behavior is set to `'raise'`.
        """
        if not isinstance(request.model, ChatAnthropic):
            msg = (
                "AnthropicPromptCachingMiddleware caching middleware only supports "
                f"Anthropic models, not instances of {type(request.model)}"
            )
            if self.unsupported_model_behavior == "raise":
                raise ValueError(msg)
            if self.unsupported_model_behavior == "warn":
                warn(msg, stacklevel=3)
            return False

        messages_count = (
            len(request.messages) + 1
            if request.system_message
            else len(request.messages)
        )
        return messages_count >= self.min_messages_to_cache

    def _apply_caching(self, request: ModelRequest) -> ModelRequest:
        """Apply cache control to system message, tools, and model settings.

        Args:
            request: The model request to modify.

        Returns:
            New request with cache control applied.
        """
        overrides: dict[str, Any] = {}
        cache_control = self._cache_control

        # Always set top-level `cache_control` on model settings. The Anthropic
        # chat model translates the kwarg to the correct wire format for the
        # active transport: direct API receives it as-is, while Bedrock has it
        # expanded into a block-level breakpoint by `_get_request_payload`.
        overrides["model_settings"] = {
            **request.model_settings,
            "cache_control": cache_control,
        }

        system_message = _tag_system_message(request.system_message, cache_control)
        if system_message is not request.system_message:
            overrides["system_message"] = system_message

        tools = _tag_tools(request.tools, cache_control)
        if tools is not request.tools:
            overrides["tools"] = tools

        return request.override(**overrides)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Modify the model request to add cache control blocks.

        Args:
            request: The model request to potentially modify.
            handler: The handler to execute the model request.

        Returns:
            The model response from the handler.
        """
        if not self._should_apply_caching(request):
            return handler(request)

        return handler(self._apply_caching(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Modify the model request to add cache control blocks (async version).

        Args:
            request: The model request to potentially modify.
            handler: The async handler to execute the model request.

        Returns:
            The model response from the handler.
        """
        if not self._should_apply_caching(request):
            return await handler(request)

        return await handler(self._apply_caching(request))


def _tag_system_message(
    system_message: Any,
    cache_control: dict[str, str],
) -> Any:
    """Tag the last content block of a system message with cache_control.

    Returns the original system_message unchanged if there are no blocks
    to tag.

    Args:
        system_message: The system message to tag.
        cache_control: The cache control dict to apply.

    Returns:
        A new SystemMessage with cache_control on the last block, or the
        original if no modification was needed.
    """
    if system_message is None:
        return system_message

    content = system_message.content
    if isinstance(content, str):
        if not content:
            return system_message
        new_content: list[str | dict[str, Any]] = [
            {"type": "text", "text": content, "cache_control": cache_control}
        ]
    elif isinstance(content, list):
        if not content:
            return system_message
        new_content = list(content)
        last = new_content[-1]
        base = last if isinstance(last, dict) else {}
        new_content[-1] = {**base, "cache_control": cache_control}
    else:
        return system_message

    return SystemMessage(content=new_content)


def _tag_tools(
    tools: list[Any] | None,
    cache_control: dict[str, str],
) -> list[Any] | None:
    """Tag the last tool with cache_control via its extras dict.

    Only the last tool is tagged to minimize the number of explicit cache
    breakpoints (Anthropic limits these to 4 per request). Since tool
    definitions are sent as a contiguous block, a single breakpoint on the
    last tool caches the entire set.

    Creates a copy of the last tool with cache_control added to extras,
    without mutating the original.

    Args:
        tools: The list of tools to tag.
        cache_control: The cache control dict to apply.

    Returns:
        A new list with cache_control on the last tool's extras, or the
        original if no tools are present.
    """
    if not tools:
        return tools

    last = tools[-1]
    if not isinstance(last, BaseTool):
        return tools

    new_extras = {**(last.extras or {}), "cache_control": cache_control}
    return [*tools[:-1], last.model_copy(update={"extras": new_extras})]

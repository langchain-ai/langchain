"""Anthropic prompt caching middleware.

Requires:
    - langchain: For agent middleware framework
    - langchain-anthropic: For ChatAnthropic model (already a dependency)
"""

from collections.abc import Awaitable, Callable
from typing import Literal
from warnings import warn

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

    Requires both 'langchain' and 'langchain-anthropic' packages to be installed.

    Learn more about Anthropic prompt caching
    [here](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching).
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
            type: The type of cache to use, only "ephemeral" is supported.
            ttl: The time to live for the cache, only "5m" and "1h" are
                supported.
            min_messages_to_cache: The minimum number of messages until the
                cache is used, default is 0.
            unsupported_model_behavior: The behavior to take when an
                unsupported model is used. "ignore" will ignore the unsupported
                model and continue without caching. "warn" will warn the user
                and continue without caching. "raise" will raise an error and
                stop the agent.
        """
        self.type = type
        self.ttl = ttl
        self.min_messages_to_cache = min_messages_to_cache
        self.unsupported_model_behavior = unsupported_model_behavior

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Modify the model request to add cache control blocks."""
        if not isinstance(request.model, ChatAnthropic):
            msg = (
                "AnthropicPromptCachingMiddleware caching middleware only supports "
                f"Anthropic models, not instances of {type(request.model)}"
            )
            if self.unsupported_model_behavior == "raise":
                raise ValueError(msg)
            if self.unsupported_model_behavior == "warn":
                warn(msg, stacklevel=3)
            return handler(request)

        messages_count = (
            len(request.messages) + 1
            if request.system_prompt
            else len(request.messages)
        )
        if messages_count < self.min_messages_to_cache:
            return handler(request)

        request.model_settings["cache_control"] = {"type": self.type, "ttl": self.ttl}

        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Modify the model request to add cache control blocks (async version)."""
        if not isinstance(request.model, ChatAnthropic):
            msg = (
                "AnthropicPromptCachingMiddleware caching middleware only supports "
                f"Anthropic models, not instances of {type(request.model)}"
            )
            if self.unsupported_model_behavior == "raise":
                raise ValueError(msg)
            if self.unsupported_model_behavior == "warn":
                warn(msg, stacklevel=3)
            return await handler(request)

        messages_count = (
            len(request.messages) + 1
            if request.system_prompt
            else len(request.messages)
        )
        if messages_count < self.min_messages_to_cache:
            return await handler(request)

        request.model_settings["cache_control"] = {"type": self.type, "ttl": self.ttl}

        return await handler(request)

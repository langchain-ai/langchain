"""Anthropic prompt caching middleware."""

from typing import Literal
from warnings import warn

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest


class AnthropicPromptCachingMiddleware(AgentMiddleware):
    """Prompt Caching Middleware.

    Optimizes API usage by caching conversation prefixes for Anthropic models.

    Learn more about Anthropic prompt caching
    `here <https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching>`__.
    """

    def __init__(
        self,
        type: Literal["ephemeral"] = "ephemeral",
        ttl: Literal["5m", "1h"] = "5m",
        min_messages_to_cache: int = 0,
        unsupported_model_behavior: Literal["ignore", "warn", "raise"] = "warn",
    ) -> None:
        """Initialize the middleware with cache control settings.

        Args:
            type: The type of cache to use, only "ephemeral" is supported.
            ttl: The time to live for the cache, only "5m" and "1h" are supported.
            min_messages_to_cache: The minimum number of messages until the cache is used,
                default is 0.
            unsupported_model_behavior: The behavior to take when an unsupported model is used.
                "ignore" will ignore the unsupported model and continue without caching.
                "warn" will warn the user and continue without caching.
                "raise" will raise an error and stop the agent.
        """
        self.type = type
        self.ttl = ttl
        self.min_messages_to_cache = min_messages_to_cache
        self.unsupported_model_behavior = unsupported_model_behavior

    def modify_model_request(  # type: ignore[override]
        self,
        request: ModelRequest,
    ) -> ModelRequest:
        """Modify the model request to add cache control blocks."""
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            ChatAnthropic = None  # noqa: N806

        msg: str | None = None

        if ChatAnthropic is None:
            msg = (
                "AnthropicPromptCachingMiddleware caching middleware only supports "
                "Anthropic models. "
                "Please install langchain-anthropic."
            )
        elif not isinstance(request.model, ChatAnthropic):
            msg = (
                "AnthropicPromptCachingMiddleware caching middleware only supports "
                f"Anthropic models, not instances of {type(request.model)}"
            )

        if msg is not None:
            if self.unsupported_model_behavior == "raise":
                raise ValueError(msg)
            if self.unsupported_model_behavior == "warn":
                warn(msg, stacklevel=3)
            else:
                return request

        messages_count = (
            len(request.messages) + 1 if request.system_prompt else len(request.messages)
        )
        if messages_count < self.min_messages_to_cache:
            return request

        request.model_settings["cache_control"] = {"type": self.type, "ttl": self.ttl}

        return request

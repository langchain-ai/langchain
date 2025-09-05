from typing import Literal

from langchain.agents.types import AgentMiddleware, AgentState, ModelRequest


class AnthropicPromptCachingMiddleware(AgentMiddleware):
    """Prompt Caching Middleware - Optimizes API usage by caching conversation prefixes for Anthropic models.

    Learn more about anthropic prompt caching [here](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching).
    """

    def __init__(
        self,
        type: Literal["ephemeral"] = "ephemeral",
        ttl: Literal["5m", "1h"] = "5m",
        min_messages_to_cache: int = 0,
    ):
        """Initialize the middleware with cache control settings.

        Args:
            type: The type of cache to use, only "ephemeral" is supported.
            ttl: The time to live for the cache, only "5m" and "1h" are supported.
            min_messages_to_cache: The minimum number of messages until the cache is used, default is 0.
        """
        self.type = type
        self.ttl = ttl
        self.min_messages_to_cache = min_messages_to_cache

    def modify_model_request(self, request: ModelRequest, state: AgentState) -> ModelRequest:
        """Modify the model request to add cache control blocks."""
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ValueError(
                "AnthropicPromptCachingMiddleware caching middleware only supports Anthropic models."
                "Please install langchain-anthropic."
            )

        if not isinstance(request.model, ChatAnthropic):
            raise ValueError(
                "AnthropicPromptCachingMiddleware caching middleware only supports Anthropic models, "
                f"not instances of {type(request.model)}"
            )

        messages_count = (
            len(request.messages) + 1 if request.system_prompt else len(request.messages)
        )
        if messages_count < self.min_messages_to_cache:
            return request

        request.model_settings["cache_control"] = {"type": self.type, "ttl": self.ttl}

        return request

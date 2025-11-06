from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware


class TestMiddlewareWithFallback:
    """Test middleware behavior with actual model fallback scenarios."""

    def test_fallback_to_openai_without_error(self) -> None:
        """Test that fallback to OpenAI works without cache_control errors."""
        # This test requires API keys but can be mocked for CI
        # Skip if API keys not available
        pytest.importorskip("langchain_openai")

        from langchain_openai import ChatOpenAI  # type: ignore[import-not-found]

        # Create agent with fallback middleware
        # Use invalid Anthropic model to force immediate fallback
        agent: Any = create_agent(
            model=ChatAnthropic(model="invalid-model-name", api_key="invalid"),
            checkpointer=MemorySaver(),
            middleware=[
                ModelFallbackMiddleware(
                    ChatOpenAI(model="gpt-4o-mini")  # Will fallback to this
                ),
                AnthropicPromptCachingMiddleware(ttl="5m"),
            ],
        )

        # This should not raise TypeError about cache_control
        # It will fail due to invalid API keys, but that's expected
        # The key is that it doesn't fail with cache_control error
        config: Any = {"configurable": {"thread_id": "test"}}

        with pytest.raises(Exception) as exc_info:
            agent.invoke({"messages": [HumanMessage(content="Hello")]}, config)
        # Should not be a cache_control TypeError
        assert "cache_control" not in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_async_fallback_to_openai(self) -> None:
        """Test async version with fallback to OpenAI."""
        pytest.importorskip("langchain_openai")

        from langchain_openai import ChatOpenAI  # type: ignore[import-not-found]

        agent: Any = create_agent(
            model=ChatAnthropic(model="invalid-model", api_key="invalid"),
            checkpointer=MemorySaver(),
            middleware=[
                ModelFallbackMiddleware(ChatOpenAI(model="gpt-4o-mini")),
                AnthropicPromptCachingMiddleware(ttl="5m"),
            ],
        )

        config: Any = {"configurable": {"thread_id": "test"}}

        with pytest.raises(Exception) as exc_info:
            await agent.ainvoke({"messages": [HumanMessage(content="Hello")]}, config)
        # Should not be a cache_control TypeError
        assert "cache_control" not in str(exc_info.value).lower()

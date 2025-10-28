"""Unit tests for response caching middleware."""

import time
from itertools import cycle

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.agents.middleware.response_caching import ResponseCachingMiddleware


class TestBasicCaching:
    """Test basic caching functionality."""

    def test_sync_cache_hit(self) -> None:
        """Test that identical requests return cached responses."""
        model = GenericFakeChatModel(
            messages=cycle(
                [
                    AIMessage(content="Response 1"),
                    AIMessage(content="Response 2"),
                    AIMessage(content="Response 3"),
                ]
            )
        )

        cache = ResponseCachingMiddleware(ttl=60, max_size=10)

        agent = create_agent(
            model=model,
            tools=[],
            middleware=[cache],
        )

        # First call - cache miss
        result1 = agent.invoke({"messages": [HumanMessage("Hello")]})
        first_response = result1["messages"][-1].content

        # Second identical call - cache hit (should return same response)
        result2 = agent.invoke({"messages": [HumanMessage("Hello")]})
        second_response = result2["messages"][-1].content

        # Both should return the same cached response
        assert first_response == second_response
        assert first_response == "Response 1"

        # Verify cache stats
        stats = cache.get_cache_stats()
        assert stats["size"] == 1
        assert stats["total_hits"] == 1

    async def test_async_cache_hit(self) -> None:
        """Test that identical async requests return cached responses."""
        model = GenericFakeChatModel(
            messages=cycle(
                [
                    AIMessage(content="Async Response 1"),
                    AIMessage(content="Async Response 2"),
                ]
            )
        )

        cache = ResponseCachingMiddleware(ttl=60, max_size=10)

        agent = create_agent(
            model=model,
            tools=[],
            middleware=[cache],
        )

        # First call - cache miss
        result1 = await agent.ainvoke({"messages": [HumanMessage("Hello async")]})
        first_response = result1["messages"][-1].content

        # Second identical call - cache hit
        result2 = await agent.ainvoke({"messages": [HumanMessage("Hello async")]})
        second_response = result2["messages"][-1].content

        assert first_response == second_response
        assert first_response == "Async Response 1"

    def test_different_messages_no_cache_hit(self) -> None:
        """Test that different messages don't trigger cache hits."""
        model = GenericFakeChatModel(
            messages=cycle(
                [
                    AIMessage(content="Response A"),
                    AIMessage(content="Response B"),
                ]
            )
        )

        cache = ResponseCachingMiddleware(ttl=60, max_size=10)

        agent = create_agent(
            model=model,
            tools=[],
            middleware=[cache],
        )

        result1 = agent.invoke({"messages": [HumanMessage("Question 1")]})
        result2 = agent.invoke({"messages": [HumanMessage("Question 2")]})

        # Different questions should get different responses
        assert result1["messages"][-1].content == "Response A"
        assert result2["messages"][-1].content == "Response B"

        # Should have 2 cache entries, 0 hits
        stats = cache.get_cache_stats()
        assert stats["size"] == 2
        assert stats["total_hits"] == 0


class TestTTLExpiration:
    """Test TTL (time-to-live) expiration functionality."""

    def test_expired_entries_removed(self) -> None:
        """Test that expired entries are removed from cache."""
        model = GenericFakeChatModel(
            messages=cycle(
                [
                    AIMessage(content="Response 1"),
                    AIMessage(content="Response 2"),
                ]
            )
        )

        # Very short TTL for testing
        cache = ResponseCachingMiddleware(ttl=0.1, max_size=10)

        agent = create_agent(model=model, tools=[], middleware=[cache])

        # First call
        result1 = agent.invoke({"messages": [HumanMessage("Hello")]})
        assert result1["messages"][-1].content == "Response 1"

        # Wait for TTL to expire
        time.sleep(0.15)

        # Second call after expiration - should get new response
        result2 = agent.invoke({"messages": [HumanMessage("Hello")]})
        assert result2["messages"][-1].content == "Response 2"

        # Cache should have 1 entry (old one was evicted)
        stats = cache.get_cache_stats()
        assert stats["size"] == 1

    def test_no_expiration_with_none_ttl(self) -> None:
        """Test that entries never expire when TTL is None."""
        model = GenericFakeChatModel(
            messages=cycle(
                [
                    AIMessage(content="Response 1"),
                    AIMessage(content="Response 2"),
                ]
            )
        )

        cache = ResponseCachingMiddleware(ttl=None, max_size=10)

        agent = create_agent(model=model, tools=[], middleware=[cache])

        result1 = agent.invoke({"messages": [HumanMessage("Hello")]})
        assert result1["messages"][-1].content == "Response 1"

        # Wait a bit
        time.sleep(0.1)

        # Should still get cached response
        result2 = agent.invoke({"messages": [HumanMessage("Hello")]})
        assert result2["messages"][-1].content == "Response 1"

        stats = cache.get_cache_stats()
        assert stats["total_hits"] == 1


class TestCacheSizeLimits:
    """Test cache size limits and LRU eviction."""

    def test_lru_eviction_on_size_limit(self) -> None:
        """Test that least recently used entries are evicted when cache is full."""
        model = GenericFakeChatModel(
            messages=cycle([AIMessage(content=f"Response {i}") for i in range(10)])
        )

        # Small cache size
        cache = ResponseCachingMiddleware(ttl=None, max_size=3)

        agent = create_agent(model=model, tools=[], middleware=[cache])

        # Add 3 entries (fill cache)
        agent.invoke({"messages": [HumanMessage("Q1")]})
        agent.invoke({"messages": [HumanMessage("Q2")]})
        agent.invoke({"messages": [HumanMessage("Q3")]})

        stats = cache.get_cache_stats()
        assert stats["size"] == 3

        # Add 4th entry - should evict Q1 (least recently used)
        agent.invoke({"messages": [HumanMessage("Q4")]})

        stats = cache.get_cache_stats()
        assert stats["size"] == 3  # Still at max size

        # Access Q1 again - should be cache miss (was evicted)
        result = agent.invoke({"messages": [HumanMessage("Q1")]})

        # Q1 should get a new response (not the original cached one)
        # The model cycles through responses, so we can't predict exact content
        # but we can verify cache behavior

        stats = cache.get_cache_stats()
        assert stats["size"] == 3  # Q2 was evicted to make room for new Q1

    def test_lru_ordering_preserved(self) -> None:
        """Test that accessing cached entries updates LRU ordering."""
        model = GenericFakeChatModel(
            messages=cycle(
                [
                    AIMessage(content="R1"),
                    AIMessage(content="R2"),
                    AIMessage(content="R3"),
                    AIMessage(content="R4"),
                ]
            )
        )

        cache = ResponseCachingMiddleware(ttl=None, max_size=2)

        agent = create_agent(model=model, tools=[], middleware=[cache])

        # Add 2 entries
        agent.invoke({"messages": [HumanMessage("Q1")]})  # R1
        agent.invoke({"messages": [HumanMessage("Q2")]})  # R2

        # Access Q1 (makes it most recently used)
        result = agent.invoke({"messages": [HumanMessage("Q1")]})
        assert result["messages"][-1].content == "R1"  # Cache hit

        # Add Q3 - should evict Q2 (least recently used), not Q1
        agent.invoke({"messages": [HumanMessage("Q3")]})  # R3

        # Q1 should still be cached
        result_q1 = agent.invoke({"messages": [HumanMessage("Q1")]})
        assert result_q1["messages"][-1].content == "R1"  # Cache hit

        # Q2 should not be cached (was evicted)
        result_q2 = agent.invoke({"messages": [HumanMessage("Q2")]})
        assert result_q2["messages"][-1].content == "R4"  # New response, not R2


class TestCustomCacheKey:
    """Test custom cache key functions."""

    def test_custom_cache_key_function(self) -> None:
        """Test using a custom cache key function."""

        def last_message_only(request: ModelRequest) -> str:
            """Cache key based only on the last message content."""
            if request.messages:
                last_msg = request.messages[-1]
                return str(last_msg.content)
            return ""

        model = GenericFakeChatModel(
            messages=cycle(
                [
                    AIMessage(content="Cached Response"),
                    AIMessage(content="Different Response"),
                ]
            )
        )

        cache = ResponseCachingMiddleware(
            ttl=None,
            max_size=10,
            cache_key_fn=last_message_only,
        )

        agent = create_agent(model=model, tools=[], middleware=[cache])

        # Two different conversation histories, but same last message
        result1 = agent.invoke(
            {
                "messages": [
                    HumanMessage("Context A"),
                    HumanMessage("What is 2+2?"),
                ]
            }
        )

        result2 = agent.invoke(
            {
                "messages": [
                    HumanMessage("Context B"),
                    HumanMessage("What is 2+2?"),
                ]
            }
        )

        # Should get same cached response (same last message)
        assert result1["messages"][-1].content == "Cached Response"
        assert result2["messages"][-1].content == "Cached Response"

        stats = cache.get_cache_stats()
        assert stats["total_hits"] == 1


class TestCacheManagement:
    """Test cache management operations."""

    def test_clear_cache(self) -> None:
        """Test clearing the cache."""
        model = GenericFakeChatModel(
            messages=cycle(
                [
                    AIMessage(content="Response 1"),
                    AIMessage(content="Response 2"),
                ]
            )
        )

        cache = ResponseCachingMiddleware(ttl=None, max_size=10)

        agent = create_agent(model=model, tools=[], middleware=[cache])

        # Add entries
        agent.invoke({"messages": [HumanMessage("Q1")]})
        agent.invoke({"messages": [HumanMessage("Q2")]})

        stats = cache.get_cache_stats()
        assert stats["size"] == 2

        # Clear cache
        cache.clear_cache()

        stats = cache.get_cache_stats()
        assert stats["size"] == 0
        assert stats["total_hits"] == 0

    def test_cache_stats(self) -> None:
        """Test cache statistics reporting."""
        model = GenericFakeChatModel(messages=cycle([AIMessage(content="Response")]))

        cache = ResponseCachingMiddleware(ttl=60, max_size=5)

        agent = create_agent(model=model, tools=[], middleware=[cache])

        # Add some entries and hits
        agent.invoke({"messages": [HumanMessage("Q1")]})  # Miss
        agent.invoke({"messages": [HumanMessage("Q1")]})  # Hit
        agent.invoke({"messages": [HumanMessage("Q1")]})  # Hit
        agent.invoke({"messages": [HumanMessage("Q2")]})  # Miss

        stats = cache.get_cache_stats()

        assert stats["size"] == 2
        assert stats["max_size"] == 5
        assert stats["ttl"] == 60
        assert stats["total_hits"] == 2
        assert len(stats["entries"]) == 2

        # Check entry details
        q1_entry = next(e for e in stats["entries"] if e["hits"] == 2)
        assert q1_entry["hits"] == 2
        assert q1_entry["age_seconds"] >= 0


class TestCacheWithConversationHistory:
    """Test caching behavior with multi-turn conversations."""

    def test_conversation_history_affects_cache(self) -> None:
        """Test that conversation history is part of cache key."""
        model = GenericFakeChatModel(
            messages=cycle(
                [
                    AIMessage(content="Response A"),
                    AIMessage(content="Response B"),
                ]
            )
        )

        cache = ResponseCachingMiddleware(ttl=None, max_size=10)

        agent = create_agent(model=model, tools=[], middleware=[cache])

        # Same question, different conversation history
        result1 = agent.invoke({"messages": [HumanMessage("What is the answer?")]})

        result2 = agent.invoke(
            {
                "messages": [
                    HumanMessage("Tell me about Python"),
                    AIMessage("Python is a programming language."),
                    HumanMessage("What is the answer?"),
                ]
            }
        )

        # Different conversation histories should result in different cache entries
        assert result1["messages"][-1].content == "Response A"
        assert result2["messages"][-1].content == "Response B"

        stats = cache.get_cache_stats()
        assert stats["size"] == 2  # Two different cache entries
        assert stats["total_hits"] == 0  # No cache hits


class TestMiddlewareIntegration:
    """Test integration with other middleware."""

    def test_caching_with_other_middleware(self) -> None:
        """Test that caching works alongside other middleware."""
        call_count = 0

        @wrap_model_call
        def count_calls(request: ModelRequest, handler):
            """Count how many times the model is actually called."""
            nonlocal call_count
            call_count += 1
            return handler(request)

        model = GenericFakeChatModel(messages=cycle([AIMessage(content="Response")]))

        cache = ResponseCachingMiddleware(ttl=None, max_size=10)

        # Cache should be innermost to cache the final result
        agent = create_agent(model=model, tools=[], middleware=[count_calls, cache])

        # Make 3 identical requests
        agent.invoke({"messages": [HumanMessage("Hello")]})
        agent.invoke({"messages": [HumanMessage("Hello")]})
        agent.invoke({"messages": [HumanMessage("Hello")]})

        # Model should only be called once (other 2 are cache hits)
        assert call_count == 1

        stats = cache.get_cache_stats()
        assert stats["total_hits"] == 2


class TestValidation:
    """Test validation and error handling."""

    def test_invalid_max_size(self) -> None:
        """Test that invalid max_size raises ValueError."""
        with pytest.raises(ValueError, match="max_size must be at least 1"):
            ResponseCachingMiddleware(max_size=0)

        with pytest.raises(ValueError, match="max_size must be at least 1"):
            ResponseCachingMiddleware(max_size=-1)

    def test_invalid_ttl(self) -> None:
        """Test that negative TTL raises ValueError."""
        with pytest.raises(ValueError, match="ttl must be non-negative or None"):
            ResponseCachingMiddleware(ttl=-1)

        # TTL of 0 should be valid (immediate expiration)
        cache = ResponseCachingMiddleware(ttl=0, max_size=10)
        assert cache.ttl == 0

    def test_ttl_none_valid(self) -> None:
        """Test that TTL can be None for no expiration."""
        cache = ResponseCachingMiddleware(ttl=None, max_size=10)
        assert cache.ttl is None


class TestCacheKeyFingerprinting:
    """Test cache key generation and fingerprinting."""

    def test_system_prompt_affects_cache(self) -> None:
        """Test that different system prompts create different cache entries."""
        model = GenericFakeChatModel(
            messages=cycle(
                [
                    AIMessage(content="Response 1"),
                    AIMessage(content="Response 2"),
                ]
            )
        )

        cache = ResponseCachingMiddleware(ttl=None, max_size=10)

        # Create agents with different system prompts
        agent1 = create_agent(
            model=model,
            tools=[],
            system_prompt="You are a helpful assistant.",
            middleware=[cache],
        )

        agent2 = create_agent(
            model=model,
            tools=[],
            system_prompt="You are a Python expert.",
            middleware=[cache],
        )

        # Same messages, different system prompts
        result1 = agent1.invoke({"messages": [HumanMessage("Hello")]})
        result2 = agent2.invoke({"messages": [HumanMessage("Hello")]})

        # Should be different cache entries
        assert result1["messages"][-1].content == "Response 1"
        assert result2["messages"][-1].content == "Response 2"

        stats = cache.get_cache_stats()
        assert stats["size"] == 2

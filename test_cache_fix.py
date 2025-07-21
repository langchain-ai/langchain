#!/usr/bin/env python3
"""Simple test script to verify the cache fix works."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'libs', 'core'))

from typing import Optional, Any
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.messages import AIMessage


class InMemoryCache(BaseCache):
    """In-memory cache used for testing purposes."""

    def __init__(self) -> None:
        """Initialize with empty cache."""
        self._cache: dict[tuple[str, str], RETURN_VAL_TYPE] = {}

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        return self._cache.get((prompt, llm_string), None)

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        self._cache[prompt, llm_string] = return_val

    def clear(self, **kwargs: Any) -> None:
        """Clear cache."""
        self._cache = {}


def test_cache_fix():
    """Test that the cache fix works with Generation objects."""
    print("Testing cache fix...")
    
    cache = InMemoryCache()
    chat_model = FakeListChatModel(cache=cache, responses=["hello world"])
    
    # First call - normal operation 
    print("Making first call...")
    result1 = chat_model.invoke("test prompt")
    print(f"First result: {result1.content}")
    assert result1.content == "hello world"
    
    # Get the cache key and manually corrupt the cache by replacing 
    # ChatGeneration with Generation to simulate the bug scenario
    print("Corrupting cache with Generation objects...")
    cache_key = list(cache._cache.keys())[0]
    cached_chat_generations = cache._cache[cache_key]
    
    # Replace with Generation objects (missing message field)
    corrupted_generations = [
        Generation(
            text=gen.text,
            generation_info=gen.generation_info,
            type="Generation"  # This is the key - wrong type
        )
        for gen in cached_chat_generations
    ]
    cache._cache[cache_key] = corrupted_generations
    print(f"Cache corrupted. Generation types: {[type(g).__name__ for g in corrupted_generations]}")
    
    # Second call should handle the Generation objects gracefully
    # instead of throwing a validation error
    print("Making second call (should handle Generation objects)...")
    try:
        result2 = chat_model.invoke("test prompt")
        print(f"Second result: {result2.content}")
        assert result2.content == "hello world"
        print("✓ Test PASSED - Cache fix works!")
        return True
    except Exception as e:
        print(f"✗ Test FAILED - Cache fix did not work: {e}")
        return False


if __name__ == "__main__":
    success = test_cache_fix()
    sys.exit(0 if success else 1)
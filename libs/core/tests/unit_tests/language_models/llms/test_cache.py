from typing import Any

from typing_extensions import override

from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.globals import set_llm_cache
from langchain_core.language_models import FakeListLLM


class InMemoryCache(BaseCache):
    """In-memory cache used for testing purposes."""

    def __init__(self) -> None:
        """Initialize with empty cache."""
        self._cache: dict[tuple[str, str], RETURN_VAL_TYPE] = {}

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        """Look up based on `prompt` and `llm_string`."""
        return self._cache.get((prompt, llm_string), None)

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on `prompt` and `llm_string`."""
        self._cache[prompt, llm_string] = return_val

    @override
    def clear(self, **kwargs: Any) -> None:
        """Clear cache."""
        self._cache = {}


async def test_local_cache_generate_async() -> None:
    global_cache = InMemoryCache()
    local_cache = InMemoryCache()
    try:
        set_llm_cache(global_cache)
        llm = FakeListLLM(cache=local_cache, responses=["foo", "bar"])
        output = await llm.agenerate(["foo"])
        assert output.generations[0][0].text == "foo"
        output = await llm.agenerate(["foo"])
        assert output.generations[0][0].text == "foo"
        assert global_cache._cache == {}
        assert len(local_cache._cache) == 1
    finally:
        set_llm_cache(None)


def test_local_cache_generate_sync() -> None:
    global_cache = InMemoryCache()
    local_cache = InMemoryCache()
    try:
        set_llm_cache(global_cache)
        llm = FakeListLLM(cache=local_cache, responses=["foo", "bar"])
        output = llm.generate(["foo"])
        assert output.generations[0][0].text == "foo"
        output = llm.generate(["foo"])
        assert output.generations[0][0].text == "foo"
        assert global_cache._cache == {}
        assert len(local_cache._cache) == 1
    finally:
        set_llm_cache(None)


class InMemoryCacheBad(BaseCache):
    """In-memory cache used for testing purposes."""

    def __init__(self) -> None:
        """Initialize with empty cache."""
        self._cache: dict[tuple[str, str], RETURN_VAL_TYPE] = {}

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        """Look up based on `prompt` and `llm_string`."""
        msg = "This code should not be triggered"
        raise NotImplementedError(msg)

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on `prompt` and `llm_string`."""
        msg = "This code should not be triggered"
        raise NotImplementedError(msg)

    @override
    def clear(self, **kwargs: Any) -> None:
        """Clear cache."""
        self._cache = {}


def test_no_cache_generate_sync() -> None:
    global_cache = InMemoryCacheBad()
    try:
        set_llm_cache(global_cache)
        llm = FakeListLLM(cache=False, responses=["foo", "bar"])
        output = llm.generate(["foo"])
        assert output.generations[0][0].text == "foo"
        output = llm.generate(["foo"])
        assert output.generations[0][0].text == "bar"
        assert global_cache._cache == {}
    finally:
        set_llm_cache(None)


async def test_no_cache_generate_async() -> None:
    global_cache = InMemoryCacheBad()
    try:
        set_llm_cache(global_cache)
        llm = FakeListLLM(cache=False, responses=["foo", "bar"])
        output = await llm.agenerate(["foo"])
        assert output.generations[0][0].text == "foo"
        output = await llm.agenerate(["foo"])
        assert output.generations[0][0].text == "bar"
        assert global_cache._cache == {}
    finally:
        set_llm_cache(None)


def test_fine_grained_caching_overlapping_prompts() -> None:
    """Test that fine-grained caching works with overlapping prompt sets.

    This demonstrates that caching is now done at the prompt level,
    not the batch level. If you generate prompts [a, b], then [a, c],
    prompt 'a' will be retrieved from cache.
    """
    local_cache = InMemoryCache()
    try:
        # Create an LLM with specific responses
        llm = FakeListLLM(
            cache=local_cache,
            responses=["response_a", "response_b", "response_c"]
        )

        # First call: generate for prompts "a" and "b"
        output1 = llm.generate(["a", "b"])
        assert output1.generations[0][0].text == "response_a"
        assert output1.generations[1][0].text == "response_b"

        # Cache should now have 2 entries
        assert len(local_cache._cache) == 2

        # Second call: generate for prompts "a" and "c"
        # "a" should be from cache (not consuming another response)
        # "c" should be newly generated
        output2 = llm.generate(["a", "c"])
        assert output2.generations[0][0].text == "response_a"  # From cache
        assert output2.generations[1][0].text == "response_c"  # New generation

        # Cache should now have 3 entries (a, b, c)
        assert len(local_cache._cache) == 3
    finally:
        pass


async def test_fine_grained_caching_overlapping_prompts_async() -> None:
    """Async version of fine-grained caching test."""
    local_cache = InMemoryCache()
    try:
        llm = FakeListLLM(
            cache=local_cache,
            responses=["response_a", "response_b", "response_c"]
        )

        # First call: generate for prompts "a" and "b"
        output1 = await llm.agenerate(["a", "b"])
        assert output1.generations[0][0].text == "response_a"
        assert output1.generations[1][0].text == "response_b"
        assert len(local_cache._cache) == 2

        # Second call: generate for prompts "a" and "c"
        output2 = await llm.agenerate(["a", "c"])
        assert output2.generations[0][0].text == "response_a"  # From cache
        assert output2.generations[1][0].text == "response_c"  # New generation
        assert len(local_cache._cache) == 3
    finally:
        pass


def test_fine_grained_caching_all_cached() -> None:
    """Test that all cached prompts returns without generating."""
    local_cache = InMemoryCache()
    try:
        llm = FakeListLLM(
            cache=local_cache,
            responses=["response_a", "response_b", "response_c"]
        )

        # First call: cache prompts "a" and "b"
        output1 = llm.generate(["a", "b"])
        assert output1.generations[0][0].text == "response_a"
        assert output1.generations[1][0].text == "response_b"
        assert len(local_cache._cache) == 2

        # Second call: request same prompts in different order
        # Both should be from cache, so call "c" should not be used
        output2 = llm.generate(["b", "a"])
        assert output2.generations[0][0].text == "response_b"  # From cache
        assert output2.generations[1][0].text == "response_a"  # From cache

        # Third response should still be available (not consumed)
        assert len(local_cache._cache) == 2
        output3 = llm.generate(["c"])
        assert output3.generations[0][0].text == "response_c"  # New generation
    finally:
        pass

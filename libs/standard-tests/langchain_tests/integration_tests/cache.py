"""
Standard tests for the BaseCache abstraction

We don't recommend implementing externally managed BaseCache abstractions at this time.

:private:
"""

from abc import abstractmethod

import pytest
from langchain_core.caches import BaseCache
from langchain_core.outputs import Generation

from langchain_tests.base import BaseStandardTests


class SyncCacheTestSuite(BaseStandardTests):
    """Test suite for checking the BaseCache API of a caching layer for LLMs.

    This test suite verifies the basic caching API of a caching layer for LLMs.

    The test suite is designed for synchronous caching layers.

    Implementers should subclass this test suite and provide a fixture
    that returns an empty cache for each test.
    """

    @abstractmethod
    @pytest.fixture
    def cache(self) -> BaseCache:
        """Get the cache class to test.

        The returned cache should be EMPTY.
        """

    def get_sample_prompt(self) -> str:
        """Return a sample prompt for testing."""
        return "Sample prompt for testing."

    def get_sample_llm_string(self) -> str:
        """Return a sample LLM string for testing."""
        return "Sample LLM string configuration."

    def get_sample_generation(self) -> Generation:
        """Return a sample Generation object for testing."""
        return Generation(
            text="Sample generated text.", generation_info={"reason": "test"}
        )

    def test_cache_is_empty(self, cache: BaseCache) -> None:
        """Test that the cache is empty."""
        assert (
            cache.lookup(self.get_sample_prompt(), self.get_sample_llm_string()) is None
        )

    def test_update_cache(self, cache: BaseCache) -> None:
        """Test updating the cache."""
        prompt = self.get_sample_prompt()
        llm_string = self.get_sample_llm_string()
        generation = self.get_sample_generation()
        cache.update(prompt, llm_string, [generation])
        assert cache.lookup(prompt, llm_string) == [generation]

    def test_cache_still_empty(self, cache: BaseCache) -> None:
        """This test should follow a test that updates the cache.

        This just verifies that the fixture is set up properly to be empty
        after each test.
        """
        assert (
            cache.lookup(self.get_sample_prompt(), self.get_sample_llm_string()) is None
        )

    def test_clear_cache(self, cache: BaseCache) -> None:
        """Test clearing the cache."""
        prompt = self.get_sample_prompt()
        llm_string = self.get_sample_llm_string()
        generation = self.get_sample_generation()
        cache.update(prompt, llm_string, [generation])
        cache.clear()
        assert cache.lookup(prompt, llm_string) is None

    def test_cache_miss(self, cache: BaseCache) -> None:
        """Test cache miss."""
        assert cache.lookup("Nonexistent prompt", self.get_sample_llm_string()) is None

    def test_cache_hit(self, cache: BaseCache) -> None:
        """Test cache hit."""
        prompt = self.get_sample_prompt()
        llm_string = self.get_sample_llm_string()
        generation = self.get_sample_generation()
        cache.update(prompt, llm_string, [generation])
        assert cache.lookup(prompt, llm_string) == [generation]

    def test_update_cache_with_multiple_generations(self, cache: BaseCache) -> None:
        """Test updating the cache with multiple Generation objects."""
        prompt = self.get_sample_prompt()
        llm_string = self.get_sample_llm_string()
        generations = [
            self.get_sample_generation(),
            Generation(text="Another generated text."),
        ]
        cache.update(prompt, llm_string, generations)
        assert cache.lookup(prompt, llm_string) == generations


class AsyncCacheTestSuite(BaseStandardTests):
    """Test suite for checking the BaseCache API of a caching layer for LLMs.

    This test suite verifies the basic caching API of a caching layer for LLMs.

    The test suite is designed for synchronous caching layers.

    Implementers should subclass this test suite and provide a fixture
    that returns an empty cache for each test.
    """

    @abstractmethod
    @pytest.fixture
    async def cache(self) -> BaseCache:
        """Get the cache class to test.

        The returned cache should be EMPTY.
        """

    def get_sample_prompt(self) -> str:
        """Return a sample prompt for testing."""
        return "Sample prompt for testing."

    def get_sample_llm_string(self) -> str:
        """Return a sample LLM string for testing."""
        return "Sample LLM string configuration."

    def get_sample_generation(self) -> Generation:
        """Return a sample Generation object for testing."""
        return Generation(
            text="Sample generated text.", generation_info={"reason": "test"}
        )

    async def test_cache_is_empty(self, cache: BaseCache) -> None:
        """Test that the cache is empty."""
        assert (
            await cache.alookup(self.get_sample_prompt(), self.get_sample_llm_string())
            is None
        )

    async def test_update_cache(self, cache: BaseCache) -> None:
        """Test updating the cache."""
        prompt = self.get_sample_prompt()
        llm_string = self.get_sample_llm_string()
        generation = self.get_sample_generation()
        await cache.aupdate(prompt, llm_string, [generation])
        assert await cache.alookup(prompt, llm_string) == [generation]

    async def test_cache_still_empty(self, cache: BaseCache) -> None:
        """This test should follow a test that updates the cache.

        This just verifies that the fixture is set up properly to be empty
        after each test.
        """
        assert (
            await cache.alookup(self.get_sample_prompt(), self.get_sample_llm_string())
            is None
        )

    async def test_clear_cache(self, cache: BaseCache) -> None:
        """Test clearing the cache."""
        prompt = self.get_sample_prompt()
        llm_string = self.get_sample_llm_string()
        generation = self.get_sample_generation()
        await cache.aupdate(prompt, llm_string, [generation])
        await cache.aclear()
        assert await cache.alookup(prompt, llm_string) is None

    async def test_cache_miss(self, cache: BaseCache) -> None:
        """Test cache miss."""
        assert (
            await cache.alookup("Nonexistent prompt", self.get_sample_llm_string())
            is None
        )

    async def test_cache_hit(self, cache: BaseCache) -> None:
        """Test cache hit."""
        prompt = self.get_sample_prompt()
        llm_string = self.get_sample_llm_string()
        generation = self.get_sample_generation()
        await cache.aupdate(prompt, llm_string, [generation])
        assert await cache.alookup(prompt, llm_string) == [generation]

    async def test_update_cache_with_multiple_generations(
        self, cache: BaseCache
    ) -> None:
        """Test updating the cache with multiple Generation objects."""
        prompt = self.get_sample_prompt()
        llm_string = self.get_sample_llm_string()
        generations = [
            self.get_sample_generation(),
            Generation(text="Another generated text."),
        ]
        await cache.aupdate(prompt, llm_string, generations)
        assert await cache.alookup(prompt, llm_string) == generations

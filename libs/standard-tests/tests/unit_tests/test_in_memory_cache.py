import pytest
from langchain_core.caches import InMemoryCache

from langchain_standard_tests.integration_tests.cache import (
    AsyncCacheTestSuite,
    SyncCacheTestSuite,
)


class TestInMemoryCache(SyncCacheTestSuite):
    @pytest.fixture
    def cache(self) -> InMemoryCache:
        return InMemoryCache()


class TestInMemoryCacheAsync(AsyncCacheTestSuite):
    @pytest.fixture
    async def cache(self) -> InMemoryCache:
        return InMemoryCache()

import pytest
from langchain_core.caches import InMemoryCache
from typing_extensions import override

from langchain_tests.integration_tests.cache import (
    AsyncCacheTestSuite,
    SyncCacheTestSuite,
)


class TestInMemoryCache(SyncCacheTestSuite):
    @pytest.fixture
    @override
    def cache(self) -> InMemoryCache:
        return InMemoryCache()


class TestInMemoryCacheAsync(AsyncCacheTestSuite):
    @pytest.fixture
    @override
    async def cache(self) -> InMemoryCache:
        return InMemoryCache()

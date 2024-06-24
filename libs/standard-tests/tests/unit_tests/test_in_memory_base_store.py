"""Tests for the InMemoryStore class."""
import pytest
from langchain_core.stores import InMemoryStore

from langchain_standard_tests.integration_tests.base_store import (
    BaseStoreAsyncTests,
    BaseStoreSyncTests,
)


class TestInMemoryStore(BaseStoreSyncTests):
    @pytest.fixture
    def kv_store(self) -> InMemoryStore:
        return InMemoryStore()


class TestInMemoryStoreAsync(BaseStoreAsyncTests):
    @pytest.fixture
    async def kv_store(self) -> InMemoryStore:
        return InMemoryStore()

"""Tests for the InMemoryStore class."""

import pytest
from langchain_core.stores import InMemoryStore
from typing_extensions import override

from langchain_tests.integration_tests.base_store import (
    BaseStoreAsyncTests,
    BaseStoreSyncTests,
)


class TestInMemoryStore(BaseStoreSyncTests[str]):
    @pytest.fixture
    @override
    def three_values(self) -> tuple[str, str, str]:
        return "foo", "bar", "buzz"

    @pytest.fixture
    @override
    def kv_store(self) -> InMemoryStore:
        return InMemoryStore()


class TestInMemoryStoreAsync(BaseStoreAsyncTests[str]):
    @pytest.fixture
    @override
    def three_values(self) -> tuple[str, str, str]:
        return "foo", "bar", "buzz"

    @pytest.fixture
    @override
    async def kv_store(self) -> InMemoryStore:
        return InMemoryStore()

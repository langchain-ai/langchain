"""Tests for the InMemoryStore class."""

from typing import Tuple

import pytest
from langchain_core.stores import InMemoryStore

from langchain_tests.integration_tests.base_store import (
    BaseStoreAsyncTests,
    BaseStoreSyncTests,
)


class TestInMemoryStore(BaseStoreSyncTests):
    @pytest.fixture
    def three_values(self) -> Tuple[str, str, str]:
        return "foo", "bar", "buzz"

    @pytest.fixture
    def kv_store(self) -> InMemoryStore:
        return InMemoryStore()


class TestInMemoryStoreAsync(BaseStoreAsyncTests):
    @pytest.fixture
    def three_values(self) -> Tuple[str, str, str]:  # type: ignore
        return "foo", "bar", "buzz"

    @pytest.fixture
    async def kv_store(self) -> InMemoryStore:
        return InMemoryStore()

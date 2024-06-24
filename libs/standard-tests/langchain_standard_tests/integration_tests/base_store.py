from abc import ABC, abstractmethod
from typing import AsyncGenerator, Generator, TypeVar

import pytest
from langchain_core.stores import BaseStore

K = TypeVar("K")
V = TypeVar("V")


class BaseStoreSyncTests(ABC):
    """Test suite for checking the key-value API of a BaseStore.

    This test suite verifies the basic key-value API of a BaseStore.

    The test suite is designed for synchronous key-value stores.

    Implementers should subclass this test suite and provide a fixture
    that returns an empty key-value store for each test.
    """

    @abstractmethod
    @pytest.fixture
    def kv_store(self) -> BaseStore[str, str]:
        """Get the key-value store class to test.

        The returned key-value store should be EMPTY.
        """

    def test_kv_store_is_empty(self, kv_store: BaseStore[str, str]) -> None:
        """Test that the key-value store is empty."""
        keys = ["foo"]
        assert kv_store.mget(keys) == [None]

    def test_set_and_get_values(self, kv_store: BaseStore[str, str]) -> None:
        """Test setting and getting values in the key-value store."""
        key_value_pairs = [("foo", "bar"), ("baz", "qux")]
        kv_store.mset(key_value_pairs)
        assert kv_store.mget(["foo", "baz"]) == ["bar", "qux"]

    def test_store_still_empty(self, kv_store: BaseStore[str, str]) -> None:
        """This test should follow a test that sets values.

        This just verifies that the fixture is set up properly to be empty
        after each test.
        """
        keys = ["foo"]
        assert kv_store.mget(keys) == [None]

    def test_delete_values(self, kv_store: BaseStore[str, str]) -> None:
        """Test deleting values from the key-value store."""
        key_value_pairs = [("foo", "bar"), ("baz", "qux")]
        kv_store.mset(key_value_pairs)
        kv_store.mdelete(["foo"])
        assert kv_store.mget(["foo", "baz"]) == [None, "qux"]

    def test_delete_bulk_values(self, kv_store: BaseStore[str, str]) -> None:
        """Test that we can delete several values at once."""
        key_value_pairs = [("foo", "bar"), ("baz", "qux"), ("quux", "corge")]
        kv_store.mset(key_value_pairs)
        kv_store.mdelete(["foo", "baz"])
        assert kv_store.mget(["foo", "baz", "quux"]) == [None, None, "corge"]

    def test_delete_missing_keys(self, kv_store: BaseStore[str, str]) -> None:
        """Deleting missing keys should not raise an exception."""
        kv_store.mdelete(["foo"])
        kv_store.mdelete(["foo", "bar", "baz"])

    def test_set_values_is_idempotent(self, kv_store: BaseStore[str, str]) -> None:
        """Setting values by key should be idempotent."""
        key_value_pairs = [("foo", "bar"), ("baz", "qux")]
        kv_store.mset(key_value_pairs)
        kv_store.mset(key_value_pairs)
        assert kv_store.mget(["foo", "baz"]) == ["bar", "qux"]

    def test_get_can_get_same_value(self, kv_store: BaseStore[str, str]) -> None:
        """Test that the same value can be retrieved multiple times."""
        key_value_pairs = [("foo", "bar"), ("baz", "qux")]
        kv_store.mset(key_value_pairs)
        # This test assumes kv_store does not handle duplicates by default
        assert kv_store.mget(["foo", "baz", "foo", "baz"]) == [
            "bar",
            "qux",
            "bar",
            "qux",
        ]

    def test_overwrite_values_by_key(self, kv_store: BaseStore[str, str]) -> None:
        """Test that we can overwrite values by key using mset."""
        key_value_pairs = [("foo", "bar"), ("baz", "qux")]
        kv_store.mset(key_value_pairs)

        # Now overwrite value of key "foo"
        new_key_value_pairs = [("foo", "new_bar")]
        kv_store.mset(new_key_value_pairs)

        # Check that the value has been updated
        assert kv_store.mget(["foo", "baz"]) == ["new_bar", "qux"]

    def test_yield_keys(self, kv_store: BaseStore[str, str]):
        """Test that we can yield keys from the store."""
        key_value_pairs = [("foo", "bar"), ("baz", "qux")]
        kv_store.mset(key_value_pairs)

        generator = kv_store.yield_keys()
        assert isinstance(generator, Generator)

        assert sorted(kv_store.yield_keys()) == ["baz", "foo"]
        assert sorted(kv_store.yield_keys(prefix="foo")) == ["foo"]


class BaseStoreAsyncTests(ABC):
    """Test suite for checking the key-value API of a BaseStore.

    This test suite verifies the basic key-value API of a BaseStore.

    The test suite is designed for synchronous key-value stores.

    Implementers should subclass this test suite and provide a fixture
    that returns an empty key-value store for each test.
    """

    @abstractmethod
    @pytest.fixture
    async def kv_store(self) -> BaseStore[str, str]:
        """Get the key-value store class to test.

        The returned key-value store should be EMPTY.
        """

    async def test_kv_store_is_empty(self, kv_store: BaseStore[str, str]) -> None:
        """Test that the key-value store is empty."""
        keys = ["foo"]
        assert await kv_store.amget(keys) == [None]

    async def test_set_and_get_values(self, kv_store: BaseStore[str, str]) -> None:
        """Test setting and getting values in the key-value store."""
        key_value_pairs = [("foo", "bar"), ("baz", "qux")]
        await kv_store.amset(key_value_pairs)
        assert await kv_store.amget(["foo", "baz"]) == ["bar", "qux"]

    async def test_store_still_empty(self, kv_store: BaseStore[str, str]) -> None:
        """This test should follow a test that sets values.

        This just verifies that the fixture is set up properly to be empty
        after each test.
        """
        keys = ["foo"]
        assert await kv_store.amget(keys) == [None]

    async def test_delete_values(self, kv_store: BaseStore[str, str]) -> None:
        """Test deleting values from the key-value store."""
        key_value_pairs = [("foo", "bar"), ("baz", "qux")]
        await kv_store.amset(key_value_pairs)
        kv_store.mdelete(["foo"])
        assert await kv_store.amget(["foo", "baz"]) == [None, "qux"]

    async def test_delete_bulk_values(self, kv_store: BaseStore[str, str]) -> None:
        """Test that we can delete several values at once."""
        key_value_pairs = [("foo", "bar"), ("baz", "qux"), ("quux", "corge")]
        await kv_store.amset(key_value_pairs)
        kv_store.mdelete(["foo", "baz"])
        assert await kv_store.amget(["foo", "baz", "quux"]) == [None, None, "corge"]

    async def test_delete_missing_keys(self, kv_store: BaseStore[str, str]) -> None:
        """Deleting missing keys should not raise an exception."""
        kv_store.mdelete(["foo"])
        kv_store.mdelete(["foo", "bar", "baz"])

    async def test_set_values_is_idempotent(
        self, kv_store: BaseStore[str, str]
    ) -> None:
        """Setting values by key should be idempotent."""
        key_value_pairs = [("foo", "bar"), ("baz", "qux")]
        await kv_store.amset(key_value_pairs)
        await kv_store.amset(key_value_pairs)
        assert await kv_store.amget(["foo", "baz"]) == ["bar", "qux"]

    async def test_get_can_get_same_value(self, kv_store: BaseStore[str, str]) -> None:
        """Test that the same value can be retrieved multiple times."""
        key_value_pairs = [("foo", "bar"), ("baz", "qux")]
        await kv_store.amset(key_value_pairs)
        # This test assumes kv_store does not handle duplicates by async default
        assert await kv_store.amget(["foo", "baz", "foo", "baz"]) == [
            "bar",
            "qux",
            "bar",
            "qux",
        ]

    async def test_overwrite_values_by_key(self, kv_store: BaseStore[str, str]) -> None:
        """Test that we can overwrite values by key using mset."""
        key_value_pairs = [("foo", "bar"), ("baz", "qux")]
        await kv_store.amset(key_value_pairs)

        # Now overwrite value of key "foo"
        new_key_value_pairs = [("foo", "new_bar")]
        await kv_store.amset(new_key_value_pairs)

        # Check that the value has been updated
        assert await kv_store.amget(["foo", "baz"]) == ["new_bar", "qux"]

    async def test_yield_keys(self, kv_store: BaseStore[str, str]):
        """Test that we can yield keys from the store."""
        key_value_pairs = [("foo", "bar"), ("baz", "qux")]
        await kv_store.amset(key_value_pairs)

        generator = kv_store.ayield_keys()
        assert isinstance(generator, AsyncGenerator)

        assert sorted([key async for key in kv_store.ayield_keys()]) == ["baz", "foo"]
        assert sorted([key async for key in kv_store.ayield_keys(prefix="foo")]) == [
            "foo"
        ]

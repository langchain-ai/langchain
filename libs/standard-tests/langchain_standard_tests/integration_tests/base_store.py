from abc import ABC, abstractmethod
from typing import TypeVar

import pytest
from langchain_core.stores import BaseStore

K = TypeVar("K")
V = TypeVar("V")

class KeyValueTestSuite(ABC):
    """Test suite for checking the key-value API of a BaseStore.

    This test suite verifies the basic key-value API of a BaseStore.

    The test suite is designed for synchronous key-value stores.

    Implementers should subclass this test suite and provide a fixture
    that returns an empty key-value store for each test.
    """

    @abstractmethod
    @pytest.fixture
    def kv_store(self) -> BaseStore[K, V]:
        """Get the key-value store class to test.

        The returned key-value store should be EMPTY.
        """

    def test_kv_store_is_empty(self, kv_store: BaseStore[K, V]) -> None:
        """Test that the key-value store is empty."""
        keys = ["foo"]
        assert kv_store.mget(keys) == [None]

    def test_set_and_get_values(self, kv_store: BaseStore[K, V]) -> None:
        """Test setting and getting values in the key-value store."""
        key_value_pairs = [("foo", "bar"), ("baz", "qux")]
        kv_store.mset(key_value_pairs)
        assert kv_store.mget(["foo", "baz"]) == ["bar", "qux"]

    def test_store_still_empty(self, kv_store: BaseStore[K, V]) -> None:
        """This test should follow a test that sets values.

        This just verifies that the fixture is set up properly to be empty
        after each test.
        """
        keys = ["foo"]
        assert kv_store.mget(keys) == [None]

    def test_delete_values(self, kv_store: BaseStore[K, V]) -> None:
        """Test deleting values from the key-value store."""
        key_value_pairs = [("foo", "bar"), ("baz", "qux")]
        kv_store.mset(key_value_pairs)
        kv_store.mdelete(["foo"])
        assert kv_store.mget(["foo", "baz"]) == [None, "qux"]

    def test_delete_bulk_values(self, kv_store: BaseStore[K, V]) -> None:
        """Test that we can delete several values at once."""
        key_value_pairs = [("foo", "bar"), ("baz", "qux"), ("quux", "corge")]
        kv_store.mset(key_value_pairs)
        kv_store.mdelete(["foo", "baz"])
        assert kv_store.mget(["foo", "baz", "quux"]) == [None, None, "corge"]

    def test_delete_missing_keys(self, kv_store: BaseStore[K, V]) -> None:
        """Deleting missing keys should not raise an exception."""
        kv_store.mdelete(["foo"])
        kv_store.mdelete(["foo", "bar", "baz"])

    def test_set_values_is_idempotent(self, kv_store: BaseStore[K, V]) -> None:
        """Setting values by key should be idempotent."""
        key_value_pairs = [("foo", "bar"), ("baz", "qux")]
        kv_store.mset(key_value_pairs)
        kv_store.mset(key_value_pairs)
        assert kv_store.mget(["foo", "baz"]) == ["bar", "qux"]

    def test_duplicate_keys_without_ids(self, kv_store: BaseStore[K, V]) -> None:
        """Setting values without unique keys should duplicate content."""
        key_value_pairs = [("foo", "bar"), ("baz", "qux")]
        kv_store.mset(key_value_pairs)
        kv_store.mset(key_value_pairs)
        # This test assumes kv_store does not handle duplicates by default
        assert kv_store.mget(["foo", "baz", "foo", "baz"]) == ["bar", "qux", "bar", "qux"]

    def test_overwrite_values_by_key(self, kv_store: BaseStore[K, V]) -> None:
        """Test that we can overwrite values by key using mset."""
        key_value_pairs = [("foo", "bar"), ("baz", "qux")]
        kv_store.mset(key_value_pairs)

        # Now overwrite value of key "foo"
        new_key_value_pairs = [("foo", "new_bar")]
        kv_store.mset(new_key_value_pairs)

        # Check that the value has been updated
        assert kv_store.mget(["foo", "baz"]) == ["new_bar", "qux"]

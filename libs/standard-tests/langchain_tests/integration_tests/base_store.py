"""Standard tests for the `BaseStore` abstraction.

We don't recommend implementing externally managed `BaseStore` abstractions at this
time.
"""

from abc import abstractmethod
from collections.abc import AsyncGenerator, Generator
from typing import Generic, TypeVar

import pytest
from langchain_core.stores import BaseStore

from langchain_tests.base import BaseStandardTests

V = TypeVar("V")


class BaseStoreSyncTests(BaseStandardTests, Generic[V]):
    """Test suite for checking the key-value API of a `BaseStore`.

    This test suite verifies the basic key-value API of a `BaseStore`.

    The test suite is designed for synchronous key-value stores.

    Implementers should subclass this test suite and provide a fixture
    that returns an empty key-value store for each test.
    """

    @abstractmethod
    @pytest.fixture
    def kv_store(self) -> BaseStore[str, V]:
        """Get the key-value store class to test.

        The returned key-value store should be EMPTY.
        """

    @abstractmethod
    @pytest.fixture
    def three_values(self) -> tuple[V, V, V]:
        """Three example values that will be used in the tests."""

    def test_three_values(self, three_values: tuple[V, V, V]) -> None:
        """Test that the fixture provides three values."""
        assert isinstance(three_values, tuple)
        assert len(three_values) == 3

    def test_kv_store_is_empty(self, kv_store: BaseStore[str, V]) -> None:
        """Test that the key-value store is empty."""
        keys = ["foo", "bar", "buzz"]
        assert kv_store.mget(keys) == [None, None, None]

    def test_set_and_get_values(
        self,
        kv_store: BaseStore[str, V],
        three_values: tuple[V, V, V],
    ) -> None:
        """Test setting and getting values in the key-value store."""
        foo = three_values[0]
        bar = three_values[1]
        key_value_pairs = [("foo", foo), ("bar", bar)]
        kv_store.mset(key_value_pairs)
        assert kv_store.mget(["foo", "bar"]) == [foo, bar]

    def test_store_still_empty(self, kv_store: BaseStore[str, V]) -> None:
        """Test that the store is still empty.

        This test should follow a test that sets values.

        This just verifies that the fixture is set up properly to be empty
        after each test.
        """
        keys = ["foo"]
        assert kv_store.mget(keys) == [None]

    def test_delete_values(
        self,
        kv_store: BaseStore[str, V],
        three_values: tuple[V, V, V],
    ) -> None:
        """Test deleting values from the key-value store."""
        foo = three_values[0]
        bar = three_values[1]
        key_value_pairs = [("foo", foo), ("bar", bar)]
        kv_store.mset(key_value_pairs)
        kv_store.mdelete(["foo"])
        assert kv_store.mget(["foo", "bar"]) == [None, bar]

    def test_delete_bulk_values(
        self,
        kv_store: BaseStore[str, V],
        three_values: tuple[V, V, V],
    ) -> None:
        """Test that we can delete several values at once."""
        foo, bar, buz = three_values
        key_values = [("foo", foo), ("bar", bar), ("buz", buz)]
        kv_store.mset(key_values)
        kv_store.mdelete(["foo", "buz"])
        assert kv_store.mget(["foo", "bar", "buz"]) == [None, bar, None]

    def test_delete_missing_keys(self, kv_store: BaseStore[str, V]) -> None:
        """Deleting missing keys should not raise an exception."""
        kv_store.mdelete(["foo"])
        kv_store.mdelete(["foo", "bar", "baz"])

    def test_set_values_is_idempotent(
        self,
        kv_store: BaseStore[str, V],
        three_values: tuple[V, V, V],
    ) -> None:
        """Setting values by key should be idempotent."""
        foo, bar, _ = three_values
        key_value_pairs = [("foo", foo), ("bar", bar)]
        kv_store.mset(key_value_pairs)
        kv_store.mset(key_value_pairs)
        assert kv_store.mget(["foo", "bar"]) == [foo, bar]
        assert sorted(kv_store.yield_keys()) == ["bar", "foo"]

    def test_get_can_get_same_value(
        self,
        kv_store: BaseStore[str, V],
        three_values: tuple[V, V, V],
    ) -> None:
        """Test that the same value can be retrieved multiple times."""
        foo, bar, _ = three_values
        key_value_pairs = [("foo", foo), ("bar", bar)]
        kv_store.mset(key_value_pairs)
        # This test assumes kv_store does not handle duplicates by default
        assert kv_store.mget(["foo", "bar", "foo", "bar"]) == [foo, bar, foo, bar]

    def test_overwrite_values_by_key(
        self,
        kv_store: BaseStore[str, V],
        three_values: tuple[V, V, V],
    ) -> None:
        """Test that we can overwrite values by key using mset."""
        foo, bar, buzz = three_values
        key_value_pairs = [("foo", foo), ("bar", bar)]
        kv_store.mset(key_value_pairs)

        # Now overwrite value of key "foo"
        new_key_value_pairs = [("foo", buzz)]
        kv_store.mset(new_key_value_pairs)

        # Check that the value has been updated
        assert kv_store.mget(["foo", "bar"]) == [buzz, bar]

    def test_yield_keys(
        self,
        kv_store: BaseStore[str, V],
        three_values: tuple[V, V, V],
    ) -> None:
        """Test that we can yield keys from the store."""
        foo, bar, _buzz = three_values
        key_value_pairs = [("foo", foo), ("bar", bar)]
        kv_store.mset(key_value_pairs)

        generator = kv_store.yield_keys()
        assert isinstance(generator, Generator)

        assert sorted(kv_store.yield_keys()) == ["bar", "foo"]
        assert sorted(kv_store.yield_keys(prefix="foo")) == ["foo"]


class BaseStoreAsyncTests(BaseStandardTests, Generic[V]):
    """Test suite for checking the key-value API of a `BaseStore`.

    This test suite verifies the basic key-value API of a `BaseStore`.

    The test suite is designed for synchronous key-value stores.

    Implementers should subclass this test suite and provide a fixture
    that returns an empty key-value store for each test.
    """

    @abstractmethod
    @pytest.fixture
    async def kv_store(self) -> BaseStore[str, V]:
        """Get the key-value store class to test.

        The returned key-value store should be EMPTY.
        """

    @abstractmethod
    @pytest.fixture
    def three_values(self) -> tuple[V, V, V]:
        """Three example values that will be used in the tests."""

    async def test_three_values(self, three_values: tuple[V, V, V]) -> None:
        """Test that the fixture provides three values."""
        assert isinstance(three_values, tuple)
        assert len(three_values) == 3

    async def test_kv_store_is_empty(self, kv_store: BaseStore[str, V]) -> None:
        """Test that the key-value store is empty."""
        keys = ["foo", "bar", "buzz"]
        assert await kv_store.amget(keys) == [None, None, None]

    async def test_set_and_get_values(
        self,
        kv_store: BaseStore[str, V],
        three_values: tuple[V, V, V],
    ) -> None:
        """Test setting and getting values in the key-value store."""
        foo = three_values[0]
        bar = three_values[1]
        key_value_pairs = [("foo", foo), ("bar", bar)]
        await kv_store.amset(key_value_pairs)
        assert await kv_store.amget(["foo", "bar"]) == [foo, bar]

    async def test_store_still_empty(self, kv_store: BaseStore[str, V]) -> None:
        """Test that the store is still empty.

        This test should follow a test that sets values.

        This just verifies that the fixture is set up properly to be empty
        after each test.
        """
        keys = ["foo"]
        assert await kv_store.amget(keys) == [None]

    async def test_delete_values(
        self,
        kv_store: BaseStore[str, V],
        three_values: tuple[V, V, V],
    ) -> None:
        """Test deleting values from the key-value store."""
        foo = three_values[0]
        bar = three_values[1]
        key_value_pairs = [("foo", foo), ("bar", bar)]
        await kv_store.amset(key_value_pairs)
        await kv_store.amdelete(["foo"])
        assert await kv_store.amget(["foo", "bar"]) == [None, bar]

    async def test_delete_bulk_values(
        self,
        kv_store: BaseStore[str, V],
        three_values: tuple[V, V, V],
    ) -> None:
        """Test that we can delete several values at once."""
        foo, bar, buz = three_values
        key_values = [("foo", foo), ("bar", bar), ("buz", buz)]
        await kv_store.amset(key_values)
        await kv_store.amdelete(["foo", "buz"])
        assert await kv_store.amget(["foo", "bar", "buz"]) == [None, bar, None]

    async def test_delete_missing_keys(self, kv_store: BaseStore[str, V]) -> None:
        """Deleting missing keys should not raise an exception."""
        await kv_store.amdelete(["foo"])
        await kv_store.amdelete(["foo", "bar", "baz"])

    async def test_set_values_is_idempotent(
        self,
        kv_store: BaseStore[str, V],
        three_values: tuple[V, V, V],
    ) -> None:
        """Setting values by key should be idempotent."""
        foo, bar, _ = three_values
        key_value_pairs = [("foo", foo), ("bar", bar)]
        await kv_store.amset(key_value_pairs)
        await kv_store.amset(key_value_pairs)
        assert await kv_store.amget(["foo", "bar"]) == [foo, bar]
        assert sorted([key async for key in kv_store.ayield_keys()]) == ["bar", "foo"]

    async def test_get_can_get_same_value(
        self,
        kv_store: BaseStore[str, V],
        three_values: tuple[V, V, V],
    ) -> None:
        """Test that the same value can be retrieved multiple times."""
        foo, bar, _ = three_values
        key_value_pairs = [("foo", foo), ("bar", bar)]
        await kv_store.amset(key_value_pairs)
        # This test assumes kv_store does not handle duplicates by async default
        assert await kv_store.amget(["foo", "bar", "foo", "bar"]) == [
            foo,
            bar,
            foo,
            bar,
        ]

    async def test_overwrite_values_by_key(
        self,
        kv_store: BaseStore[str, V],
        three_values: tuple[V, V, V],
    ) -> None:
        """Test that we can overwrite values by key using mset."""
        foo, bar, buzz = three_values
        key_value_pairs = [("foo", foo), ("bar", bar)]
        await kv_store.amset(key_value_pairs)

        # Now overwrite value of key "foo"
        new_key_value_pairs = [("foo", buzz)]
        await kv_store.amset(new_key_value_pairs)

        # Check that the value has been updated
        assert await kv_store.amget(["foo", "bar"]) == [buzz, bar]

    async def test_yield_keys(
        self,
        kv_store: BaseStore[str, V],
        three_values: tuple[V, V, V],
    ) -> None:
        """Test that we can yield keys from the store."""
        foo, bar, _buzz = three_values
        key_value_pairs = [("foo", foo), ("bar", bar)]
        await kv_store.amset(key_value_pairs)

        generator = kv_store.ayield_keys()
        assert isinstance(generator, AsyncGenerator)

        assert sorted([key async for key in kv_store.ayield_keys()]) == ["bar", "foo"]
        assert sorted([key async for key in kv_store.ayield_keys(prefix="foo")]) == [
            "foo",
        ]

import pytest
from langchain_tests.integration_tests.base_store import (
    BaseStoreAsyncTests,
    BaseStoreSyncTests,
)

from langchain_core.stores import InMemoryStore


# Check against standard tests
class TestSyncInMemoryStore(BaseStoreSyncTests):
    @pytest.fixture
    def kv_store(self) -> InMemoryStore:
        return InMemoryStore()

    @pytest.fixture
    def three_values(self) -> tuple[str, str, str]:  # type: ignore
        return "value1", "value2", "value3"


class TestAsyncInMemoryStore(BaseStoreAsyncTests):
    @pytest.fixture
    async def kv_store(self) -> InMemoryStore:
        return InMemoryStore()

    @pytest.fixture
    def three_values(self) -> tuple[str, str, str]:  # type: ignore
        return "value1", "value2", "value3"


def test_mget() -> None:
    store = InMemoryStore()
    store.mset([("key1", "value1"), ("key2", "value2")])

    values = store.mget(["key1", "key2"])
    assert values == ["value1", "value2"]

    # Test non-existent key
    non_existent_value = store.mget(["key3"])
    assert non_existent_value == [None]


async def test_amget() -> None:
    store = InMemoryStore()
    await store.amset([("key1", "value1"), ("key2", "value2")])

    values = await store.amget(["key1", "key2"])
    assert values == ["value1", "value2"]

    # Test non-existent key
    non_existent_value = await store.amget(["key3"])
    assert non_existent_value == [None]


def test_mset() -> None:
    store = InMemoryStore()
    store.mset([("key1", "value1"), ("key2", "value2")])

    values = store.mget(["key1", "key2"])
    assert values == ["value1", "value2"]


async def test_amset() -> None:
    store = InMemoryStore()
    await store.amset([("key1", "value1"), ("key2", "value2")])

    values = await store.amget(["key1", "key2"])
    assert values == ["value1", "value2"]


def test_mdelete() -> None:
    store = InMemoryStore()
    store.mset([("key1", "value1"), ("key2", "value2")])

    store.mdelete(["key1"])

    values = store.mget(["key1", "key2"])
    assert values == [None, "value2"]

    # Test deleting non-existent key
    store.mdelete(["key3"])  # No error should be raised


async def test_amdelete() -> None:
    store = InMemoryStore()
    await store.amset([("key1", "value1"), ("key2", "value2")])

    await store.amdelete(["key1"])

    values = await store.amget(["key1", "key2"])
    assert values == [None, "value2"]

    # Test deleting non-existent key
    await store.amdelete(["key3"])  # No error should be raised


def test_yield_keys() -> None:
    store = InMemoryStore()
    store.mset([("key1", "value1"), ("key2", "value2"), ("key3", "value3")])

    keys = list(store.yield_keys())
    assert set(keys) == {"key1", "key2", "key3"}

    keys_with_prefix = list(store.yield_keys(prefix="key"))
    assert set(keys_with_prefix) == {"key1", "key2", "key3"}

    keys_with_invalid_prefix = list(store.yield_keys(prefix="x"))
    assert keys_with_invalid_prefix == []


async def test_ayield_keys() -> None:
    store = InMemoryStore()
    await store.amset([("key1", "value1"), ("key2", "value2"), ("key3", "value3")])

    keys = [key async for key in store.ayield_keys()]
    assert set(keys) == {"key1", "key2", "key3"}

    keys_with_prefix = [key async for key in store.ayield_keys(prefix="key")]
    assert set(keys_with_prefix) == {"key1", "key2", "key3"}

    keys_with_invalid_prefix = [key async for key in store.ayield_keys(prefix="x")]
    assert keys_with_invalid_prefix == []

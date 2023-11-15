from langchain.storage.in_memory import InMemoryStore


def test_mget() -> None:
    store = InMemoryStore()
    store.mset([("key1", "value1"), ("key2", "value2")])

    values = store.mget(["key1", "key2"])
    assert values == ["value1", "value2"]

    # Test non-existent key
    non_existent_value = store.mget(["key3"])
    assert non_existent_value == [None]


def test_mset() -> None:
    store = InMemoryStore()
    store.mset([("key1", "value1"), ("key2", "value2")])

    values = store.mget(["key1", "key2"])
    assert values == ["value1", "value2"]


def test_mdelete() -> None:
    store = InMemoryStore()
    store.mset([("key1", "value1"), ("key2", "value2")])

    store.mdelete(["key1"])

    values = store.mget(["key1", "key2"])
    assert values == [None, "value2"]

    # Test deleting non-existent key
    store.mdelete(["key3"])  # No error should be raised


def test_yield_keys() -> None:
    store = InMemoryStore()
    store.mset([("key1", "value1"), ("key2", "value2"), ("key3", "value3")])

    keys = list(store.yield_keys())
    assert set(keys) == {"key1", "key2", "key3"}

    keys_with_prefix = list(store.yield_keys(prefix="key"))
    assert set(keys_with_prefix) == {"key1", "key2", "key3"}

    keys_with_invalid_prefix = list(store.yield_keys(prefix="x"))
    assert keys_with_invalid_prefix == []

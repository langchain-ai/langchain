"""Implement integration tests for Redis storage."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from langchain.storage.upstash_redis import UpstashRedisStore

if TYPE_CHECKING:
    from upstash_redis import Redis

pytest.importorskip("upstash_redis")

URL = "<UPSTASH_REDIS_REST_URL>"
TOKEN = "<UPSTASH_REDIS_REST_TOKEN>"


@pytest.fixture
def redis_client() -> Redis:
    """Yield redis client."""
    from upstash_redis import Redis

    # This fixture flushes the database!

    client = Redis(url=URL, token=TOKEN)
    try:
        client.ping()
    except Exception:
        pytest.skip("Ping request failed. Verify that credentials are correct.")

    client.flushdb()
    return client


def test_mget(redis_client: Redis) -> None:
    store = UpstashRedisStore(client=redis_client, ttl=None)
    keys = ["key1", "key2"]
    redis_client.mset({"key1": "value1", "key2": "value2"})
    result = store.mget(keys)
    assert result == ["value1", "value2"]


def test_mset(redis_client: Redis) -> None:
    store = UpstashRedisStore(client=redis_client, ttl=None)
    key_value_pairs = [("key1", "value1"), ("key2", "value2")]
    store.mset(key_value_pairs)
    result = redis_client.mget("key1", "key2")
    assert result == ["value1", "value2"]


def test_mdelete(redis_client: Redis) -> None:
    """Test that deletion works as expected."""
    store = UpstashRedisStore(client=redis_client, ttl=None)
    keys = ["key1", "key2"]
    redis_client.mset({"key1": "value1", "key2": "value2"})
    store.mdelete(keys)
    result = redis_client.mget(*keys)
    assert result == [None, None]


def test_yield_keys(redis_client: Redis) -> None:
    store = UpstashRedisStore(client=redis_client, ttl=None)
    redis_client.mset({"key1": "value2", "key2": "value2"})
    assert sorted(store.yield_keys()) == ["key1", "key2"]
    assert sorted(store.yield_keys(prefix="key*")) == ["key1", "key2"]
    assert sorted(store.yield_keys(prefix="lang*")) == []


def test_namespace(redis_client: Redis) -> None:
    store = UpstashRedisStore(client=redis_client, ttl=None, namespace="meow")
    key_value_pairs = [("key1", "value1"), ("key2", "value2")]
    store.mset(key_value_pairs)

    cursor, all_keys = redis_client.scan(0)
    while cursor != 0:
        cursor, keys = redis_client.scan(cursor)
        if len(keys) != 0:
            all_keys.extend(keys)

    assert sorted(all_keys) == [
        "meow/key1",
        "meow/key2",
    ]

    store.mdelete(["key1"])

    cursor, all_keys = redis_client.scan(0, match="*")
    while cursor != 0:
        cursor, keys = redis_client.scan(cursor, match="*")
        if len(keys) != 0:
            all_keys.extend(keys)

    assert sorted(all_keys) == [
        "meow/key2",
    ]

    assert list(store.yield_keys()) == ["key2"]
    assert list(store.yield_keys(prefix="key*")) == ["key2"]
    assert list(store.yield_keys(prefix="key1")) == []

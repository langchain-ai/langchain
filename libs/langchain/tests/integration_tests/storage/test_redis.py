"""Implement integration tests for Redis storage."""
import os
import typing
import uuid

import pytest

from langchain.storage.redis import RedisStore

if typing.TYPE_CHECKING:
    from redis import Redis

pytest.importorskip("redis")


@pytest.fixture
def redis_client() -> Redis:
    """Yield redis client."""
    import redis

    # Using standard port, but protecting against accidental data loss
    # by requiring a password.
    # This fixture flushes the database!
    # The only role of the password is to prevent users from accidentally
    # deleting their data.
    # The password should establish the identity of the server being.
    port = 6379
    password = os.environ.get("REDIS_PASSWORD") or str(uuid.uuid4())
    client = redis.Redis(host="localhost", port=port, password=password, db=0)
    try:
        client.ping()
    except redis.exceptions.ConnectionError:
        pytest.skip(
            "Redis server is not running or is not accessible. "
            "Verify that credentials are correct. "
        )
    # ATTENTION: This will delete all keys in the database!
    client.flushdb()
    return client


def test_mget(redis_client: Redis) -> None:
    """Test mget method."""
    store = RedisStore(client=redis_client, ttl=None)
    keys = ["key1", "key2"]
    redis_client.mset({"key1": b"value1", "key2": b"value2"})
    result = store.mget(keys)
    assert result == [b"value1", b"value2"]


def test_mset(redis_client: Redis) -> None:
    """Test that multiple keys can be set."""
    store = RedisStore(client=redis_client, ttl=None)
    key_value_pairs = [("key1", b"value1"), ("key2", b"value2")]
    store.mset(key_value_pairs)
    result = redis_client.mget(["key1", "key2"])
    assert result == [b"value1", b"value2"]


def test_mdelete(redis_client: Redis) -> None:
    """Test that deletion works as expected."""
    store = RedisStore(client=redis_client, ttl=None)
    keys = ["key1", "key2"]
    redis_client.mset({"key1": b"value1", "key2": b"value2"})
    store.mdelete(keys)
    result = redis_client.mget(keys)
    assert result == [None, None]


def test_yield_keys(redis_client: Redis) -> None:
    store = RedisStore(client=redis_client, ttl=None)
    redis_client.mset({"key1": b"value1", "key2": b"value2"})
    assert sorted(store.yield_keys()) == ["key1", "key2"]
    assert sorted(store.yield_keys(prefix="key*")) == ["key1", "key2"]
    assert sorted(store.yield_keys(prefix="lang*")) == []


def test_namespace(redis_client: Redis) -> None:
    """Test that a namespace is prepended to all keys properly."""
    store = RedisStore(client=redis_client, ttl=None, namespace="meow")
    key_value_pairs = [("key1", b"value1"), ("key2", b"value2")]
    store.mset(key_value_pairs)

    assert sorted(redis_client.scan_iter("*")) == [
        b"meow/key1",
        b"meow/key2",
    ]

    store.mdelete(["key1"])

    assert sorted(redis_client.scan_iter("*")) == [
        b"meow/key2",
    ]

    assert list(store.yield_keys()) == ["key2"]
    assert list(store.yield_keys(prefix="key*")) == ["key2"]
    assert list(store.yield_keys(prefix="key1")) == []

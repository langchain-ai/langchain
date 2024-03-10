"""**Storage** is an implementation of key-value store.

Storage module provides implementations of various key-value stores that conform
to a simple key-value interface.

The primary goal of these storages is to support caching.


**Class hierarchy:**

.. code-block::

    BaseStore --> <name>Store  # Examples: MongoDBStore, RedisStore

"""  # noqa: E501

from langchain_community.storage.astradb import (
    AstraDBByteStore,
    AstraDBStore,
)
from langchain_community.storage.mongodb import MongoDBStore
from langchain_community.storage.redis import RedisStore
from langchain_community.storage.upstash_redis import (
    UpstashRedisByteStore,
    UpstashRedisStore,
)

__all__ = [
    "AstraDBStore",
    "AstraDBByteStore",
    "MongoDBStore",
    "RedisStore",
    "UpstashRedisByteStore",
    "UpstashRedisStore",
]

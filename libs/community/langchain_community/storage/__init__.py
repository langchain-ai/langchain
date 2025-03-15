"""**Storage** is an implementation of key-value store.

Storage module provides implementations of various key-value stores that conform
to a simple key-value interface.

The primary goal of these storages is to support caching.


**Class hierarchy:**

.. code-block::

    BaseStore --> <name>Store  # Examples: MongoDBStore, RedisStore

"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_community.storage.astradb import (
        AstraDBByteStore,
        AstraDBStore,
    )
    from langchain_community.storage.cassandra import (
        CassandraByteStore,
    )
    from langchain_community.storage.mongodb import MongoDBByteStore, MongoDBStore
    from langchain_community.storage.redis import (
        RedisStore,
    )
    from langchain_community.storage.sql import (
        SQLStore,
    )
    from langchain_community.storage.upstash_redis import (
        UpstashRedisByteStore,
        UpstashRedisStore,
    )
    from langchain_community.storage.opensearch import (
        OpenSearchStore
    )

__all__ = [
    "AstraDBByteStore",
    "AstraDBStore",
    "CassandraByteStore",
    "MongoDBStore",
    "MongoDBByteStore",
    "RedisStore",
    "SQLStore",
    "UpstashRedisByteStore",
    "UpstashRedisStore",
    "OpenSearchStore"
]

_module_lookup = {
    "AstraDBByteStore": "langchain_community.storage.astradb",
    "AstraDBStore": "langchain_community.storage.astradb",
    "CassandraByteStore": "langchain_community.storage.cassandra",
    "MongoDBStore": "langchain_community.storage.mongodb",
    "MongoDBByteStore": "langchain_community.storage.mongodb",
    "RedisStore": "langchain_community.storage.redis",
    "SQLStore": "langchain_community.storage.sql",
    "UpstashRedisByteStore": "langchain_community.storage.upstash_redis",
    "UpstashRedisStore": "langchain_community.storage.upstash_redis",
    "OpenSearchStore": "langchain_community.storage.opensearch",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")

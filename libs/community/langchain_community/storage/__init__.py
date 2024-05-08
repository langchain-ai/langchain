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
    from langchain_community.storage.mongodb import (
        MongoDBStore,
    )
    from langchain_community.storage.redis import (
        RedisStore,
    )
    from langchain_community.storage.upstash_redis import (
        UpstashRedisByteStore,
        UpstashRedisStore,
    )

__all__ = [
    "AstraDBByteStore",
    "AstraDBStore",
    "MongoDBStore",
    "RedisStore",
    "UpstashRedisByteStore",
    "UpstashRedisStore",
]

_module_lookup = {
    "AstraDBByteStore": "langchain_community.storage.astradb",
    "AstraDBStore": "langchain_community.storage.astradb",
    "MongoDBStore": "langchain_community.storage.mongodb",
    "RedisStore": "langchain_community.storage.redis",
    "UpstashRedisByteStore": "langchain_community.storage.upstash_redis",
    "UpstashRedisStore": "langchain_community.storage.upstash_redis",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = list(_module_lookup.keys())

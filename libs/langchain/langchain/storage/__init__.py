"""Implementations of key-value stores and storage helpers.

Module provides implementations of various key-value stores that conform
to a simple key-value interface.

The primary goal of these storages is to support implementation of caching.
"""

from typing import TYPE_CHECKING, Any

from langchain_core.stores import (
    InMemoryByteStore,
    InMemoryStore,
    InvalidKeyException,
)

from langchain._api import create_importer
from langchain.storage._lc_store import create_kv_docstore, create_lc_store
from langchain.storage.encoder_backed import EncoderBackedStore
from langchain.storage.file_system import LocalFileStore

if TYPE_CHECKING:
    from langchain_community.storage import (
        RedisStore,
        UpstashRedisByteStore,
        UpstashRedisStore,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "RedisStore": "langchain_community.storage",
    "UpstashRedisByteStore": "langchain_community.storage",
    "UpstashRedisStore": "langchain_community.storage",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "create_kv_docstore",
    "create_lc_store",
    "EncoderBackedStore",
    "InMemoryByteStore",
    "InMemoryStore",
    "InvalidKeyException",
    "LocalFileStore",
    "RedisStore",
    "UpstashRedisByteStore",
    "UpstashRedisStore",
]

"""Implementations of key-value stores and storage helpers.

Module provides implementations of various key-value stores that conform
to a simple key-value interface.

The primary goal of these storages is to support implementation of caching.
"""
from typing import Any

from langchain.storage._lc_store import create_kv_docstore, create_lc_store
from langchain.storage.encoder_backed import EncoderBackedStore
from langchain.storage.file_system import LocalFileStore
from langchain.storage.in_memory import InMemoryByteStore, InMemoryStore

DEPRECATED_IMPORTS = [
    "RedisStore",
    "UpstashRedisByteStore",
    "UpstashRedisStore",
]


def __getattr__(name: str) -> Any:
    if name in DEPRECATED_IMPORTS:
        raise ImportError(
            f"{name} has been moved to the langchain-community package. "
            f"See https://github.com/langchain-ai/langchain/discussions/19083 for more "
            f"information.\n\nTo use it install langchain-community:\n\n"
            f"`pip install -U langchain-community`\n\n"
            f"then import with:\n\n"
            f"`from langchain_community.storage import {name}`"
        )

    raise AttributeError()


__all__ = [
    "EncoderBackedStore",
    "InMemoryStore",
    "InMemoryByteStore",
    "LocalFileStore",
    "create_lc_store",
    "create_kv_docstore",
]

"""Implementations of `key-value stores` and storage helpers.

Module provides implementations of various key-value stores that conform
to a simple key-value interface.

The primary goal of these storages is to support implementation of caching.
"""

from langchain_core.storage._lc_store import create_kv_docstore, create_lc_store
from langchain_core.storage.base import BaseStore, ByteStore, K, V
from langchain_core.storage.encoder_backed import EncoderBackedStore
from langchain_core.storage.exceptions import InvalidKeyException
from langchain_core.storage.file_system import LocalFileStore
from langchain_core.storage.in_memory import InMemoryByteStore, InMemoryStore

__all__ = [
    "BaseStore",
    "ByteStore",
    "EncoderBackedStore",
    "InMemoryByteStore",
    "InMemoryStore",
    "InvalidKeyException",
    "K",
    "LocalFileStore",
    "V",
    "create_kv_docstore",
    "create_lc_store",
]

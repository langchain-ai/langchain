"""Implementations of key-value stores and storage helpers.

Module provides implementations of various key-value stores that conform
to a simple key-value interface.

The primary goal of these storages is to support implementation of caching.
"""
import warnings
from typing import Any

from langchain_core._api import LangChainDeprecationWarning
from langchain_core.stores import (
    InMemoryByteStore,
    InMemoryStore,
    InvalidKeyException,
)

from langchain.storage._lc_store import create_kv_docstore, create_lc_store
from langchain.storage.encoder_backed import EncoderBackedStore
from langchain.storage.file_system import LocalFileStore
from langchain.utils.interactive_env import is_interactive_env


def __getattr__(name: str) -> Any:
    from langchain_community import storage

    # If not in interactive env, raise warning.
    if not is_interactive_env():
        warnings.warn(
            "Importing stores from langchain is deprecated. Importing from "
            "langchain will no longer be supported as of langchain==0.2.0. "
            "Please import from langchain-community instead:\n\n"
            f"`from langchain_community.storage import {name}`.\n\n"
            "To install langchain-community run `pip install -U langchain-community`.",
            category=LangChainDeprecationWarning,
        )

    return getattr(storage, name)


__all__ = [
    "EncoderBackedStore",
    "RedisStore",
    "create_lc_store",
    "create_kv_docstore",
    "LocalFileStore",
    "InMemoryStore",
    "InvalidKeyException",
    "InMemoryByteStore",
    "UpstashRedisByteStore",
    "UpstashRedisStore",
]

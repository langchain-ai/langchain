"""In memory store that is not thread safe and has no eviction policy.

This is a simple implementation of the BaseStore using a dictionary that is useful
primarily for unit testing purposes.
"""

from langchain_core.stores import InMemoryBaseStore, InMemoryByteStore, InMemoryStore

__all__ = [
    "InMemoryStore",
    "InMemoryBaseStore",
    "InMemoryByteStore",
]

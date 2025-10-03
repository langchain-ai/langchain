"""**Store** implements the key-value stores and storage helpers.

Module provides implementations of various key-value stores that conform
to a simple key-value interface.

The primary goal of these storages is to support implementation of caching.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import (
    Any,
    Generic,
    Optional,
    TypeVar,
    Union,
)

from typing_extensions import override

from langchain_core.exceptions import LangChainException
from langchain_core.runnables import run_in_executor

K = TypeVar("K")
V = TypeVar("V")


class BaseStore(ABC, Generic[K, V]):
    """Abstract interface for a key-value store.

    This is an interface that's meant to abstract away the details of
    different key-value stores. It provides a simple interface for
    getting, setting, and deleting key-value pairs.

    The basic methods are `mget`, `mset`, and `mdelete` for getting,
    setting, and deleting multiple key-value pairs at once. The `yield_keys`
    method is used to iterate over keys that match a given prefix.

    The async versions of these methods are also provided, which are
    meant to be used in async contexts. The async methods are named with
    an `a` prefix, e.g., `amget`, `amset`, `amdelete`, and `ayield_keys`.

    By default, the `amget`, `amset`, `amdelete`, and `ayield_keys` methods
    are implemented using the synchronous methods. If the store can natively
    support async  operations, it should override these methods.

    By design the methods only accept batches of keys and values, and not
    single keys or values. This is done to force user code to work with batches
    which will usually be more efficient by saving on round trips to the store.

    Examples:

        .. code-block:: python

            from langchain.storage import BaseStore


            class MyInMemoryStore(BaseStore[str, int]):
                def __init__(self) -> None:
                    self.store: dict[str, int] = {}

                def mget(self, keys: Sequence[str]) -> list[int | None]:
                    return [self.store.get(key) for key in keys]

                def mset(self, key_value_pairs: Sequence[tuple[str, int]]) -> None:
                    for key, value in key_value_pairs:
                        self.store[key] = value

                def mdelete(self, keys: Sequence[str]) -> None:
                    for key in keys:
                        if key in self.store:
                            del self.store[key]

                def yield_keys(self, prefix: str | None = None) -> Iterator[str]:
                    if prefix is None:
                        yield from self.store.keys()
                    else:
                        for key in self.store.keys():
                            if key.startswith(prefix):
                                yield key

    """

    @abstractmethod
    def mget(self, keys: Sequence[K]) -> list[Optional[V]]:
        """Get the values associated with the given keys.

        Args:
            keys (Sequence[K]): A sequence of keys.

        Returns:
            A sequence of optional values associated with the keys.
            If a key is not found, the corresponding value will be None.
        """

    async def amget(self, keys: Sequence[K]) -> list[Optional[V]]:
        """Async get the values associated with the given keys.

        Args:
            keys (Sequence[K]): A sequence of keys.

        Returns:
            A sequence of optional values associated with the keys.
            If a key is not found, the corresponding value will be None.
        """
        return await run_in_executor(None, self.mget, keys)

    @abstractmethod
    def mset(self, key_value_pairs: Sequence[tuple[K, V]]) -> None:
        """Set the values for the given keys.

        Args:
            key_value_pairs (Sequence[tuple[K, V]]): A sequence of key-value pairs.
        """

    async def amset(self, key_value_pairs: Sequence[tuple[K, V]]) -> None:
        """Async set the values for the given keys.

        Args:
            key_value_pairs (Sequence[tuple[K, V]]): A sequence of key-value pairs.
        """
        return await run_in_executor(None, self.mset, key_value_pairs)

    @abstractmethod
    def mdelete(self, keys: Sequence[K]) -> None:
        """Delete the given keys and their associated values.

        Args:
            keys (Sequence[K]): A sequence of keys to delete.
        """

    async def amdelete(self, keys: Sequence[K]) -> None:
        """Async delete the given keys and their associated values.

        Args:
            keys (Sequence[K]): A sequence of keys to delete.
        """
        return await run_in_executor(None, self.mdelete, keys)

    @abstractmethod
    def yield_keys(
        self, *, prefix: Optional[str] = None
    ) -> Union[Iterator[K], Iterator[str]]:
        """Get an iterator over keys that match the given prefix.

        Args:
            prefix (str): The prefix to match.

        Yields:
            Iterator[K | str]: An iterator over keys that match the given prefix.
            This method is allowed to return an iterator over either K or str
            depending on what makes more sense for the given store.
        """

    async def ayield_keys(
        self, *, prefix: Optional[str] = None
    ) -> Union[AsyncIterator[K], AsyncIterator[str]]:
        """Async get an iterator over keys that match the given prefix.

        Args:
            prefix (str): The prefix to match.

        Yields:
            Iterator[K | str]: An iterator over keys that match the given prefix.
            This method is allowed to return an iterator over either K or str
            depending on what makes more sense for the given store.
        """
        iterator = await run_in_executor(None, self.yield_keys, prefix=prefix)
        done = object()
        while True:
            item = await run_in_executor(None, lambda it: next(it, done), iterator)
            if item is done:
                break
            yield item  # type: ignore[misc]


ByteStore = BaseStore[str, bytes]


class InMemoryBaseStore(BaseStore[str, V], Generic[V]):
    """In-memory implementation of the BaseStore using a dictionary."""

    def __init__(self) -> None:
        """Initialize an empty store."""
        self.store: dict[str, V] = {}

    def mget(self, keys: Sequence[str]) -> list[Optional[V]]:
        """Get the values associated with the given keys.

        Args:
            keys (Sequence[str]): A sequence of keys.

        Returns:
            A sequence of optional values associated with the keys.
            If a key is not found, the corresponding value will be None.
        """
        return [self.store.get(key) for key in keys]

    async def amget(self, keys: Sequence[str]) -> list[Optional[V]]:
        """Async get the values associated with the given keys.

        Args:
            keys (Sequence[str]): A sequence of keys.

        Returns:
            A sequence of optional values associated with the keys.
            If a key is not found, the corresponding value will be None.
        """
        return self.mget(keys)

    @override
    def mset(self, key_value_pairs: Sequence[tuple[str, V]]) -> None:
        for key, value in key_value_pairs:
            self.store[key] = value

    @override
    async def amset(self, key_value_pairs: Sequence[tuple[str, V]]) -> None:
        return self.mset(key_value_pairs)

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete the given keys and their associated values.

        Args:
            keys (Sequence[str]): A sequence of keys to delete.
        """
        for key in keys:
            if key in self.store:
                del self.store[key]

    async def amdelete(self, keys: Sequence[str]) -> None:
        """Async delete the given keys and their associated values.

        Args:
            keys (Sequence[str]): A sequence of keys to delete.
        """
        self.mdelete(keys)

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        """Get an iterator over keys that match the given prefix.

        Args:
            prefix (str, optional): The prefix to match. Defaults to None.

        Yields:
            Iterator[str]: An iterator over keys that match the given prefix.
        """
        if prefix is None:
            yield from self.store.keys()
        else:
            for key in self.store:
                if key.startswith(prefix):
                    yield key

    async def ayield_keys(self, prefix: Optional[str] = None) -> AsyncIterator[str]:
        """Async get an async iterator over keys that match the given prefix.

        Args:
            prefix (str, optional): The prefix to match. Defaults to None.

        Yields:
            AsyncIterator[str]: An async iterator over keys that match the given prefix.
        """
        if prefix is None:
            for key in self.store:
                yield key
        else:
            for key in self.store:
                if key.startswith(prefix):
                    yield key


class InMemoryStore(InMemoryBaseStore[Any]):
    """In-memory store for any type of data.

    Attributes:
        store (dict[str, Any]): The underlying dictionary that stores
            the key-value pairs.

    Examples:

        .. code-block:: python

            from langchain.storage import InMemoryStore

            store = InMemoryStore()
            store.mset([("key1", "value1"), ("key2", "value2")])
            store.mget(["key1", "key2"])
            # ['value1', 'value2']
            store.mdelete(["key1"])
            list(store.yield_keys())
            # ['key2']
            list(store.yield_keys(prefix="k"))
            # ['key2']

    """


class InMemoryByteStore(InMemoryBaseStore[bytes]):
    """In-memory store for bytes.

    Attributes:
        store (dict[str, bytes]): The underlying dictionary that stores
            the key-value pairs.

    Examples:

        .. code-block:: python

            from langchain.storage import InMemoryByteStore

            store = InMemoryByteStore()
            store.mset([("key1", b"value1"), ("key2", b"value2")])
            store.mget(["key1", "key2"])
            # [b'value1', b'value2']
            store.mdelete(["key1"])
            list(store.yield_keys())
            # ['key2']
            list(store.yield_keys(prefix="k"))
            # ['key2']

    """


class InvalidKeyException(LangChainException):
    """Raised when a key is invalid; e.g., uses incorrect characters."""

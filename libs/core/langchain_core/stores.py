"""**Store** implements the key-value stores and storage helpers.

Module provides implementations of various key-value stores that conform
to a simple key-value interface.

The primary goal of these storages is to support implementation of caching.
"""
from abc import ABC, abstractmethod
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from langchain_core.exceptions import LangChainException
from langchain_core.runnables import run_in_executor

K = TypeVar("K")
V = TypeVar("V")


class BaseStore(Generic[K, V], ABC):
    """Abstract interface for a key-value store."""

    @abstractmethod
    def mget(self, keys: Sequence[K]) -> List[Optional[V]]:
        """Get the values associated with the given keys.

        Args:
            keys (Sequence[K]): A sequence of keys.

        Returns:
            A sequence of optional values associated with the keys.
            If a key is not found, the corresponding value will be None.
        """

    async def amget(self, keys: Sequence[K]) -> List[Optional[V]]:
        """Get the values associated with the given keys.

        Args:
            keys (Sequence[K]): A sequence of keys.

        Returns:
            A sequence of optional values associated with the keys.
            If a key is not found, the corresponding value will be None.
        """
        return await run_in_executor(None, self.mget, keys)

    @abstractmethod
    def mset(self, key_value_pairs: Sequence[Tuple[K, V]]) -> None:
        """Set the values for the given keys.

        Args:
            key_value_pairs (Sequence[Tuple[K, V]]): A sequence of key-value pairs.
        """

    async def amset(self, key_value_pairs: Sequence[Tuple[K, V]]) -> None:
        """Set the values for the given keys.

        Args:
            key_value_pairs (Sequence[Tuple[K, V]]): A sequence of key-value pairs.
        """
        return await run_in_executor(None, self.mset, key_value_pairs)

    @abstractmethod
    def mdelete(self, keys: Sequence[K]) -> None:
        """Delete the given keys and their associated values.

        Args:
            keys (Sequence[K]): A sequence of keys to delete.
        """

    async def amdelete(self, keys: Sequence[K]) -> None:
        """Delete the given keys and their associated values.

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

        Returns:
            Iterator[K | str]: An iterator over keys that match the given prefix.

            This method is allowed to return an iterator over either K or str
            depending on what makes more sense for the given store.
        """

    async def ayield_keys(
        self, *, prefix: Optional[str] = None
    ) -> Union[AsyncIterator[K], AsyncIterator[str]]:
        """Get an iterator over keys that match the given prefix.

        Args:
            prefix (str): The prefix to match.

        Returns:
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
        self.store: Dict[str, V] = {}

    def mget(self, keys: Sequence[str]) -> List[Optional[V]]:
        """Get the values associated with the given keys.

        Args:
            keys (Sequence[str]): A sequence of keys.

        Returns:
            A sequence of optional values associated with the keys.
            If a key is not found, the corresponding value will be None.
        """
        return [self.store.get(key) for key in keys]

    async def amget(self, keys: Sequence[str]) -> List[Optional[V]]:
        """Get the values associated with the given keys.

        Args:
            keys (Sequence[str]): A sequence of keys.

        Returns:
            A sequence of optional values associated with the keys.
            If a key is not found, the corresponding value will be None.
        """
        return self.mget(keys)

    def mset(self, key_value_pairs: Sequence[Tuple[str, V]]) -> None:
        """Set the values for the given keys.

        Args:
            key_value_pairs (Sequence[Tuple[str, V]]): A sequence of key-value pairs.

        Returns:
            None
        """
        for key, value in key_value_pairs:
            self.store[key] = value

    async def amset(self, key_value_pairs: Sequence[Tuple[str, V]]) -> None:
        """Set the values for the given keys.

        Args:
            key_value_pairs (Sequence[Tuple[str, V]]): A sequence of key-value pairs.

        Returns:
            None
        """
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
        """Delete the given keys and their associated values.

        Args:
            keys (Sequence[str]): A sequence of keys to delete.
        """
        self.mdelete(keys)

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        """Get an iterator over keys that match the given prefix.

        Args:
            prefix (str, optional): The prefix to match. Defaults to None.

        Returns:
            Iterator[str]: An iterator over keys that match the given prefix.
        """
        if prefix is None:
            yield from self.store.keys()
        else:
            for key in self.store.keys():
                if key.startswith(prefix):
                    yield key

    async def ayield_keys(self, prefix: Optional[str] = None) -> AsyncIterator[str]:
        """Get an async iterator over keys that match the given prefix.

        Args:
            prefix (str, optional): The prefix to match. Defaults to None.

        Returns:
            AsyncIterator[str]: An async iterator over keys that match the given prefix.
        """
        if prefix is None:
            for key in self.store.keys():
                yield key
        else:
            for key in self.store.keys():
                if key.startswith(prefix):
                    yield key


class InMemoryStore(InMemoryBaseStore[Any]):
    """In-memory store for any type of data.

    Attributes:
        store (Dict[str, Any]): The underlying dictionary that stores
            the key-value pairs.

    Examples:

        .. code-block:: python

            from langchain.storage import InMemoryStore

            store = InMemoryStore()
            store.mset([('key1', 'value1'), ('key2', 'value2')])
            store.mget(['key1', 'key2'])
            # ['value1', 'value2']
            store.mdelete(['key1'])
            list(store.yield_keys())
            # ['key2']
            list(store.yield_keys(prefix='k'))
            # ['key2']
    """


class InMemoryByteStore(InMemoryBaseStore[bytes]):
    """In-memory store for bytes.

    Attributes:
        store (Dict[str, bytes]): The underlying dictionary that stores
            the key-value pairs.

    Examples:

        .. code-block:: python

            from langchain.storage import InMemoryByteStore

            store = InMemoryByteStore()
            store.mset([('key1', b'value1'), ('key2', b'value2')])
            store.mget(['key1', 'key2'])
            # [b'value1', b'value2']
            store.mdelete(['key1'])
            list(store.yield_keys())
            # ['key2']
            list(store.yield_keys(prefix='k'))
            # ['key2']
    """


class InvalidKeyException(LangChainException):
    """Raised when a key is invalid; e.g., uses incorrect characters."""

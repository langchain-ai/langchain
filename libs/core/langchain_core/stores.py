"""**Store** implements the key-value stores and storage helpers.

Module provides implementations of various key-value stores that conform
to a simple key-value interface.

The primary goal of these storages is to support implementation of caching.
"""
from abc import ABC, abstractmethod
from typing import (
    AsyncIterator,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

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
            yield item


ByteStore = BaseStore[str, bytes]

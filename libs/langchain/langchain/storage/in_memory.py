"""In memory store that is not thread safe and has no eviction policy.

This is a simple implementation of the BaseStore using a dictionary that is useful
primarily for unit testing purposes.
"""
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from langchain_core.stores import BaseStore


class InMemoryStore(BaseStore[str, Any]):
    """In-memory implementation of the BaseStore using a dictionary.

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

    def __init__(self) -> None:
        """Initialize an empty store."""
        self.store: Dict[str, Any] = {}

    def mget(self, keys: Sequence[str]) -> List[Optional[Any]]:
        """Get the values associated with the given keys.

        Args:
            keys (Sequence[str]): A sequence of keys.

        Returns:
            A sequence of optional values associated with the keys.
            If a key is not found, the corresponding value will be None.
        """
        return [self.store.get(key) for key in keys]

    def mset(self, key_value_pairs: Sequence[Tuple[str, Any]]) -> None:
        """Set the values for the given keys.

        Args:
            key_value_pairs (Sequence[Tuple[str, V]]): A sequence of key-value pairs.

        Returns:
            None
        """
        for key, value in key_value_pairs:
            self.store[key] = value

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete the given keys and their associated values.

        Args:
            keys (Sequence[str]): A sequence of keys to delete.
        """
        for key in keys:
            self.store.pop(key, None)

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

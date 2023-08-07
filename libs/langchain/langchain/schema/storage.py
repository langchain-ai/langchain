from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Sequence, List, Optional, Tuple, Iterator

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

    @abstractmethod
    def mset(self, key_value_pairs: Sequence[Tuple[K, V]]) -> None:
        """Set the values for the given keys.

        Args:
            key_value_pairs (Sequence[Tuple[K, V]]): A sequence of key-value pairs.
        """

    @abstractmethod
    def mdelete(self, keys: Sequence[K]) -> None:
        """Delete the given keys and their associated values.

        Args:
            keys (Sequence[K]): A sequence of keys to delete.
        """

    @abstractmethod
    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[K]:
        """Get an iterator over keys that match the given prefix.

        Args:
            prefix (str): The prefix to match.

        Returns:
            Iterator[K]: An iterator over keys that match the given prefix.
        """

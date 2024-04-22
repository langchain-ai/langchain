"""**Store** implements the key-value stores and storage helpers.

Module provides implementations of various key-value stores that conform
to a simple key-value interface.

The primary goal of these storages is to support implementation of caching.
"""
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
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


class LocalFileStore(ByteStore):
    """BaseStore interface that works on the local file system.

    Examples:
        Create a LocalFileStore instance and perform operations on it:

        .. code-block:: python

            from langchain.storage import LocalFileStore

            # Instantiate the LocalFileStore with the root path
            file_store = LocalFileStore("/path/to/root")

            # Set values for keys
            file_store.mset([("key1", b"value1"), ("key2", b"value2")])

            # Get values for keys
            values = file_store.mget(["key1", "key2"])  # Returns [b"value1", b"value2"]

            # Delete keys
            file_store.mdelete(["key1"])

            # Iterate over keys
            for key in file_store.yield_keys():
                print(key)  # noqa: T201

    """

    def __init__(
        self,
        root_path: Union[str, Path],
        *,
        chmod_file: Optional[int] = None,
        chmod_dir: Optional[int] = None,
    ) -> None:
        """Implement the BaseStore interface for the local file system.

        Args:
            root_path (Union[str, Path]): The root path of the file store. All keys are
                interpreted as paths relative to this root.
            chmod_file: (optional, defaults to `None`) If specified, sets permissions
                for newly created files, overriding the current `umask` if needed.
            chmod_dir: (optional, defaults to `None`) If specified, sets permissions
                for newly created dirs, overriding the current `umask` if needed.
        """
        self.root_path = Path(root_path).absolute()
        self.chmod_file = chmod_file
        self.chmod_dir = chmod_dir

    def _get_full_path(self, key: str) -> Path:
        """Get the full path for a given key relative to the root path.

        Args:
            key (str): The key relative to the root path.

        Returns:
            Path: The full path for the given key.
        """
        if not re.match(r"^[a-zA-Z0-9_.\-/]+$", key):
            raise InvalidKeyException(f"Invalid characters in key: {key}")
        full_path = os.path.abspath(self.root_path / key)
        common_path = os.path.commonpath([str(self.root_path), full_path])
        if common_path != str(self.root_path):
            raise InvalidKeyException(
                f"Invalid key: {key}. Key should be relative to the full path."
                f"{self.root_path} vs. {common_path} and full path of {full_path}"
            )

        return Path(full_path)

    def _mkdir_for_store(self, dir: Path) -> None:
        """Makes a store directory path (including parents) with specified permissions

        This is needed because `Path.mkdir()` is restricted by the current `umask`,
        whereas the explicit `os.chmod()` used here is not.

        Args:
            dir: (Path) The store directory to make

        Returns:
            None
        """
        if not dir.exists():
            self._mkdir_for_store(dir.parent)
            dir.mkdir(exist_ok=True)
            if self.chmod_dir is not None:
                os.chmod(dir, self.chmod_dir)

    def mget(self, keys: Sequence[str]) -> List[Optional[bytes]]:
        """Get the values associated with the given keys.

        Args:
            keys: A sequence of keys.

        Returns:
            A sequence of optional values associated with the keys.
            If a key is not found, the corresponding value will be None.
        """
        values: List[Optional[bytes]] = []
        for key in keys:
            full_path = self._get_full_path(key)
            if full_path.exists():
                value = full_path.read_bytes()
                values.append(value)
            else:
                values.append(None)
        return values

    def mset(self, key_value_pairs: Sequence[Tuple[str, bytes]]) -> None:
        """Set the values for the given keys.

        Args:
            key_value_pairs: A sequence of key-value pairs.

        Returns:
            None
        """
        for key, value in key_value_pairs:
            full_path = self._get_full_path(key)
            self._mkdir_for_store(full_path.parent)
            full_path.write_bytes(value)
            if self.chmod_file is not None:
                os.chmod(full_path, self.chmod_file)

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete the given keys and their associated values.

        Args:
            keys (Sequence[str]): A sequence of keys to delete.

        Returns:
            None
        """
        for key in keys:
            full_path = self._get_full_path(key)
            if full_path.exists():
                full_path.unlink()

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        """Get an iterator over keys that match the given prefix.

        Args:
            prefix (Optional[str]): The prefix to match.

        Returns:
            Iterator[str]: An iterator over keys that match the given prefix.
        """
        prefix_path = self._get_full_path(prefix) if prefix else self.root_path
        for file in prefix_path.rglob("*"):
            if file.is_file():
                relative_path = file.relative_to(self.root_path)
                yield str(relative_path)


class InMemoryBaseStore(BaseStore[str, V], Generic[V]):
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


InMemoryStore = InMemoryBaseStore[Any]
InMemoryByteStore = InMemoryBaseStore[bytes]


class InvalidKeyException(LangChainException):
    """Raised when a key is invalid; e.g., uses incorrect characters."""

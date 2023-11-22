import re
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Union

from langchain_core.stores import BaseStore

from langchain.storage.exceptions import InvalidKeyException


class LocalFileStore(BaseStore[str, bytes]):
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
                print(key)

    """

    def __init__(self, root_path: Union[str, Path]) -> None:
        """Implement the BaseStore interface for the local file system.

        Args:
            root_path (Union[str, Path]): The root path of the file store. All keys are
                interpreted as paths relative to this root.
        """
        self.root_path = Path(root_path)

    def _get_full_path(self, key: str) -> Path:
        """Get the full path for a given key relative to the root path.

        Args:
            key (str): The key relative to the root path.

        Returns:
            Path: The full path for the given key.
        """
        if not re.match(r"^[a-zA-Z0-9_.\-/]+$", key):
            raise InvalidKeyException(f"Invalid characters in key: {key}")
        return self.root_path / key

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
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_bytes(value)

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

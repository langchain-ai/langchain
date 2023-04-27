"""Use to load blobs from the local file system."""
from pathlib import Path
from typing import Iterable, Union

from langchain.document_loaders.blob_loaders.schema import Blob, BlobLoader


# PUBLIC API


class FileSystemBlobLoader(BlobLoader):
    """Blob loader for the local file system.

    Example:

    .. code-block:: python

        from langchain.document_loaders.blob_loaders import FileSystemBlobLoader
        loader = FileSystemBlobLoader("/path/to/directory")
        for blob in loader.yield_blobs():
            print(blob)
    """

    def __init__(
        self,
        path: Union[str, Path],
        *,
        glob: str = "**/[!.]*",
    ) -> None:
        """Initialize with path to directory and how to glob over it.

        Args:
            path: Path to directory to load from.
            glob: Glob pattern to use to find files.

        Examples:

        ... code-block:: python

            # Recursively load all text files in a directory.
            loader = FileSystemBlobLoader("/path/to/directory", glob="**/*.txt")

            # Recursively load all non-hidden files in a directory.
            loader = FileSystemBlobLoader("/path/to/directory", glob="**/[!.]*")

            # Load all files in a directory without recursion.
            loader = FileSystemBlobLoader("/path/to/directory", glob="*")
        """
        if isinstance(path, Path):
            _path = path
        elif isinstance(path, str):
            _path = Path(path)
        else:
            raise TypeError(f"Expected str or Path, got {type(path)}")

        self.path = _path
        self.glob = glob

    def yield_blobs(
        self,
    ) -> Iterable[Blob]:
        """Yield blobs that match the requested pattern."""
        paths = Path(self.path).glob(self.glob)
        for path in paths:
            if path.is_file():
                yield Blob.from_path(str(path))

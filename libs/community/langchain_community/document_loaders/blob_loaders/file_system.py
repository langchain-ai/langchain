"""Use to load blobs from the local file system."""
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Sequence, TypeVar, Union

from langchain_community.document_loaders.blob_loaders.schema import Blob, BlobLoader

T = TypeVar("T")


def _make_iterator(
    length_func: Callable[[], int], show_progress: bool = False
) -> Callable[[Iterable[T]], Iterator[T]]:
    """Create a function that optionally wraps an iterable in tqdm."""
    if show_progress:
        try:
            from tqdm.auto import tqdm
        except ImportError:
            raise ImportError(
                "You must install tqdm to use show_progress=True."
                "You can install tqdm with `pip install tqdm`."
            )

        # Make sure to provide `total` here so that tqdm can show
        # a progress bar that takes into account the total number of files.
        def _with_tqdm(iterable: Iterable[T]) -> Iterator[T]:
            """Wrap an iterable in a tqdm progress bar."""
            return tqdm(iterable, total=length_func())

        iterator = _with_tqdm
    else:
        iterator = iter  # type: ignore

    return iterator


# PUBLIC API


class FileSystemBlobLoader(BlobLoader):
    """Load blobs in the local file system.

    Example:

    .. code-block:: python

        from langchain_community.document_loaders.blob_loaders import FileSystemBlobLoader
        loader = FileSystemBlobLoader("/path/to/directory")
        for blob in loader.yield_blobs():
            print(blob)
    """  # noqa: E501

    def __init__(
        self,
        path: Union[str, Path],
        *,
        glob: str = "**/[!.]*",
        exclude: Sequence[str] = (),
        suffixes: Optional[Sequence[str]] = None,
        show_progress: bool = False,
    ) -> None:
        """Initialize with a path to directory and how to glob over it.

        Args:
            path: Path to directory to load from or path to file to load.
                  If a path to a file is provided, glob/exclude/suffixes are ignored.
            glob: Glob pattern relative to the specified path
                  by default set to pick up all non-hidden files
            exclude: patterns to exclude from results, use glob syntax
            suffixes: Provide to keep only files with these suffixes
                      Useful when wanting to keep files with different suffixes
                      Suffixes must include the dot, e.g. ".txt"
            show_progress: If true, will show a progress bar as the files are loaded.
                           This forces an iteration through all matching files
                           to count them prior to loading them.

        Examples:

            .. code-block:: python
                from langchain_community.document_loaders.blob_loaders import FileSystemBlobLoader

                # Load a single file.
                loader = FileSystemBlobLoader("/path/to/file.txt")

                # Recursively load all text files in a directory.
                loader = FileSystemBlobLoader("/path/to/directory", glob="**/*.txt")

                # Recursively load all non-hidden files in a directory.
                loader = FileSystemBlobLoader("/path/to/directory", glob="**/[!.]*")

                # Load all files in a directory without recursion.
                loader = FileSystemBlobLoader("/path/to/directory", glob="*")

                # Recursively load all files in a directory, except for py or pyc files.
                loader = FileSystemBlobLoader(
                    "/path/to/directory",
                    glob="**/*.txt",
                    exclude=["**/*.py", "**/*.pyc"]
                )
        """  # noqa: E501
        if isinstance(path, Path):
            _path = path
        elif isinstance(path, str):
            _path = Path(path)
        else:
            raise TypeError(f"Expected str or Path, got {type(path)}")

        self.path = _path.expanduser()  # Expand user to handle ~
        self.glob = glob
        self.suffixes = set(suffixes or [])
        self.show_progress = show_progress
        self.exclude = exclude

    def yield_blobs(
        self,
    ) -> Iterable[Blob]:
        """Yield blobs that match the requested pattern."""
        iterator = _make_iterator(
            length_func=self.count_matching_files, show_progress=self.show_progress
        )

        for path in iterator(self._yield_paths()):
            yield Blob.from_path(path)

    def _yield_paths(self) -> Iterable[Path]:
        """Yield paths that match the requested pattern."""
        if self.path.is_file():
            yield self.path
            return

        paths = self.path.glob(self.glob)
        for path in paths:
            if self.exclude:
                if any(path.match(glob) for glob in self.exclude):
                    continue
            if path.is_file():
                if self.suffixes and path.suffix not in self.suffixes:
                    continue
                yield path

    def count_matching_files(self) -> int:
        """Count files that match the pattern without loading them."""
        # Carry out a full iteration to count the files without
        # materializing anything expensive in memory.
        num = 0
        for _ in self._yield_paths():
            num += 1
        return num

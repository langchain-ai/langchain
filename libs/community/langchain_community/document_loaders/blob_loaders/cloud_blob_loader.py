"""Use to load blobs from the local file system."""
import contextlib
import mimetypes
import tempfile
from io import BufferedReader, BytesIO
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Generator,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    TypeVar,
    Union,
)
from urllib.parse import urlparse

if TYPE_CHECKING:
    from cloudpathlib import AnyPath

from langchain_community.document_loaders.blob_loaders.schema import (
    Blob,
    BlobLoader,
)

T = TypeVar("T")


class _CloudBlob(Blob):
    def as_string(self) -> str:
        """Read data as a string."""
        from cloudpathlib import AnyPath

        if self.data is None and self.path:
            return AnyPath(self.path).read_text(encoding=self.encoding)  # type: ignore
        elif isinstance(self.data, bytes):
            return self.data.decode(self.encoding)
        elif isinstance(self.data, str):
            return self.data
        else:
            raise ValueError(f"Unable to get string for blob {self}")

    def as_bytes(self) -> bytes:
        """Read data as bytes."""
        from cloudpathlib import AnyPath

        if isinstance(self.data, bytes):
            return self.data
        elif isinstance(self.data, str):
            return self.data.encode(self.encoding)
        elif self.data is None and self.path:
            return AnyPath(self.path).read_bytes()  # type: ignore
        else:
            raise ValueError(f"Unable to get bytes for blob {self}")

    @contextlib.contextmanager
    def as_bytes_io(self) -> Generator[Union[BytesIO, BufferedReader], None, None]:
        """Read data as a byte stream."""
        from cloudpathlib import AnyPath

        if isinstance(self.data, bytes):
            yield BytesIO(self.data)
        elif self.data is None and self.path:
            return AnyPath(self.path).read_bytes()  # type: ignore
        else:
            raise NotImplementedError(f"Unable to convert blob {self}")


def _url_to_filename(url: str) -> str:
    """
    Convert file:, s3:, az: or gs: url to localfile.
    If the file is not here, download it in a temporary file.
    """
    from cloudpathlib import AnyPath

    url_parsed = urlparse(url)
    suffix = Path(url_parsed.path).suffix
    if url_parsed.scheme in ["s3", "az", "gs"]:
        with AnyPath(url).open("rb") as f:  # type: ignore
            temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            while True:
                buf = f.read()
                if not buf:
                    break
                temp_file.write(buf)
            temp_file.close()
            file_path = temp_file.name
    elif url_parsed.scheme in ["file", ""]:
        file_path = url_parsed.path
    else:
        raise ValueError(f"Scheme {url_parsed.scheme} not supported")
    return file_path


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


class CloudBlobLoader(BlobLoader):
    """Load blobs from cloud URL or file:.

    Example:

    .. code-block:: python

        loader = CloudBlobLoader("s3://mybucket/id")

        for blob in loader.yield_blobs():
            print(blob)
    """  # noqa: E501

    def __init__(
        self,
        url: Union[str, "AnyPath"],
        *,
        glob: str = "**/[!.]*",
        exclude: Sequence[str] = (),
        suffixes: Optional[Sequence[str]] = None,
        show_progress: bool = False,
    ) -> None:
        """Initialize with a url and how to glob over it.

        Use [CloudPathLib](https://cloudpathlib.drivendata.org/).

        Args:
            url: Cloud URL to load from.
                 Supports s3://, az://, gs://, file:// schemes.
                 If no scheme is provided, it is assumed to be a local file.
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
                from langchain_community.document_loaders.blob_loaders import CloudBlobLoader

                # Load a single file.
                loader = CloudBlobLoader("s3://mybucket/id") # az://

                # Recursively load all text files in a directory.
                loader = CloudBlobLoader("az://mybucket/id", glob="**/*.txt")

                # Recursively load all non-hidden files in a directory.
                loader = CloudBlobLoader("gs://mybucket/id", glob="**/[!.]*")

                # Load all files in a directory without recursion.
                loader = CloudBlobLoader("s3://mybucket/id", glob="*")

                # Recursively load all files in a directory, except for py or pyc files.
                loader = CloudBlobLoader(
                    "s3://mybucket/id",
                    glob="**/*.txt",
                    exclude=["**/*.py", "**/*.pyc"]
                )
        """  # noqa: E501
        from cloudpathlib import AnyPath

        url_parsed = urlparse(str(url))

        if url_parsed.scheme == "file":
            url = url_parsed.path

        if isinstance(url, str):
            self.path = AnyPath(url)
        else:
            self.path = url

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
            # yield Blob.from_path(path)
            yield self.from_path(path)

    def _yield_paths(self) -> Iterable["AnyPath"]:
        """Yield paths that match the requested pattern."""
        if self.path.is_file():  # type: ignore
            yield self.path
            return

        paths = self.path.glob(self.glob)
        for path in paths:
            if self.exclude:
                if any(path.match(glob) for glob in self.exclude):
                    continue
            if path.is_file():
                if self.suffixes and path.suffix not in self.suffixes:
                    continue  # FIXME
                yield path

    def count_matching_files(self) -> int:
        """Count files that match the pattern without loading them."""
        # Carry out a full iteration to count the files without
        # materializing anything expensive in memory.
        num = 0
        for _ in self._yield_paths():
            num += 1
        return num

    @classmethod
    def from_path(
        cls,
        path: "AnyPath",
        *,
        encoding: str = "utf-8",
        mime_type: Optional[str] = None,
        guess_type: bool = True,
        metadata: Optional[dict] = None,
    ) -> Blob:
        """Load the blob from a path like object.

        Args:
            path: path like object to file to be read
                  Supports s3://, az://, gs://, file:// schemes.
                  If no scheme is provided, it is assumed to be a local file.
            encoding: Encoding to use if decoding the bytes into a string
            mime_type: if provided, will be set as the mime-type of the data
            guess_type: If True, the mimetype will be guessed from the file extension,
                        if a mime-type was not provided
            metadata: Metadata to associate with the blob

        Returns:
            Blob instance
        """
        if mime_type is None and guess_type:
            _mimetype = mimetypes.guess_type(path)[0] if guess_type else None  # type: ignore
        else:
            _mimetype = mime_type

        url_parsed = urlparse(str(path))
        if url_parsed.scheme in ["file", ""]:
            if url_parsed.scheme == "file":
                local_path = url_parsed.path
            else:
                local_path = str(path)
            return Blob(
                data=None,
                mimetype=_mimetype,
                encoding=encoding,
                path=local_path,
                metadata=metadata if metadata is not None else {},
            )

        return _CloudBlob(
            data=None,
            mimetype=_mimetype,
            encoding=encoding,
            path=str(path),
            metadata=metadata if metadata is not None else {},
        )

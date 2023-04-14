import contextlib
import mimetypes
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import PurePath
from pydantic import BaseModel
from typing import Union, Optional, Generator

PathLike = Union[str, PurePath]


class Blob(BaseModel):
    """A blob is used to represent raw data by either reference or value.

    Provides an interface to materialize the blob in different representations.

    This is inspired by: https://developer.mozilla.org/en-US/docs/Web/API/Blob

    """

    data: Union[bytes, str, None]
    mimetype: Optional[str] = None
    encoding: str = "utf-8"  # Use utf-8 as default encoding, if decoding to string
    # Location where the original content was found
    # Represent location on the local file system
    # Useful for situations where downstream code assumes it must work with file paths
    # rather than in-memory content.
    path: Optional[PathLike] = None

    class Config:
        arbitrary_types_allowed = True
        frozen = True

    @property
    def source(self) -> Optional[str]:
        """The source location of the blob as string if known otherwise none."""
        return str(self.path) if self.path else None

    def as_string(self) -> str:
        """Read data as a string."""
        encoding = self.encoding or "utf-8"
        if self.data is None and self.path:
            with open(str(self.path), "r", encoding=self.encoding) as f:
                return f.read()
        elif isinstance(self.data, bytes):
            return self.data.decode(encoding)
        else:
            return self.data

    def as_bytes(self) -> bytes:
        """Read data as bytes."""
        if isinstance(self.data, bytes):
            return self.data
        elif self.data is None and self.path:
            with open(str(self.path), "rb") as f:
                return f.read()
        else:
            raise NotImplementedError(f"Unable to get bytes for blob {self}")

    @contextlib.contextmanager
    def as_bytes_io(self) -> BytesIO:
        """Read data as a byte stream."""
        if isinstance(self.data, bytes):
            yield BytesIO(self.data)
        elif self.data is None and self.path:
            with open(str(self.path), "rb") as f:
                yield f
        else:
            raise NotImplementedError(f"Unable to convert blob {self}")

    @classmethod
    def from_path(
        cls,
        path: Union[str, PurePath],
        *,
        encoding: str = "utf-8",
        guess_type: bool = True,
    ) -> "Blob":
        """Load the blob from a path like object.

        Args:
            path: path like object to file to be read
            encoding: Encoding to use if decoding the bytes into a string
            guess_type: If True, the mimetype will be guessed from the file extension

        Returns:
            Blob instance
        """
        mimetype = mimetypes.guess_type(path)[0] if guess_type else None
        # We do not load the data immediately!
        # And instead we treat the blob has containing a reference to the underlying data.
        return cls(data=None, mimetype=mimetype, encoding=encoding, path=path)

    @classmethod
    def from_data(
        cls,
        data: Union[str, bytes],
        *,
        encoding: str = "utf-8",
        mime_type: Optional[str] = None,
        path: Optional[str] = None,
    ) -> "Blob":
        """Initialize the blob from in-memory data.
        Args:
            data: the in-memory data associated with the blob
            encoding: Encoding to use if decoding the bytes into a string
            mime_type: if provided, will be set as the mime-type of the data
            path: if provided, will be set as the source from which the data came

        Returns:
            Blob instance
        """
        return cls(data=data, mime_type=mime_type, encoding=encoding, path=path)

    def __repr__(self) -> str:
        """Define the blob representation."""
        str_repr = f"Blob {id(self)}"
        if self.source:
            str_repr += f" {self.source}"
        return str_repr


class BlobLoader(ABC):
    @abstractmethod
    def yield_blobs(
        self,
    ) -> Generator[Blob, None, None]:
        """A lazy loader for raw data represented by LangChain's Blob object."""
        raise NotImplementedError()

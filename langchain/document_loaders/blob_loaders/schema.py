from __future__ import annotations

import contextlib
import mimetypes
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import PurePath
from pydantic import BaseModel
from typing import Union, Optional, Generator

PathLike = Union[str, PurePath]


class Blob(BaseModel):
    """Langchain representation of a blob.

    This representation is inspired by: https://developer.mozilla.org/en-US/docs/Web/API/Blob
    """

    data: Union[bytes, str, None]
    mimetype: Optional[str] = None
    encoding: str = "utf-8"  # Use utf-8 as default encoding, if decoding to string
    # Location where the original content was found
    # Represent location on the local file system
    # Useful for situations where downstream code assumes it must work with file paths
    # rather than in-memory content.
    path_like: Optional[PathLike] = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def source(self) -> Optional[str]:
        """The source location of the blob as string if known otherwise none."""
        return str(self.path_like) if self.path_like else None

    def as_string(self) -> str:
        """Read data as a string."""
        encoding = self.encoding or "utf-8"
        if self.data is None and self.path_like:
            pass
        if isinstance(self.data, bytes):
            return self.data.decode(encoding)
        else:
            return self.data

    def as_bytes(self) -> bytes:
        """Read data as bytes."""
        if isinstance(self.data, bytes):
            return self.data
        else:
            return self.data.encode(self.encoding or "utf-8")

    @contextlib.contextmanager
    def as_bytes_io(self) -> BytesIO:
        """Read data as a byte stream."""
        if isinstance(self.data, bytes):
            yield BytesIO(self.data)
        elif self.data is None and self.path_like:
            with open(str(self.path_like), "rb") as f:
                yield f
        else:
            raise NotImplementedError(f"Unable to convert blob {self}")

    @classmethod
    def from_path(
        cls,
        path_like: Union[str, PurePath],
        *,
        encoding: Optional[str] = None,
        guess_type: bool = True,
    ) -> "Blob":
        """Load the blob from a path like object.

        Args:
            path_like: path like object to file to be read
            encoding: If provided, the file will be read as text, and the encoding will be used.
            guess_type: If True, the mimetype will be guessed from the file extension

        Returns:
            Blob instance
        """
        mimetype = mimetypes.guess_type(path_like)[0] if guess_type else None
        return cls(data=None, mimetype=mimetype, encoding=encoding, location=path_like)

    def __repr__(self) -> str:
        """Define the blob representation."""
        return f"Blob {id(self)} ({self.source})"


class BlobLoader(ABC):
    @abstractmethod
    def yield_blobs(
        self,
    ) -> Generator[Blob, None, None]:
        """A lazy loader for raw data represented by LangChain's Blob object."""
        raise NotImplementedError()

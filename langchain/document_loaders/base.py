"""Base loader class."""
from __future__ import annotations

import abc
import contextlib
import mimetypes
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import PurePath
from pydantic import BaseModel
from typing import List, Optional, Union, Generator

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter

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
            raise NotImplementedError()

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


class BaseLoader(ABC):
    """Base loader class."""

    @abstractmethod
    def load(self) -> List[Document]:
        """Load data into document objects."""

    def load_and_split(
        self, text_splitter: Optional[TextSplitter] = None
    ) -> List[Document]:
        """Load documents and split into chunks."""
        if text_splitter is None:
            _text_splitter: TextSplitter = RecursiveCharacterTextSplitter()
        else:
            _text_splitter = text_splitter
        docs = self.load()
        return _text_splitter.split_documents(docs)

    @abstractmethod
    def lazy_load(
        self,
    ) -> Union[Generator[Blob, None, None], Generator[Document, None, None]]:
        """A lazy loader for content.

        Content can be represented as a `blob` or as a `document`.

        Yielding `blobs` is preferred as it allows to decouple parsing of blobs from loading
        the blobs.

        Future implementations should favor implementing a lazy loader to avoid loading all content
        eagerly into memory.

        The Union on the output type is a bit unfortunate, as it'll force users of sub-classes
        to use `from typing import cast` to cast the output to the correct type.

        TODO(Eugene): Check if there is an overload solution
        """
        raise NotImplementedError()


class BaseBlobParser(ABC):
    """Abstract interface for blob parsers.

    * A blob is a representation of raw data
    * A blob parser provides a way to parse a blob into one or more documents
    """

    @abc.abstractmethod
    def lazy_parse(self, blob: Blob) -> Generator[Document, None, None]:
        """Lazy parsing interface.

        Subclasses should implement this method and

        Args:
            blob: Blob instance

        Returns:
            Generator of documents
        """
        raise NotImplementedError()

    def parse(self, blob: Blob) -> List[Document]:
        """Eagerly parse the blob into a document or documents.

        This is a convenience method when prototyping interactively.

        For serious use, the lazy_parse method should be used instead as it allows
        for lazy loading of content.

        Subclasses should generally not over-ride this parse method.

        Args:
            blob: Blob instance

        Returns:
            List of documents
        """
        return list(self.lazy_parse(blob))

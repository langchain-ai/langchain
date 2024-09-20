from __future__ import annotations

import contextlib
import mimetypes
from collections.abc import Generator
from io import BufferedReader, BytesIO
from pathlib import PurePath
from typing import Any, Literal, Optional, Union, cast

from pydantic import ConfigDict, Field, field_validator, model_validator

from langchain_core.load.serializable import Serializable

PathLike = Union[str, PurePath]


class BaseMedia(Serializable):
    """Use to represent media content.

    Media objects can be used to represent raw data, such as text or binary data.

    LangChain Media objects allow associating metadata and an optional identifier
    with the content.

    The presence of an ID and metadata make it easier to store, index, and search
    over the content in a structured way.
    """

    # The ID field is optional at the moment.
    # It will likely become required in a future major release after
    # it has been adopted by enough vectorstore implementations.
    id: Optional[str] = None
    """An optional identifier for the document.

    Ideally this should be unique across the document collection and formatted 
    as a UUID, but this will not be enforced.
    
    .. versionadded:: 0.2.11
    """

    metadata: dict = Field(default_factory=dict)
    """Arbitrary metadata associated with the content."""

    @field_validator("id", mode="before")
    def cast_id_to_str(cls, id_value: Any) -> Optional[str]:
        if id_value is not None:
            return str(id_value)
        else:
            return id_value


class Blob(BaseMedia):
    """Blob represents raw data by either reference or value.

    Provides an interface to materialize the blob in different representations, and
    help to decouple the development of data loaders from the downstream parsing of
    the raw data.

    Inspired by: https://developer.mozilla.org/en-US/docs/Web/API/Blob

    Example: Initialize a blob from in-memory data

        .. code-block:: python

            from langchain_core.documents import Blob

            blob = Blob.from_data("Hello, world!")

            # Read the blob as a string
            print(blob.as_string())

            # Read the blob as bytes
            print(blob.as_bytes())

            # Read the blob as a byte stream
            with blob.as_bytes_io() as f:
                print(f.read())

    Example: Load from memory and specify mime-type and metadata

        .. code-block:: python

            from langchain_core.documents import Blob

            blob = Blob.from_data(
                data="Hello, world!",
                mime_type="text/plain",
                metadata={"source": "https://example.com"}
            )

    Example: Load the blob from a file

        .. code-block:: python

            from langchain_core.documents import Blob

            blob = Blob.from_path("path/to/file.txt")

            # Read the blob as a string
            print(blob.as_string())

            # Read the blob as bytes
            print(blob.as_bytes())

            # Read the blob as a byte stream
            with blob.as_bytes_io() as f:
                print(f.read())
    """

    data: Union[bytes, str, None]
    """Raw data associated with the blob."""
    mimetype: Optional[str] = None
    """MimeType not to be confused with a file extension."""
    encoding: str = "utf-8"
    """Encoding to use if decoding the bytes into a string.

    Use utf-8 as default encoding, if decoding to string.
    """
    path: Optional[PathLike] = None
    """Location where the original content was found."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    @property
    def source(self) -> Optional[str]:
        """The source location of the blob as string if known otherwise none.

        If a path is associated with the blob, it will default to the path location.

        Unless explicitly set via a metadata field called "source", in which
        case that value will be used instead.
        """
        if self.metadata and "source" in self.metadata:
            return cast(Optional[str], self.metadata["source"])
        return str(self.path) if self.path else None

    @model_validator(mode="before")
    @classmethod
    def check_blob_is_valid(cls, values: dict[str, Any]) -> Any:
        """Verify that either data or path is provided."""
        if "data" not in values and "path" not in values:
            raise ValueError("Either data or path must be provided")
        return values

    def as_string(self) -> str:
        """Read data as a string."""
        if self.data is None and self.path:
            with open(str(self.path), encoding=self.encoding) as f:
                return f.read()
        elif isinstance(self.data, bytes):
            return self.data.decode(self.encoding)
        elif isinstance(self.data, str):
            return self.data
        else:
            raise ValueError(f"Unable to get string for blob {self}")

    def as_bytes(self) -> bytes:
        """Read data as bytes."""
        if isinstance(self.data, bytes):
            return self.data
        elif isinstance(self.data, str):
            return self.data.encode(self.encoding)
        elif self.data is None and self.path:
            with open(str(self.path), "rb") as f:
                return f.read()
        else:
            raise ValueError(f"Unable to get bytes for blob {self}")

    @contextlib.contextmanager
    def as_bytes_io(self) -> Generator[Union[BytesIO, BufferedReader], None, None]:
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
        path: PathLike,
        *,
        encoding: str = "utf-8",
        mime_type: Optional[str] = None,
        guess_type: bool = True,
        metadata: Optional[dict] = None,
    ) -> Blob:
        """Load the blob from a path like object.

        Args:
            path: path like object to file to be read
            encoding: Encoding to use if decoding the bytes into a string
            mime_type: if provided, will be set as the mime-type of the data
            guess_type: If True, the mimetype will be guessed from the file extension,
                        if a mime-type was not provided
            metadata: Metadata to associate with the blob

        Returns:
            Blob instance
        """
        if mime_type is None and guess_type:
            _mimetype = mimetypes.guess_type(path)[0] if guess_type else None
        else:
            _mimetype = mime_type
        # We do not load the data immediately, instead we treat the blob as a
        # reference to the underlying data.
        return cls(
            data=None,
            mimetype=_mimetype,
            encoding=encoding,
            path=path,
            metadata=metadata if metadata is not None else {},
        )

    @classmethod
    def from_data(
        cls,
        data: Union[str, bytes],
        *,
        encoding: str = "utf-8",
        mime_type: Optional[str] = None,
        path: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Blob:
        """Initialize the blob from in-memory data.

        Args:
            data: the in-memory data associated with the blob
            encoding: Encoding to use if decoding the bytes into a string
            mime_type: if provided, will be set as the mime-type of the data
            path: if provided, will be set as the source from which the data came
            metadata: Metadata to associate with the blob

        Returns:
            Blob instance
        """
        return cls(
            data=data,
            mimetype=mime_type,
            encoding=encoding,
            path=path,
            metadata=metadata if metadata is not None else {},
        )

    def __repr__(self) -> str:
        """Define the blob representation."""
        str_repr = f"Blob {id(self)}"
        if self.source:
            str_repr += f" {self.source}"
        return str_repr


class Document(BaseMedia):
    """Class for storing a piece of text and associated metadata.

    Example:

        .. code-block:: python

            from langchain_core.documents import Document

            document = Document(
                page_content="Hello, world!",
                metadata={"source": "https://example.com"}
            )
    """

    page_content: str
    """String text."""
    type: Literal["Document"] = "Document"

    def __init__(self, page_content: str, **kwargs: Any) -> None:
        """Pass page_content in as positional or named arg."""
        # my-py is complaining that page_content is not defined on the base class.
        # Here, we're relying on pydantic base class to handle the validation.
        super().__init__(page_content=page_content, **kwargs)  # type: ignore[call-arg]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "document"]

    def __str__(self) -> str:
        """Override __str__ to restrict it to page_content and metadata."""
        # The format matches pydantic format for __str__.
        #
        # The purpose of this change is to make sure that user code that
        # feeds Document objects directly into prompts remains unchanged
        # due to the addition of the id field (or any other fields in the future).
        #
        # This override will likely be removed in the future in favor of
        # a more general solution of formatting content directly inside the prompts.
        if self.metadata:
            return f"page_content='{self.page_content}' metadata={self.metadata}"
        else:
            return f"page_content='{self.page_content}'"

"""Base classes for media and documents.

This module contains core abstractions for **data retrieval and processing workflows**:

- `BaseMedia`: Base class providing `id` and `metadata` fields
- `Blob`: Raw data loading (files, binary data) - used by document loaders
- `Document`: Text content for retrieval (RAG, vector stores, semantic search)

!!! note "Not for LLM chat messages"

    These classes are for data processing pipelines, not LLM I/O. For multimodal
    content in chat messages (images, audio in conversations), see
    `langchain.messages` content blocks instead.
"""

from __future__ import annotations

import contextlib
import mimetypes
from io import BufferedReader, BytesIO
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import ConfigDict, Field, model_validator

from langchain_core.load.serializable import Serializable

if TYPE_CHECKING:
    from collections.abc import Generator

PathLike = str | PurePath


class BaseMedia(Serializable):
    """Base class for content used in retrieval and data processing workflows.

    Provides common fields for content that needs to be stored, indexed, or searched.

    !!! note

        For multimodal content in **chat messages** (images, audio sent to/from LLMs),
        use `langchain.messages` content blocks instead.
    """

    # The ID field is optional at the moment.
    # It will likely become required in a future major release after
    # it has been adopted by enough VectorStore implementations.
    id: str | None = Field(default=None, coerce_numbers_to_str=True)
    """An optional identifier for the document.

    Ideally this should be unique across the document collection and formatted
    as a UUID, but this will not be enforced.
    """

    metadata: dict = Field(default_factory=dict)
    """Arbitrary metadata associated with the content."""


class Blob(BaseMedia):
    """Raw data abstraction for document loading and file processing.

    Represents raw bytes or text, either in-memory or by file reference. Used
    primarily by document loaders to decouple data loading from parsing.

    Inspired by [Mozilla's `Blob`](https://developer.mozilla.org/en-US/docs/Web/API/Blob)

    ???+ example "Initialize a blob from in-memory data"

        ```python
        from langchain_core.documents import Blob

        blob = Blob.from_data("Hello, world!")

        # Read the blob as a string
        print(blob.as_string())

        # Read the blob as bytes
        print(blob.as_bytes())

        # Read the blob as a byte stream
        with blob.as_bytes_io() as f:
            print(f.read())
        ```

    ??? example "Load from memory and specify MIME type and metadata"

        ```python
        from langchain_core.documents import Blob

        blob = Blob.from_data(
            data="Hello, world!",
            mime_type="text/plain",
            metadata={"source": "https://example.com"},
        )
        ```

    ??? example "Load the blob from a file"

        ```python
        from langchain_core.documents import Blob

        blob = Blob.from_path("path/to/file.txt")

        # Read the blob as a string
        print(blob.as_string())

        # Read the blob as bytes
        print(blob.as_bytes())

        # Read the blob as a byte stream
        with blob.as_bytes_io() as f:
            print(f.read())
        ```
    """

    data: bytes | str | None = None
    """Raw data associated with the `Blob`."""

    mimetype: str | None = None
    """MIME type, not to be confused with a file extension."""

    encoding: str = "utf-8"
    """Encoding to use if decoding the bytes into a string.

    Uses `utf-8` as default encoding if decoding to string.
    """

    path: PathLike | None = None
    """Location where the original content was found."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    @property
    def source(self) -> str | None:
        """The source location of the blob as string if known otherwise none.

        If a path is associated with the `Blob`, it will default to the path location.

        Unless explicitly set via a metadata field called `'source'`, in which
        case that value will be used instead.
        """
        if self.metadata and "source" in self.metadata:
            return cast("str | None", self.metadata["source"])
        return str(self.path) if self.path else None

    @model_validator(mode="before")
    @classmethod
    def check_blob_is_valid(cls, values: dict[str, Any]) -> Any:
        """Verify that either data or path is provided."""
        if "data" not in values and "path" not in values:
            msg = "Either data or path must be provided"
            raise ValueError(msg)
        return values

    def as_string(self) -> str:
        """Read data as a string.

        Raises:
            ValueError: If the blob cannot be represented as a string.

        Returns:
            The data as a string.
        """
        if self.data is None and self.path:
            return Path(self.path).read_text(encoding=self.encoding)
        if isinstance(self.data, bytes):
            return self.data.decode(self.encoding)
        if isinstance(self.data, str):
            return self.data
        msg = f"Unable to get string for blob {self}"
        raise ValueError(msg)

    def as_bytes(self) -> bytes:
        """Read data as bytes.

        Raises:
            ValueError: If the blob cannot be represented as bytes.

        Returns:
            The data as bytes.
        """
        if isinstance(self.data, bytes):
            return self.data
        if isinstance(self.data, str):
            return self.data.encode(self.encoding)
        if self.data is None and self.path:
            return Path(self.path).read_bytes()
        msg = f"Unable to get bytes for blob {self}"
        raise ValueError(msg)

    @contextlib.contextmanager
    def as_bytes_io(self) -> Generator[BytesIO | BufferedReader, None, None]:
        """Read data as a byte stream.

        Raises:
            NotImplementedError: If the blob cannot be represented as a byte stream.

        Yields:
            The data as a byte stream.
        """
        if isinstance(self.data, bytes):
            yield BytesIO(self.data)
        elif self.data is None and self.path:
            with Path(self.path).open("rb") as f:
                yield f
        else:
            msg = f"Unable to convert blob {self}"
            raise NotImplementedError(msg)

    @classmethod
    def from_path(
        cls,
        path: PathLike,
        *,
        encoding: str = "utf-8",
        mime_type: str | None = None,
        guess_type: bool = True,
        metadata: dict | None = None,
    ) -> Blob:
        """Load the blob from a path like object.

        Args:
            path: Path-like object to file to be read
            encoding: Encoding to use if decoding the bytes into a string
            mime_type: If provided, will be set as the MIME type of the data
            guess_type: If `True`, the MIME type will be guessed from the file
                extension, if a MIME type was not provided
            metadata: Metadata to associate with the `Blob`

        Returns:
            `Blob` instance
        """
        if mime_type is None and guess_type:
            mimetype = mimetypes.guess_type(path)[0]
        else:
            mimetype = mime_type
        # We do not load the data immediately, instead we treat the blob as a
        # reference to the underlying data.
        return cls(
            data=None,
            mimetype=mimetype,
            encoding=encoding,
            path=path,
            metadata=metadata if metadata is not None else {},
        )

    @classmethod
    def from_data(
        cls,
        data: str | bytes,
        *,
        encoding: str = "utf-8",
        mime_type: str | None = None,
        path: str | None = None,
        metadata: dict | None = None,
    ) -> Blob:
        """Initialize the `Blob` from in-memory data.

        Args:
            data: The in-memory data associated with the `Blob`
            encoding: Encoding to use if decoding the bytes into a string
            mime_type: If provided, will be set as the MIME type of the data
            path: If provided, will be set as the source from which the data came
            metadata: Metadata to associate with the `Blob`

        Returns:
            `Blob` instance
        """
        return cls(
            data=data,
            mimetype=mime_type,
            encoding=encoding,
            path=path,
            metadata=metadata if metadata is not None else {},
        )

    def __repr__(self) -> str:
        """Return the blob representation."""
        str_repr = f"Blob {id(self)}"
        if self.source:
            str_repr += f" {self.source}"
        return str_repr


class Document(BaseMedia):
    """Class for storing a piece of text and associated metadata.

    !!! note

        `Document` is for **retrieval workflows**, not chat I/O. For sending text
        to an LLM in a conversation, use message types from `langchain.messages`.

    Example:
        ```python
        from langchain_core.documents import Document

        document = Document(
            page_content="Hello, world!", metadata={"source": "https://example.com"}
        )
        ```
    """

    page_content: str
    """String text."""

    type: Literal["Document"] = "Document"

    def __init__(self, page_content: str, **kwargs: Any) -> None:
        """Pass page_content in as positional or named arg."""
        # my-py is complaining that page_content is not defined on the base class.
        # Here, we're relying on pydantic base class to handle the validation.
        super().__init__(page_content=page_content, **kwargs)  # type: ignore[call-arg,unused-ignore]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return `True` as this class is serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "schema", "document"]`
        """
        return ["langchain", "schema", "document"]

    def __str__(self) -> str:
        """Override `__str__` to restrict it to page_content and metadata.

        Returns:
            A string representation of the `Document`.
        """
        # The format matches pydantic format for __str__.
        #
        # The purpose of this change is to make sure that user code that feeds
        # Document objects directly into prompts remains unchanged due to the addition
        # of the id field (or any other fields in the future).
        #
        # This override will likely be removed in the future in favor of a more general
        # solution of formatting content directly inside the prompts.
        if self.metadata:
            return f"page_content='{self.page_content}' metadata={self.metadata}"
        return f"page_content='{self.page_content}'"

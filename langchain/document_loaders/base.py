"""Base loader class."""
import mimetypes
from abc import ABC, abstractmethod
from io import IOBase
from pydantic import BaseModel
from typing import List, Optional, Union, Generator

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter


class Blob(BaseModel):
    """Langchain representation of a blob.

    This representation is inspired by: https://developer.mozilla.org/en-US/docs/Web/API/Blob
    """

    data: Union[bytes, str, IOBase]
    mimetype: Optional[str] = None
    encoding: Optional[str] = None
    location: Optional[str] = None  # Location where the original content was found

    def read_if_needed(self) -> "Blob":
        """Read data if needed."""
        if isinstance(self.data, IOBase):
            return Blob(  # Not inheritance friendly
                data=self.data.read(),  # TODO(apply encoding here?)
                mimetype=self.mimetype,
                encoding=self.encoding,
            )
        return self

    @classmethod
    def from_file(
        cls,
        path: str,
        *,
        encoding: Optional[str] = None,
        guess_type: bool = True,
    ) -> "Blob":
        """Load the blob from a file on the local file system.

        Args:
            path: Path to the file. TODO(Eugene): Change to pathlike object
            encoding: If provided, the file will be read as text, and the encoding will be used.

        Returns:
            Blob object
        """
        mimetype = mimetypes.guess_type(path)[0] if guess_type else None
        with open(path, "r") as f:
            return cls(
                data=f.read(),
                mimetype=mimetype,
                encoding=encoding,
                location=path,
            )


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

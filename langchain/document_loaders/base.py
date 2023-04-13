"""Base loader class."""

from abc import ABC, abstractmethod
from typing import List, Optional, Union, Generator
from io import IOBase
from pydantic import BaseModel

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter


class Blob(BaseModel):
    """Blob schema."""

    data: Union[bytes, str, IOBase]
    mimetype: Optional[str]


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
    ) -> Generator[Blob, None, None] | Generator[Document, None, None]:
        """A lazy loader for content.

        Content can be represented as a `blob` or as a `document`.

        Yielding `blobs` is preferred as it allows to decouple parsing of blobs from loading
        the blobs.

        Future implementations should favor implementing a lazy loader to avoid loading all content
        eagerly into memory.
        """
        raise NotImplementedError()

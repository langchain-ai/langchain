import abc
import magic
from abc import abstractmethod, ABC
from io import IOBase
from pydantic import BaseModel
from typing import List, Union, Mapping, Optional, Callable, Generator

from langchain.schema import Document


class Blob(BaseModel):
    """Blob schema."""

    data: Union[bytes, str, IOBase]
    mimetype: Optional[str]


class Loader(ABC):
    @abc.abstractmethod
    def load(self, *args, **kwargs) -> Generator[Blob, None, None]:
        """Loader interface."""
        raise NotImplementedError()


class Parser(ABC):
    @abc.abstractmethod
    def parse(self, blob: Blob) -> Generator[Document, None, None]:
        """Parser interface."""
        raise NotImplementedError()


class MimeTypeBasedLoader(Parser):
    def __init__(self, handlers: Mapping[str, Callable[[Blob], Document]]) -> None:
        """A loader based on mime-types."""
        self.handlers = handlers

    def parse(self, blob: Blob) -> Generator[Document, None, None]:
        """Load documents from a file."""
        # TODO(Eugene): Restrict to first 2048 bytes
        mime_type = magic.from_buffer(blob.data, mime=True)
        if mime_type in self.handlers:
            handler = self.handlers[mime_type]
            document = handler(blob)
            yield document
        else:
            raise ValueError(f"Unsupported mime type: {mime_type}")


#
# Textifier = MimeTypeBasedLoader({"text/html": "unstructured"})


class BaseDocumentProcessor(ABC):
    @abstractmethod
    def process(self, documents: Document) -> Document:
        """Process documents."""
        raise NotImplementedError()

    @abstractmethod
    def batch_process(self, documents: List[Document]) -> List[Document]:
        raise NotImplementedError()

    @abstractmethod
    async def aprocess_document(self, documents: Document) -> Document:
        """Process documents."""
        raise NotImplementedError()

    @abstractmethod
    async def abatch_process(self, documents: List[Document]) -> List[Document]:
        """Process documents."""


class SimpleBatchProcessor(BaseDocumentProcessor, ABC):
    """Document processor that iterates over documents and processes them."""

    def __init__(self, max_concurrency: int = 10) -> None:
        self.max_concurrency = max_concurrency

    def batch_process(self, documents: List[Document]) -> List[Document]:
        """Process documents."""
        return [self.process(doc) for doc in documents]

    async def abatch_process(self, documents: List[Document]) -> List[Document]:
        """Process documents."""
        raise NotImplementedError()


class SequentialBatchProcessor(BaseDocumentProcessor):
    """Document processor that processes documents in batches."""

    def __init__(self, processors: List[BaseDocumentProcessor]) -> None:
        self.processors = processors

    def process(self, document: Document) -> Document:
        return self.batch_process([document])[0]

    def batch_process(self, documents: List[Document]) -> List[Document]:
        for processor in self.processors:
            documents = processor.batch_process(documents)
        return documents

    async def aprocess_document(self, document: Document) -> Document:
        return (await self.abatch_process([document]))[0]

    async def abatch_process(self, documents: List[Document]) -> List[Document]:
        """Process documents."""
        for processor in self.processors:
            documents = await processor.abatch_process(documents)
        return documents

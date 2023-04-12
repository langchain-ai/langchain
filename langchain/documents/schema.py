import enum
from abc import abstractmethod, ABC
from pydantic import BaseModel, Field
from typing import List

from langchain.schema import Document


class FileType(enum.Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    HTML = "html"
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    GIF = "gif"
    MP4 = "mp4"
    MP3 = "mp3"
    WAV = "wav"


from typing import Union

from pydantic import BaseModel


class Blob(BaseModel):
    name: str
    type: str
    size: int
    data: Union[bytes, str]


class Content(BaseModel):
    data: bytes | str
    metadata: dict = Field(default_factory=dict)
    mime_type: str


class BaseDocumentProcessor(ABC):
    @abstractmethod
    def process_document(self, documents: Document) -> Document:
        """Process documents."""
        raise NotImplementedError()

    @abstractmethod
    def process_documents(self, documents: List[Document]) -> List[Document]:
        raise NotImplementedError()

    @abstractmethod
    async def aprocess_document(self, documents: Document) -> Document:
        """Process documents."""
        raise NotImplementedError()

    @abstractmethod
    async def aprocess_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents."""


class SimpleBatchProcessor(BaseDocumentProcessor, ABC):
    """Document processor that iterates over documents and processes them."""

    def __init__(self, max_concurrency: int = 10) -> None:
        self.max_concurrency = max_concurrency

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents."""
        return [self.process_document(doc) for doc in documents]

    async def aprocess_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents."""
        raise NotImplementedError()


class SequentialBatchProcessor(BaseDocumentProcessor):
    """Document processor that processes documents in batches."""

    def __init__(self, processors: List[BaseDocumentProcessor]) -> None:
        self.processors = processors

    def process_document(self, document: Document) -> Document:
        return self.process_documents(document)[0]

    def process_documents(self, documents: List[Document]) -> List[Document]:
        for processor in self.processors:
            documents = processor.process_documents(documents)
        return documents

    async def aprocess_document(self, document: Document) -> Document:
        return (await self.aprocess_documents([document]))[0]

    async def aprocess_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents."""
        for processor in self.processors:
            documents = await processor.aprocess_documents(documents)
        return documents


class Loader(BaseDocumentProcessor, ABC):
    def __init__(self, handlers) -> None:
        self.handlers = handlers

    def load_documents(self, path: str) -> Document:
        """Load documents from a file."""
        raise NotImplementedError()


class Textifier(BaseDocumentProcessor, ABC):
    def __init__(self, handlers) -> None:
        self.handlers = handlers

    def load_documents(self, file_path: str) -> Document:
        """Load documents from a file."""
        raise NotImplementedError()

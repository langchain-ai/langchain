"""Schema definitions for document processors.

Document processors take a langchain Document and produce a new Document using some custom
processing logic.
"""
from abc import ABC, abstractmethod
from typing import List, Sequence

from langchain.schema import Document


class BaseDocumentProcessor(ABC):
    @abstractmethod
    def process(self, documents: Document) -> Document:
        """Process documents."""
        raise NotImplementedError()

    @abstractmethod
    def batch_process(self, documents: Sequence[Document]) -> List[Document]:
        raise NotImplementedError()

    @abstractmethod
    async def aprocess_document(self, documents: Document) -> Document:
        """Process documents."""
        raise NotImplementedError()

    @abstractmethod
    async def abatch_process(self, documents: Sequence[Document]) -> List[Document]:
        """Process documents."""
        raise NotImplementedError()


# Something that may need to exist
# class BaseDocumentReducer(ABC):
#     @abstractmethod
#     def process(
#         self, documents: Iterable[Document]  # TODO(Eugene): Is Iterable OK here?
#     ) -> Document:
#         """Process documents."""
#         raise NotImplementedError()
#
#     @abstractmethod
#     def batch_process(self, documents: Sequence[Sequence[Document]]) -> List[Document]:
#         raise NotImplementedError()
#
#     @abstractmethod
#     async def aprocess_document(self, documents: Document) -> Document:
#         """Process documents."""
#         raise NotImplementedError()
#
#     @abstractmethod
#     async def abatch_process(self, documents: Sequence[Document]) -> List[Document]:
#         """Process documents."""


class SimpleBatchProcessor(BaseDocumentProcessor, ABC):
    """Document processor that iterates over documents and processes them."""

    def __init__(self, max_concurrency: int = 10) -> None:
        self.max_concurrency = max_concurrency

    def batch_process(self, documents: Sequence[Document]) -> List[Document]:
        """Process documents."""
        return [self.process(doc) for doc in documents]

    async def abatch_process(self, documents: Sequence[Document]) -> List[Document]:
        """Process documents."""
        raise NotImplementedError()


class SequentialBatchProcessor(BaseDocumentProcessor):
    """Document processor that processes documents in batches."""

    def __init__(self, processors: Sequence[BaseDocumentProcessor]) -> None:
        self.processors = processors

    def process(self, document: Document) -> Document:
        return self.batch_process([document])[0]

    def batch_process(self, documents: Sequence[Document]) -> List[Document]:
        for processor in self.processors:
            documents = processor.batch_process(documents)
        return documents

    async def aprocess_document(self, document: Document) -> Document:
        return (await self.abatch_process([document]))[0]

    async def abatch_process(self, documents: Sequence[Document]) -> List[Document]:
        """Process documents."""
        for processor in self.processors:
            documents = await processor.abatch_process(documents)
        return documents

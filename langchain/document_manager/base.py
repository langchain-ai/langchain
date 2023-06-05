from abc import ABC, abstractmethod
from typing import List

from langchain.docstore.document import Document
from langchain.vectorstores import VectorStore
import hashlib
import json

class DocumentManager(ABC):
    """Interface for document manager.

    The document manager is an abstraction that helps manage
    updates to a collection of documents.
    """

    def get_document_hash(self, document: Document) -> str:
        """Returns the hash of the document."""
        hashable_content = document.page_content + json.dumps(document.metadata)
        return hashlib.sha256(hashable_content.encode('utf-8')).hexdigest()

    @abstractmethod
    def add(self, documents: List[Document], ids: List[str]) -> List[DocumentWithOperation]:
        """Adds documents to the document manager."""

    @abstractmethod
    def update(self, documents: List[Document], ids: List[str]) -> List[DocumentWithOperation]:
        """Updates documents in the document manager."""

    @abstractmethod
    def update_truncate(self, documents: List[Document], ids: List[str])-> List[DocumentWithOperation]:
        """Updates the documents in the document manager.
        Additionally, removes any documents that are not in
        `documents`.
        """

class ChunkOperation(str, Enum):
    """Enum for chunk operations."""
    ADD = 'ADD'
    UPDATE = 'UPDATE'
    REMOVE = 'REMOVE'

class DocumentWithOperation(Document):
    operation: ChunkOperation
    id: str

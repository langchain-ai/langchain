"""Interface for vector stores."""
from abc import ABC, abstractmethod
from typing import List

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings


class VectorStore(ABC):
    """Interface for vector stores."""

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Return docs most similar to query."""

    @abstractmethod
    @classmethod
    def from_texts(cls, texts: List[str], embedding: Embeddings) -> "VectorStore":
        """Construct VectorStore from texts and embedddings."""

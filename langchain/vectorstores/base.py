"""Interface for vector stores."""
from abc import ABC, abstractmethod
from typing import Any, List

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings


class VectorStore(ABC):
    """Interface for vector stores."""

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Return docs most similar to query."""

    @classmethod
    @abstractmethod
    def from_texts(
        cls, texts: List[str], embedding: Embeddings, **kwargs: Any
    ) -> "VectorStore":
        """Return VectorStore initialized from texts and embeddings."""

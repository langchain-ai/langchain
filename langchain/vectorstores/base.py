"""Interface for vector stores."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings


class VectorStore(ABC):
    """Interface for vector stores."""

    @abstractmethod
    def add_texts(
        self, texts: Iterable[str], metadatas: Optional[List[dict]] = None
    ) -> None:
        """Run more texts through the embeddings and add to the vectorstore."""

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Return docs most similar to query."""

    @classmethod
    @abstractmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> VectorStore:
        """Return VectorStore initialized from texts and embeddings."""

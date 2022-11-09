"""Interface for vector stores."""
from abc import ABC, abstractmethod
from typing import List

from langchain.docstore.document import Document


class VectorStore(ABC):
    """Interface for vector stores."""

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Return docs most similar to query."""

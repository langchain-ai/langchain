"""Interface for vector stores."""
from abc import ABC, abstractmethod
from typing import Any, Callable, List

import numpy as np

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings


class VectorStore(ABC):
    """Interface for vector stores."""

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Return docs most similar to query."""

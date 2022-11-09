"""Interface for vector stores."""
from abc import ABC, abstractmethod
<<<<<<< HEAD
from typing import Any, Callable, List

import numpy as np

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
=======
from typing import List

from langchain.docstore.document import Document
>>>>>>> master


class VectorStore(ABC):
    """Interface for vector stores."""

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Return docs most similar to query."""

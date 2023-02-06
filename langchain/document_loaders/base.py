"""Base loader class."""

from abc import ABC, abstractmethod
from typing import List
from langchain.docstore.document import Document


class BaseLoader(ABC):
    """Base loader class."""

    @abstractmethod
    def load(self) -> List[Document]:
        """Load data into document objects."""

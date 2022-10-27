"""Interface to access to place that stores documents."""
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from langchain.docstore.document import Document


class Docstore(ABC):
    """Interface to access to place that stores documents."""

    @abstractmethod
    def search(self, search: str) -> Tuple[str, Optional[Document]]:
        """Search for document.

        If page exists, return the page summary, and a Document object.
        If page does not exist, return similar entries.
        """

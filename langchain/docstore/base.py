"""Interface to access to place that stores documents."""
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

from langchain.docstore.document import Document


class Docstore(ABC):
    """Interface to access to place that stores documents."""

    @abstractmethod
    def search(self, search: str) -> Tuple[str, Optional[Document]]:
        """Search for specific document.

        If page exists, return a Document object.
        If page does not exist, return a string explaining the error.
        """

    def similarity_search(self, search: str) -> List[Document]:
        """Search for documents similar to the query.

        By default, just call the generic search method.
        """
        _, document = self.search(search)
        if document is not None:
            return [document]
        else:
            return []

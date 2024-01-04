"""Interface to access to place that stores documents."""
from abc import ABC, abstractmethod
from typing import Dict, List, Union

from langchain_core.documents import Document


class DocstoreInterface(ABC):
    """Interface to access to place that stores documents."""

    @abstractmethod
    def search(self, search: str) -> Union[str, Document]:
        """Search for a document.

        If page exists, return the page summary, and a Document object.
        If the page does not exist, return similar entries.
        """

    @abstractmethod
    def delete(self, ids: List) -> None:
        """Deleting IDs from in memory dictionary."""


class AddableMixin(ABC):
    """Mixin class that supports adding texts."""

    @abstractmethod
    def add(self, texts: Dict[str, Document]) -> None:
        """Add more documents."""

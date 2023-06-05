"""Interface to access to place that stores documents."""
import abc
from abc import ABC, abstractmethod
from typing import Dict, Union, Sequence, Iterator
from uuid import UUID

from langchain.docstore.persistence import Selector
from langchain.schema import Document


class Docstore(ABC):
    """Interface to access to place that stores documents."""

    @abstractmethod
    def search(self, search: str) -> Union[str, Document]:
        """Search for document.

        If page exists, return the page summary, and a Document object.
        If page does not exist, return similar entries.
        """


class AddableMixin(ABC):
    """Mixin class that supports adding texts."""

    @abstractmethod
    def add(self, texts: Dict[str, Document]) -> None:
        """Add more documents."""


class ArtifactLayer(abc.ABC):
    """Use to keep track of artifacts generated while processing content.

    The first version of the artifact store is used to work with Documents
    rather than Blobs.

    We will likely want to evolve this into Blobs, but faster to prototype
    with Documents.
    """

    @abc.abstractmethod
    def exists(self, ids: Sequence[str]) -> Sequence[bool]:
        """Check if the artifacts with the given id exist."""

    def add(self, documents: Sequence[Document]) -> None:
        """Add the given artifacts."""
        raise NotImplementedError()

    def get_child_documents(self, hash_: UUID) -> Iterator[Document]:
        """Get the child documents of the given parent document."""
        yield from self.get_matching_documents(Selector(parent=hash_))

    def get_matching_documents(self, selector: Selector) -> Iterator[Document]:
        """Yield documents matching the given selector."""
        raise NotImplementedError()

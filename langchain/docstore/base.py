"""Interface to access to place that stores documents."""
import abc
import dataclasses
from abc import ABC, abstractmethod
from typing import Dict, Union, Sequence, Iterator, Optional, List

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


@dataclasses.dataclass(frozen=True)
class Selector:
    """Selection criteria represented in conjunctive normal form.

    https://en.wikipedia.org/wiki/Conjunctive_normal_form

    At the moment, the explicit representation is used for simplicity / prototyping.

    It may be replaced by an ability of specifying selection with jq
    if operating on JSON metadata or else something free form like SQL.
    """

    parent_uids: Optional[Sequence[str]] = None
    uids: Optional[Sequence[str]] = None
    # Pick up all artifacts with the given tags.
    # Maybe we should call this transformations.
    tags: Optional[Sequence[str]] = None


class ArtifactStore(abc.ABC):
    """Use to keep track of artifacts generated while processing content.

    The first version of the artifact store is used to work with Documents
    rather than Blobs.

    We will likely want to evolve this into Blobs, but faster to prototype
    with Documents.
    """

    def exists_by_uid(self, uids: Sequence[str]) -> List[bool]:
        """Check if the artifacts with the given uuid exist."""
        raise NotImplementedError()

    def exists_by_parent_uids(self, uids: Sequence[str]) -> List[bool]:
        """Check if the artifacts with the given id exist."""
        raise NotImplementedError()

    def upsert(
        self,
        documents: Sequence[Document],
        # Find better way to propagate information about tags
        # this may be moved into a wrapper object
        # called: DocumentWithMetadata
        tags: Sequence[str],
    ) -> None:
        """Add the given artifacts."""
        raise NotImplementedError()

    def list_documents(self, selector: Selector) -> Iterator[Document]:
        """Yield documents matching the given selector."""
        raise NotImplementedError()

    def list_document_ids(self, selector: Selector) -> Iterator[str]:
        """Yield document ids matching the given selector."""
        raise NotImplementedError()

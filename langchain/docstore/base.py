"""Interface to access to place that stores documents."""
import abc
import dataclasses
from abc import ABC, abstractmethod
from typing import (
    Dict,
    Sequence,
    Iterator,
    Optional,
    List,
    Literal,
    TypedDict,
    Tuple,
    Union,
    Any,
)

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
    tags: Optional[Sequence[str]] = None  # <-- WE DONT WANT TO DO IT THIS WAY
    transformation_path: Sequence[str] = None
    """Use to specify a transformation path according to which we select documents"""


# KNOWN WAYS THIS CAN FAIL:
# 1) If the process crashes while text splitting, creating only some of the artifacts
#    ... new pipeline will not re-create the missing artifacts! (at least for now)
#    it will use the ones that exist and assume that all of them have been created


# TODO: MAJOR MAJOR MAJOR MAJOR
# 1. FIX SEMANTICS WITH REGARDS TO ID, UUID. AND POTENTIALLY ARTIFACT_ID
# NEED TO REASON THROUGH USE CASES CAREFULLY TO REASON ABOUT WHATS MINIMAL SUFFICIENT
# 2. Using hashes throughout for implementation simplicity, but may want to switch
# to ids assigned by the a database? probability of collision is really small
class Artifact(TypedDict):

    """A representation of an artifact."""

    uid: str  # This has to be handled carefully -- we'll eventually get collisions
    """A unique identifier for the artifact."""
    type_: Union[Literal["document"], Literal["embedding"], Literal["blob"]]
    """A unique identifier for the artifact."""
    data_hash: str
    """A hash of the data of the artifact."""
    metadata_hash: str
    """A hash of the metadata of the artifact."""
    parent_uids: Tuple[str, ...]
    """A tuple of uids representing the parent artifacts."""
    parent_hashes: Tuple[str, ...]
    """A tuple of hashes representing the parent artifacts at time of transformation."""
    transformation_hash: str
    """A hash of the transformation that was applied to generate artifact.
    
    This parameterizes the transformation logic together with any transformation
    parameters.
    """
    created_at: str  # ISO-8601
    """The time the artifact was created."""
    updated_at: str  # ISO-8601
    """The time the artifact was last updated."""
    metadata: Any
    """A dictionary representing the metadata of the artifact."""
    tags: Tuple[str, ...]
    """A tuple of tags associated with the artifact.
    
    Can use tags to add information about the transformation that was applied
    to the given artifact.
    
    THIS IS NOT A GOOD REPRESENTATION.
    """
    """The type of the artifact."""  # THIS MAY NEED TO BE CHANGED
    data: Optional[bytes]
    """The data of the artifact when the artifact contains the data by value.
    
    Will likely change somehow.
    
    * For first pass contains embedding data.
    * document data and blob data stored externally.
    """
    location: Optional[str]
    # Location specifies the location of the artifact when
    # the artifact contains the data by reference (use for documents / blobs)


class ArtifactWithData(TypedDict):
    """A document with the transformation that generated it."""

    artifact: Artifact
    document: Document


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
        artifacts_with_data: Sequence[ArtifactWithData],
    ) -> None:
        """Upsert the given artifacts."""
        raise NotImplementedError()

    def list_documents(self, selector: Selector) -> Iterator[Document]:
        """Yield documents matching the given selector."""
        raise NotImplementedError()

    def list_document_ids(self, selector: Selector) -> Iterator[str]:
        """Yield document ids matching the given selector."""
        raise NotImplementedError()

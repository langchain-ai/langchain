"""Proposal for a new schema for the indexing and retrieval interfaces.

This proposal has the following goals:

1. Have a well-defined READ/WRITE API for indexing.
    - Use upsert instead of add
    - Support upsert, delete and read by ID
2. Support indexing of binary data.
3. Have a well-defined query API on retrieval
    - Define filters, limit, offset, sort
    - Support similarity search
    - Enforcement can be done by standard unit-tests against the implementation.
    - Implementations can be required to either self-describe supported features
      or else just raise a NotImplementedError for unsupported features, so
      no silent failures occur.
    - We may need to throw in an extra field for disambiguating between
      a query executing using the `raw` interface vs. via a standardized langchain
      interface.

The query API should allow us to support adhoc capabilities like:
    - mmr search
    - setting min score threshold etc

-----

Discussion points:

* Force all content to have metadata OR only ID?
    - If it's only ID, then we don't support the ability to filter by metadata in the query API.
    Abstraction is more generic, and we can sub-class to add an interface that supports metadata operations.
* Base Indexing abstraction forced to support metadata?
* Support get_with_extras for indexers or vectorstores? (See below)

*. Two options for how to deal with the retriever interface:
    1. Have a standalone interface, and a vectorstore implements both indexer and retriever interfaces.
    2. The retriever interface is defined in the vectorstore abstraction itself (which inherits from indexing interface).

*. Need to determine if we try to add an API that will support sync primitives.
    - This is not needed for the initial version of the API.
    - Requires an indexed integer or datetime field to be updatable, and filterable
"""
import abc
from abc import ABC
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    NotRequired,
    Optional,
    Sequence,
    TypedDict,
    TypeVar,
    Union,
)

from langchain_core.load.serializable import Serializable
from langchain_core.runnables import RunnableConfig


class IndexableContent(
    ABC, Serializable
):  # Introduce new base class for content. Name OK?
    id: str
    metadata: Optional[dict]  # In a more extreme form we could remove


class Document(IndexableContent):  # Re-use existing Document model in langchain_core
    content: str


class Blob(IndexableContent):  # Re-use existing Blob model in langchain_core
    """Use for images / audio / video"""

    data: bytes
    encoding: str  # base64, utf-8, URI, etc.
    mimetype: str  # image/png, text/plain, etc.


# This is likely not needed during indexing right now
# class MultiContent(Content):
#     """Group multiple pieces of content together"""
#     contents: List[Content]
#     metadata: dict


T = TypeVar("T", bound=IndexableContent)


class UpsertResponse(TypedDict):
    """An indexing result."""

    succeeded: Sequence[str]
    failed: Sequence[str]


class DeleteResponse(TypedDict):
    """A response to a delete request."""

    num_deleted: NotRequired[int]
    num_failed: NotRequired[int]
    succeeded: NotRequired[Sequence[str]]
    failed: NotRequired[Sequence[str]]


class Sort(TypedDict):
    """A sort object."""

    field: str
    ascending: NotRequired[bool]  # Assume asc=True


C = TypeVar("C")


class BaseIndex(Generic[T]):
    """An index represent a collection of items that can be queried.

    The index is responsible for:

    1. Storing the content
    2. Allowing retrieval of the content by id
    3. Supporting queries against the metadata associated with the content

    The index is **NOT** responsible for:

    1. Support search queries against the content.

    The types of supported queries depends on the implementation (e.g., a vectorstore index)
    """

    @abc.abstractmethod
    def upsert(self, data: Iterable[T], /, **kwargs: Any) -> Iterable[UpsertResponse]:
        """Upsert a stream of data by id."""

    @abc.abstractmethod
    def get_by_ids(
        self,
        ids: Sequence[str],
        /,
    ) -> List[T]:
        """Get items by id."""

    @abc.abstractmethod
    def delete_by_ids(
        self,
        ids: Sequence[str],
        /,
        **kwargs: Any,
    ) -> DeleteResponse:
        """Delete an item by id."""

    # Delete and get are part of the READ/WRITE interface.
    # They do not take advantage of indexes on the content.
    # However, all the indexers ARE assumed to have the capability to index
    # on metadata if they implement the delete_by_query and get_by_query methods.
    @abc.abstractmethod
    def get_by_query(
        self,
        *,
        filter: Optional[Union[List[Dict[str, Any], Dict[str, Any]]]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        sort: Optional[Union[Sort, List[Sort]]] = None,
        **kwargs: Any,
    ) -> Iterable[T]:
        """Get items by query."""

    @abc.abstractmethod
    def delete_by_query(
        self,
        *,
        filter: Optional[Union[List[Dict[str, Any], Dict[str, Any]]]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        sort: Optional[Union[Sort, List[Sort]]] = None,
        **kwargs: Any,
    ) -> Iterable[DeleteResponse]:
        """Delete items by query."""
        # Careful with halloween problem


class Hit(Generic[T], TypedDict):
    # Should subclass from content?
    id: str
    score: float
    # Optional source
    source: Optional[T]
    metadata: Dict[str, Any]


class QueryResponse(TypedDict):
    """A retrieval result."""

    # Free form metadata for vectorstore providers
    metadata: Dict[str, Any]
    hits: List[Hit]


class UpsertDocument(TypedDict):
    """A structured representation of a data item to upsert.

    This allows providing additional information such as the vector.
    """

    id: str  # An override of the ID for the document
    document: Document
    vector: List[float]


class VectorStore(BaseIndex[Document]):
    @abc.abstractmethod
    def query(
        self,
        query: Union[str, List[float]],
        *,
        filter: Optional[Union[List[Dict[str, Any], Dict[str, Any]]]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        sort: Optional[Union[Sort, List[Sort]]] = None,
        search_method: Optional[str] = None,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> QueryResponse[Document]:
        """Query for items."""

    @abc.abstractmethod
    def upsert_with_vector(
        self,
        data_stream: Iterable[UpsertDocument],
        /,
        **kwargs: Any,
    ) -> Iterable[UpsertResponse]:
        """Upsert vectors by id."""


# -----------------------------
# Rely a bit on generics for parameterization

Q = TypeVar("Q")


class UpsertData(TypedDict, Generic[T]):
    """A structured representation of a data item to upsert.

    This allows providing additional information such as the vector.
    """

    content: T
    vector: List[float]


class VectorStoreGeneric(Generic[Q, T], BaseIndex[T]):
    @abc.abstractmethod
    def query(
        self, query: Q, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> QueryResponse[T]:
        """Query for items."""

    @abc.abstractmethod
    def upsert_with_vector(
        self, data_stream: Iterable[UpsertData[T]], /, **kwargs: Any
    ) -> Iterable[UpsertResponse]:
        """Upsert vectors by id."""

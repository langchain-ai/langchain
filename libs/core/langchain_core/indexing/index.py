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

from langchain_core.documents.base import BaseMedia
from langchain_core.indexing.base import UpsertResponse
from langchain_core.runnables import RunnableConfig

T = TypeVar("T", bound=BaseMedia)


class Sort(TypedDict):
    """A sort object."""

    field: str
    ascending: NotRequired[bool]  # Assume asc=True


C = TypeVar("C")


class BaseIndex(Generic[T]):
    """An index represent a collection of items that can be queried.

    An implementation of a BaseIndex should support the following operations:

    1. Storing the content
    2. Allowing retrieval of the content by id
    3. Supporting queries against the metadata associated with the content

    The implementation is **NOT** responsible for supporting
    search queries against the content itself!

    This responsibility is left for more specialized interfaces like VectorStore.
    """

    # Developer guidelines:
    # Do not override streaming_upsert!
    # This interface will likely be extended in the future with additional support
    # to deal with failures and retries.
    @beta(message="Added in 0.2.11. The API is subject to change.")
    def streaming_upsert(
        self, items: Iterable[T], /, batch_size: int, **kwargs: Any
    ) -> Iterator[UpsertResponse]:
        """Upsert documents in a streaming fashion.

        Args:
            items: Iterable of Documents to add to the vectorstore.
            batch_size: The size of each batch to upsert.
            **kwargs: Additional keyword arguments.
                kwargs should only include parameters that are common to all
                documents. (e.g., timeout for indexing, retry policy, etc.)
                kwargs should not include ids to avoid ambiguous semantics.
                Instead the ID should be provided as part of the Document object.

        .. versionadded:: 0.2.11
        """
        # The default implementation of this method breaks the input into
        # batches of size `batch_size` and calls the `upsert` method on each batch.
        # Subclasses can override this method to provide a more efficient
        # implementation.
        for item_batch in batch_iterate(batch_size, items):
            yield self.upsert(item_batch, **kwargs)

    # TODO(Eugene) Update documentation
    @abc.abstractmethod
    def upsert(self, items: Sequence[Document], /, **kwargs: Any) -> UpsertResponse:
        """Upsert items into the index.

        The upsert functionality should utilize the ID field of the Document object
        if it is provided. If the ID is not provided, the upsert method is free
        to generate an ID for the document.

        When an ID is specified and the document already exists in the vectorstore,
        the upsert method should update the document with the new data. If the document
        does not exist, the upsert method should add the document to the vectorstore.

        Args:
            items: Sequence of Documents to add to the vectorstore.
            **kwargs: Additional keyword arguments.

        Returns:
            UpsertResponse: A response object that contains the list of IDs that were
            successfully added or updated in the vectorstore and the list of IDs that
            failed to be added or updated.
        """

    @beta(message="Added in 0.2.11. The API is subject to change.")
    async def astreaming_upsert(
        self,
        items: AsyncIterable[Document],
        /,
        batch_size: int,
        **kwargs: Any,
    ) -> AsyncIterator[UpsertResponse]:
        """Upsert documents in a streaming fashion. Async version of streaming_upsert.

        Args:
            items: Iterable of Documents to add to the vectorstore.
            batch_size: The size of each batch to upsert.
            **kwargs: Additional keyword arguments.
                kwargs should only include parameters that are common to all
                documents. (e.g., timeout for indexing, retry policy, etc.)
                kwargs should not include ids to avoid ambiguous semantics.
                Instead the ID should be provided as part of the Document object.

        .. versionadded:: 0.2.11
        """
        async for batch in abatch_iterate(batch_size, items):
            yield await self.aupsert(batch, **kwargs)

    @beta(message="Added in 0.2.11. The API is subject to change.")
    async def aupsert(self, items: Sequence[T], /, **kwargs: Any) -> UpsertResponse:
        """Add or update documents in the vectorstore. Async version of upsert.

        The upsert functionality should utilize the ID field of the Document object
        if it is provided. If the ID is not provided, the upsert method is free
        to generate an ID for the document.

        When an ID is specified and the document already exists in the vectorstore,
        the upsert method should update the document with the new data. If the document
        does not exist, the upsert method should add the document to the vectorstore.

        Args:
            items: Sequence of Documents to add to the vectorstore.
            **kwargs: Additional keyword arguments.

        Returns:
            UpsertResponse: A response object that contains the list of IDs that were
            successfully added or updated in the vectorstore and the list of IDs that
            failed to be added or updated.

        .. versionadded:: 0.2.11
        """
        #  Developer guidelines: See guidelines for the `upsert` method.
        # The implementation does not delegate to the `add_texts` method or
        # the `add_documents` method by default since those implementations

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

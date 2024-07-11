"""A generic indexing interface for storing and querying content."""

import abc
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Generic,
    Iterable,
    Iterator,
    List,
    NotRequired,
    Sequence,
    TypedDict,
    TypeVar,
    Optional,
    Union,
    Dict,
)

from langchain_core._api import beta
from langchain_core.documents.base import BaseMedia
from langchain_core.indexing.base import DeleteResponse, UpsertResponse
from langchain_core.runnables import run_in_executor
from langchain_core.utils import abatch_iterate, batch_iterate


class Sort(TypedDict):
    """A sort object."""

    field: str
    ascending: NotRequired[bool]  # Assume asc=True


T = TypeVar("T", bound=BaseMedia)


class Query(TypedDict, total=False):
    """Standard query"""

    filter: Optional[Union[List[Dict[str, Any], Dict[str, Any]]]]
    limit: Optional[int]
    offset: Optional[int]
    sort: Optional[Union[Sort, List[Sort]]]


Q = TypeVar("Q", bound=Query)


class BaseIndex(Generic[T, Q]):
    """An index represent a collection of items that can be queried.

    This indexing interface is designed to be a generic abstraction for storing and
    querying content that has an ID and metadata associated with it.

    The interface is designed to be agnostic to the underlying implementation of the
    indexing system.

    The interface is designed to support the following operations:

    1. Storing content in the index.
    2. Retrieving content by ID.
    3. Querying the content based on the metadata associated with the content.

    The implementation is **NOT** responsible for supporting search queries
    against the content itself! Such a responsibility is left for more specialized
    interfaces like the VectorStore.

    While strongly encouraged, implementations are not required to support
    querying based on metadata. Such implementations override the `get_by_query`
    and `delete_by_query` methods to raise a NotImplementedError.
    """

    # Developer guidelines:
    # Do not override streaming_upsert!
    # This interface will likely be extended in the future with additional support
    # to deal with failures and retries.
    @beta(message="Added in 0.2.11. The API is subject to change.")
    def streaming_upsert(
        self, items: Iterable[T], /, batch_size: int, **kwargs: Any
    ) -> Iterator[UpsertResponse]:
        """Upsert items in a streaming fashion.

        Args:
            items: Iterable of content to add to the vectorstore.
            batch_size: The size of each batch to upsert.
            **kwargs: Additional keyword arguments.
                kwargs should only include parameters that are common to all
                items. (e.g., timeout for indexing, retry policy, etc.)
                kwargs should not include ids to avoid ambiguous semantics.
                Instead the ID should be provided as part of the item.

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
    def upsert(self, items: Sequence[T], /, **kwargs: Any) -> UpsertResponse:
        """Upsert items into the index.

        The upsert functionality should utilize the ID field of the content object
        if it is provided. If the ID is not provided, the upsert method is free
        to generate an ID for the content.

        When an ID is specified and the content already exists in the vectorstore,
        the upsert method should update the content with the new data. If the content
        does not exist, the upsert method should add the item to the vectorstore.

        Args:
            items: Sequence of items to add to the vectorstore.
            **kwargs: Additional keyword arguments.

        Returns:
            UpsertResponse: A response object that contains the list of IDs that were
            successfully added or updated in the vectorstore and the list of IDs that
            failed to be added or updated.
        """

    @beta(message="Added in 0.2.11. The API is subject to change.")
    async def astreaming_upsert(
        self,
        items: AsyncIterable[T],
        /,
        batch_size: int,
        **kwargs: Any,
    ) -> AsyncIterator[UpsertResponse]:
        """Upsert items in a streaming fashion. Async version of streaming_upsert.

        Args:
            items: Iterable of items to add to the vectorstore.
            batch_size: The size of each batch to upsert.
            **kwargs: Additional keyword arguments.
                kwargs should only include parameters that are common to all
                items. (e.g., timeout for indexing, retry policy, etc.)
                kwargs should not include ids to avoid ambiguous semantics.
                Instead the ID should be provided as part of the item object.

        .. versionadded:: 0.2.11
        """
        async for batch in abatch_iterate(batch_size, items):
            yield await self.aupsert(batch, **kwargs)

    @beta(message="Added in 0.2.11. The API is subject to change.")
    async def aupsert(self, items: Sequence[T], /, **kwargs: Any) -> UpsertResponse:
        """Add or update items in the vectorstore. Async version of upsert.

        The upsert functionality should utilize the ID field of the item
        if it is provided. If the ID is not provided, the upsert method is free
        to generate an ID for the item.

        When an ID is specified and the item already exists in the vectorstore,
        the upsert method should update the item with the new data. If the item
        does not exist, the upsert method should add the item to the vectorstore.

        Args:
            items: Sequence of items to add to the vectorstore.
            **kwargs: Additional keyword arguments.

        Returns:
            UpsertResponse: A response object that contains the list of IDs that were
            successfully added or updated in the vectorstore and the list of IDs that
            failed to be added or updated.

        .. versionadded:: 0.2.11
        """
        return await run_in_executor(
            None,
            self.upsert,
            items,
            **kwargs,
        )

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
        """Delete items by id."""

    # Delete and get are part of the READ/WRITE interface.
    # They do not take advantage of indexes on the content.
    # However, all the indexers ARE assumed to have the capability to index
    # on metadata if they implement the delete_by_query and get_by_query methods.
    @abc.abstractmethod
    def get_by_query(
        self,
        query: Q,
        /,
        **kwargs: Any,
    ) -> Iterable[T]:
        """Get items by query."""

    @abc.abstractmethod
    def delete_by_query(
        self,
        query: Q,
        /,
        **kwargs: Any,
    ) -> Iterable[DeleteResponse]:
        """Delete items by query."""

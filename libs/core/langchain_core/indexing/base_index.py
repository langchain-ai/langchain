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
    Literal,
)

from langchain_core._api import beta
from langchain_core.documents.base import BaseMedia
from langchain_core.indexing.base import DeleteResponse, UpsertResponse
from langchain_core.runnables import run_in_executor
from langchain_core.utils import abatch_iterate, batch_iterate


class Sort(TypedDict):
    """Sort order for the results."""

    field: str
    """The field to sort by."""
    ascending: NotRequired[bool]
    """Sort order. True for ascending, False for descending.
    
    If missing, the default sort order is ascending.
    """


# Need to compare against supported filtering operators
Comparator = Literal[
    "$eq", "$ne", "$lt", "$lte", "$gt", "$gte", "$in", "$nin", "$exists"
]
Operator = Literal["$and", "$or", "$not"]


class Description(TypedDict, total=False):
    """Description of the index."""

    supported_comparators: List[Comparator]  # Set to [] if filtering is not supported
    supported_operators: List[Operator]  # Set to [] if filtering is not supported
    supports_sort: bool
    supports_pagination: bool


T = TypeVar("T", bound=BaseMedia)
Q = TypeVar("Q")


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

    .. versionadded:: 0.2.15
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

    @beta(message="Added in 0.2.15. The API is subject to change.")
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

        .. versionadded:: 0.2.15
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

    @beta(message="Added in 0.2.15. The API is subject to change.")
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

        .. versionadded:: 0.2.15
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
        **kwargs: Any,
    ) -> List[T]:
        """Get items by id.

        Fewer items may be returned than requested if some IDs are not found or
        if there are duplicated IDs.

        Users should not assume that the order of the returned items matches
        the order of the input IDs. Instead, users should rely on the ID field of the
        returned items.

        This method should **NOT** raise exceptions if no items are found for
        some IDs.

        Args:
            ids: List of IDs to get.
            kwargs: Additional keyword arguments. These are up to the implementation.

        Returns:
            List[T]: List of items that were found.
        """

    @abc.abstractmethod
    def delete(
        self,
        ids: Sequence[str],
        /,
        **kwargs: Any,
    ) -> Union[DeleteResponse, bool]:
        """Delete by IDs or other criteria.

        Args:
            ids: List of ids to delete.
            kwargs: Additional keyword arguments. This is up to the implementation.

        Returns:
            DeleteResponse: A response object that contains the list of IDs that were
            successfully deleted and the list of IDs that failed to be deleted.

            OR

            bool: A boolean indicating whether the delete operation was successful.
        """

    # Delete and get are part of the READ/WRITE interface.
    # They do not take advantage of indexes on the content.
    # However, all the indexers ARE assumed to have the capability to index
    # on metadata if they implement the get_by_filter and delete_by_filter methods.
    @abc.abstractmethod
    def get_by_filter(
        self,
        *,
        filter: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        limit: Optional[int] = None,
        sort: Optional[Sort] = None,
        **kwargs: Any,
    ) -> Iterable[T]:
        """Get items by a filter query.

        Args:
            filter: A filter to apply to the query. Must be a valid filter.
                Expected to follow the standard LangChain filtering syntax.
            limit: Number of items to return.
            sort: Sort order for the results if supported by the index.
            **kwargs: Additional keyword arguments.
        """
        # Developer guidelines
        # 1. The filter should be a dictionary or a list of dictionaries.
        # 2. An invalid filter should raise an exception.
        # 3. A None filter is considered valid and should return items.
        # 4. The **default** filter syntax should follow standard LangChain
        #    filtering syntax.
        #    The syntax is as follows:
        #    - All operators and comparators should be prefixed with a "$".
        #    - Field names are expected to be valid identifiers allowing [a-zA-Z0-9_]
        #      only.
        #    - Top level dict with multiple keys should be treated as an "$and" query.
        #    - Top level list should be treated as an "$and" query.
        #    - A key that starts with "$" should be treated as an operator or comparator
        #      (e.g., "$and", "$or", "$not", "$eq", "$ne", "$lt", "$lte", "$gt", "$gte",
        #    - A key that is not prefixed with "$" should be treated as a field name.
        # 5. Supported filtering operators should be documented in the description
        #   of the index.
        # 6. Providers are free to support **additional** types of filtering operators
        #    to do that, they should define the filter as
        #    Union[existing_format, provider_format]
        #    the provider format should contain an extra `type`
        #    field, so that it could be distinguished from the standard format.
        #    We suggest for the type value to be "provider". The rest of the syntax is
        #    up to the provider to define.
        #
        #    For example:
        #    {
        #        "type": "provider",
        #        "filter": "and(or(eq('field', 'value'), eq('field2', 'value2')))"A
        #    }

    @abc.abstractmethod
    def delete_by_filter(
        self,
        filter: Union[Dict[str, Any], List[Dict[str, Any]]],
        /,
        **kwargs: Any,
    ) -> DeleteResponse:
        """Delete items by a filter.

        Args:
            filter: A filter to apply to the query. Must be a valid filter.
                Expected to follow the standard LangChain filtering syntax.
            **kwargs: Additional keyword arguments.

        Returns:
            Iterable[DeleteResponse]: An iterable of delete responses.
        """
        # Developer guidelines:
        # 1. The filter should be a dictionary or a list of dictionaries.
        # 2. An invalid filter should raise an exception.
        # 3. An empty filter is considered invalid and should raise an exception.
        # 4. The **default** filter syntax should follow standard LangChain
        #    filtering syntax.
        #    The syntax is as follows:
        #    - All operators and comparators should be prefixed with a "$".
        #    - Field names are expected to be valid identifiers allowing [a-zA-Z0-9_]
        #      only.
        #    - Top level dict with multiple keys should be treated as an "$and" query.
        #    - Top level list should be treated as an "$and" query.
        #    - A key that starts with "$" should be treated as an operator or comparator
        #      (e.g., "$and", "$or", "$not", "$eq", "$ne", "$lt", "$lte", "$gt", "$gte",
        #    - A key that is not prefixed with "$" should be treated as a field name.
        # 5. Supported filtering operators should be documented in the description
        #   of the index.
        # 6. Providers are free to support **additional** types of filtering operators
        #    to do that, they should define the filter as
        #    Union[existing_format, provider_format]
        #    the provider format should contain an extra `type`
        #    field, so that it could be distinguished from the standard format.
        #    We suggest for the type value to be "provider". The rest of the syntax is
        #    up to the provider to define.
        #
        #    For example:
        #    {
        #        "type": "provider",
        #        "filter": "and(or(eq('field', 'value'), eq('field2', 'value2')))"A
        #    }

    @classmethod
    @beta(message="Added in 0.2.15. The API is subject to change.")
    def describe(cls) -> Description:
        """Get a description of the functionality supported by the index."""
        # Developer guidelines:
        # Developers are encouraged to override this method to provide a
        # detailed description of the functionality supported by the index.
        # The description will be used in the following manners:
        # 1. Surfaced in the documentation to provide users with an overview of
        #    the functionality supported by the index.
        # 2. Used by standard test suites to verify that the index actually supports
        #    the functionality it claims to support correctly.
        # 3. By you, the developer, to leverage utility code that will be used to
        #    provide run-time validation of user provided queries.
        # 4. Will be accessible to users in an interactive environment to help them
        #    understand the capabilities of the index.
        raise NotImplementedError(f"{cls.__name__} does not implement describe method.")

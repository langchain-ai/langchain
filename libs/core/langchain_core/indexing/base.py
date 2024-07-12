from __future__ import annotations

import abc
import time
from abc import ABC, abstractmethod
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

from typing_extensions import NotRequired, TypedDict

from langchain_core._api import beta
from langchain_core.documents.base import BaseMedia
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


# The list of supported filter operations will need to be updated as new standard
# operators are added to the filtering syntax.
FilterOp = Literal[
    "$eq", "$ne", "$lt", "$lte", "$gt", "$gte", "$in", "$nin", "$and", "$not", "$or"
]


class Description(TypedDict, total=False):
    """Description of the index."""

    supported_filter_operations: List[FilterOp]
    """A list of the operators and comparators supported by the index."""
    supports_sort: bool
    """Whether the index supports sorting on a single metadata field."""


T = TypeVar("T", bound=BaseMedia)


class UpsertResponse(TypedDict):
    """A generic response for upsert operations.

    The upsert response will be used by abstractions that implement an upsert
    operation for content that can be upserted by ID.

    Upsert APIs that accept inputs with IDs and generate IDs internally
    will return a response that includes the IDs that succeeded and the IDs
    that failed.

    If there are no failures, the failed list will be empty, and the order
    of the IDs in the succeeded list will match the order of the input documents.

    If there are failures, the response becomes ill defined, and a user of the API
    cannot determine which generated ID corresponds to which input document.

    It is recommended for users explicitly attach the IDs to the items being
    indexed to avoid this issue.
    """

    succeeded: List[str]
    """The IDs that were successfully indexed."""
    failed: List[str]
    """The IDs that failed to index."""


class DeleteResponse(TypedDict, total=False):
    """A generic response for delete operation.

    The fields in this response are optional and whether the vectorstore
    returns them or not is up to the implementation.
    """

    num_deleted: int
    """The number of items that were successfully deleted."""
    num_failed: int
    """The number of items that failed to be deleted."""
    succeeded: Sequence[str]
    """The IDs that were successfully deleted.
    
    Should not be returned when using delete_by_filter.
    """
    failed: Sequence[str]
    """The IDs that failed to be deleted.
    
    Should not be returned when using delete_by_filter.
    
    Please note that deleting an ID that does not exist is **NOT** considered a failure.
    """


class BaseIndex(Generic[T]):
    """A collection of items that can be queried.

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

    @abc.abstractmethod
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
        self, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> Union[DeleteResponse, bool, None]:
        """Delete by IDs or other criteria.

        Args:
            ids: List of ids to delete.
            kwargs: Additional keyword arguments. This is up to the implementation.

        Returns:
            DeleteResponse: A response object that contains the list of IDs that were
            successfully deleted and the list of IDs that failed to be deleted.

            OR

            bool: A boolean indicating whether the delete operation was successful.

            New implementations should return DeleteResponse instead of bool, but
            older implementations may end up returning bool.
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
        #        "filter": "and(or(eq('field', 'value'), eq('field2', 'value2')))"
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


class RecordManager(ABC):
    """Abstract base class representing the interface for a record manager.

    The record manager abstraction is used by the langchain indexing API.

    The record manager keeps track of which documents have been
    written into a vectorstore and when they were written.

    The indexing API computes hashes for each document and stores the hash
    together with the write time and the source id in the record manager.

    On subsequent indexing runs, the indexing API can check the record manager
    to determine which documents have already been indexed and which have not.

    This allows the indexing API to avoid re-indexing documents that have
    already been indexed, and to only index new documents.

    The main benefit of this abstraction is that it works across many vectorstores.
    To be supported, a vectorstore needs to only support the ability to add and
    delete documents by ID. Using the record manager, the indexing API will
    be able to delete outdated documents and avoid redundant indexing of documents
    that have already been indexed.

    The main constraints of this abstraction are:

    1. It relies on the time-stamps to determine which documents have been
       indexed and which have not. This means that the time-stamps must be
       monotonically increasing. The timestamp should be the timestamp
       as measured by the server to minimize issues.
    2. The record manager is currently implemented separately from the
       vectorstore, which means that the overall system becomes distributed
       and may create issues with consistency. For example, writing to
       record manager succeeds, but corresponding writing to vectorstore fails.
    """

    def __init__(
        self,
        namespace: str,
    ) -> None:
        """Initialize the record manager.

        Args:
            namespace (str): The namespace for the record manager.
        """
        self.namespace = namespace

    @abstractmethod
    def create_schema(self) -> None:
        """Create the database schema for the record manager."""

    @abstractmethod
    async def acreate_schema(self) -> None:
        """Asynchronously create the database schema for the record manager."""

    @abstractmethod
    def get_time(self) -> float:
        """Get the current server time as a high resolution timestamp!

        It's important to get this from the server to ensure a monotonic clock,
        otherwise there may be data loss when cleaning up old documents!

        Returns:
            The current server time as a float timestamp.
        """

    @abstractmethod
    async def aget_time(self) -> float:
        """Asynchronously get the current server time as a high resolution timestamp.

        It's important to get this from the server to ensure a monotonic clock,
        otherwise there may be data loss when cleaning up old documents!

        Returns:
            The current server time as a float timestamp.
        """

    @abstractmethod
    def update(
        self,
        keys: Sequence[str],
        *,
        group_ids: Optional[Sequence[Optional[str]]] = None,
        time_at_least: Optional[float] = None,
    ) -> None:
        """Upsert records into the database.

        Args:
            keys: A list of record keys to upsert.
            group_ids: A list of group IDs corresponding to the keys.
            time_at_least: Optional timestamp. Implementation can use this
                to optionally verify that the timestamp IS at least this time
                in the system that stores the data.

                e.g., use to validate that the time in the postgres database
                is equal to or larger than the given timestamp, if not
                raise an error.

                This is meant to help prevent time-drift issues since
                time may not be monotonically increasing!

        Raises:
            ValueError: If the length of keys doesn't match the length of group_ids.
        """

    @abstractmethod
    async def aupdate(
        self,
        keys: Sequence[str],
        *,
        group_ids: Optional[Sequence[Optional[str]]] = None,
        time_at_least: Optional[float] = None,
    ) -> None:
        """Asynchronously upsert records into the database.

        Args:
            keys: A list of record keys to upsert.
            group_ids: A list of group IDs corresponding to the keys.
            time_at_least: Optional timestamp. Implementation can use this
                to optionally verify that the timestamp IS at least this time
                in the system that stores the data.

                e.g., use to validate that the time in the postgres database
                is equal to or larger than the given timestamp, if not
                raise an error.

                This is meant to help prevent time-drift issues since
                time may not be monotonically increasing!

        Raises:
            ValueError: If the length of keys doesn't match the length of group_ids.
        """

    @abstractmethod
    def exists(self, keys: Sequence[str]) -> List[bool]:
        """Check if the provided keys exist in the database.

        Args:
            keys: A list of keys to check.

        Returns:
            A list of boolean values indicating the existence of each key.
        """

    @abstractmethod
    async def aexists(self, keys: Sequence[str]) -> List[bool]:
        """Asynchronously check if the provided keys exist in the database.

        Args:
            keys: A list of keys to check.

        Returns:
            A list of boolean values indicating the existence of each key.
        """

    @abstractmethod
    def list_keys(
        self,
        *,
        before: Optional[float] = None,
        after: Optional[float] = None,
        group_ids: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        """List records in the database based on the provided filters.

        Args:
            before: Filter to list records updated before this time.
            after: Filter to list records updated after this time.
            group_ids: Filter to list records with specific group IDs.
            limit: optional limit on the number of records to return.

        Returns:
            A list of keys for the matching records.
        """

    @abstractmethod
    async def alist_keys(
        self,
        *,
        before: Optional[float] = None,
        after: Optional[float] = None,
        group_ids: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        """Asynchronously list records in the database based on the provided filters.

        Args:
            before: Filter to list records updated before this time.
            after: Filter to list records updated after this time.
            group_ids: Filter to list records with specific group IDs.
            limit: optional limit on the number of records to return.

        Returns:
            A list of keys for the matching records.
        """

    @abstractmethod
    def delete_keys(self, keys: Sequence[str]) -> None:
        """Delete specified records from the database.

        Args:
            keys: A list of keys to delete.
        """

    @abstractmethod
    async def adelete_keys(self, keys: Sequence[str]) -> None:
        """Asynchronously delete specified records from the database.

        Args:
            keys: A list of keys to delete.
        """


class _Record(TypedDict):
    group_id: Optional[str]
    updated_at: float


class InMemoryRecordManager(RecordManager):
    """An in-memory record manager for testing purposes."""

    def __init__(self, namespace: str) -> None:
        """Initialize the in-memory record manager.

        Args:
            namespace (str): The namespace for the record manager.
        """
        super().__init__(namespace)
        # Each key points to a dictionary
        # of {'group_id': group_id, 'updated_at': timestamp}
        self.records: Dict[str, _Record] = {}
        self.namespace = namespace

    def create_schema(self) -> None:
        """In-memory schema creation is simply ensuring the structure is initialized."""

    async def acreate_schema(self) -> None:
        """Async in-memory schema creation is simply ensuring
        the structure is initialized.
        """

    def get_time(self) -> float:
        """Get the current server time as a high resolution timestamp!"""
        return time.time()

    async def aget_time(self) -> float:
        """Async get the current server time as a high resolution timestamp!"""
        return self.get_time()

    def update(
        self,
        keys: Sequence[str],
        *,
        group_ids: Optional[Sequence[Optional[str]]] = None,
        time_at_least: Optional[float] = None,
    ) -> None:
        """Upsert records into the database.

        Args:
            keys: A list of record keys to upsert.
            group_ids: A list of group IDs corresponding to the keys.
                Defaults to None.
            time_at_least: Optional timestamp. Implementation can use this
                to optionally verify that the timestamp IS at least this time
                in the system that stores. Defaults to None.
                E.g., use to validate that the time in the postgres database
                is equal to or larger than the given timestamp, if not
                raise an error.
                This is meant to help prevent time-drift issues since
                time may not be monotonically increasing!

        Raises:
            ValueError: If the length of keys doesn't match the length of group
                ids.
            ValueError: If time_at_least is in the future.
        """

        if group_ids and len(keys) != len(group_ids):
            raise ValueError("Length of keys must match length of group_ids")
        for index, key in enumerate(keys):
            group_id = group_ids[index] if group_ids else None
            if time_at_least and time_at_least > self.get_time():
                raise ValueError("time_at_least must be in the past")
            self.records[key] = {"group_id": group_id, "updated_at": self.get_time()}

    async def aupdate(
        self,
        keys: Sequence[str],
        *,
        group_ids: Optional[Sequence[Optional[str]]] = None,
        time_at_least: Optional[float] = None,
    ) -> None:
        """Async upsert records into the database.

        Args:
            keys: A list of record keys to upsert.
            group_ids: A list of group IDs corresponding to the keys.
                Defaults to None.
            time_at_least: Optional timestamp. Implementation can use this
                to optionally verify that the timestamp IS at least this time
                in the system that stores. Defaults to None.
                E.g., use to validate that the time in the postgres database
                is equal to or larger than the given timestamp, if not
                raise an error.
                This is meant to help prevent time-drift issues since
                time may not be monotonically increasing!

        Raises:
            ValueError: If the length of keys doesn't match the length of group
                ids.
            ValueError: If time_at_least is in the future.
        """
        self.update(keys, group_ids=group_ids, time_at_least=time_at_least)

    def exists(self, keys: Sequence[str]) -> List[bool]:
        """Check if the provided keys exist in the database.

        Args:
            keys: A list of keys to check.

        Returns:
            A list of boolean values indicating the existence of each key.
        """
        return [key in self.records for key in keys]

    async def aexists(self, keys: Sequence[str]) -> List[bool]:
        """Async check if the provided keys exist in the database.

        Args:
            keys: A list of keys to check.

        Returns:
            A list of boolean values indicating the existence of each key.
        """
        return self.exists(keys)

    def list_keys(
        self,
        *,
        before: Optional[float] = None,
        after: Optional[float] = None,
        group_ids: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        """List records in the database based on the provided filters.

        Args:
            before: Filter to list records updated before this time.
                Defaults to None.
            after: Filter to list records updated after this time.
                Defaults to None.
            group_ids: Filter to list records with specific group IDs.
                Defaults to None.
            limit: optional limit on the number of records to return.
                Defaults to None.

        Returns:
            A list of keys for the matching records.
        """
        result = []
        for key, data in self.records.items():
            if before and data["updated_at"] >= before:
                continue
            if after and data["updated_at"] <= after:
                continue
            if group_ids and data["group_id"] not in group_ids:
                continue
            result.append(key)
        if limit:
            return result[:limit]
        return result

    async def alist_keys(
        self,
        *,
        before: Optional[float] = None,
        after: Optional[float] = None,
        group_ids: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        """Async list records in the database based on the provided filters.

        Args:
            before: Filter to list records updated before this time.
                Defaults to None.
            after: Filter to list records updated after this time.
                Defaults to None.
            group_ids: Filter to list records with specific group IDs.
                Defaults to None.
            limit: optional limit on the number of records to return.
                Defaults to None.

        Returns:
            A list of keys for the matching records.
        """
        return self.list_keys(
            before=before, after=after, group_ids=group_ids, limit=limit
        )

    def delete_keys(self, keys: Sequence[str]) -> None:
        """Delete specified records from the database.

        Args:
            keys: A list of keys to delete.
        """
        for key in keys:
            if key in self.records:
                del self.records[key]

    async def adelete_keys(self, keys: Sequence[str]) -> None:
        """Async delete specified records from the database.

        Args:
            keys: A list of keys to delete.
        """
        self.delete_keys(keys)

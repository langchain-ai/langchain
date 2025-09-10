"""Base classes for indexing."""

from __future__ import annotations

import abc
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, TypedDict

from typing_extensions import override

from langchain_core._api import beta
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import run_in_executor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.documents import Document


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
    def exists(self, keys: Sequence[str]) -> list[bool]:
        """Check if the provided keys exist in the database.

        Args:
            keys: A list of keys to check.

        Returns:
            A list of boolean values indicating the existence of each key.
        """

    @abstractmethod
    async def aexists(self, keys: Sequence[str]) -> list[bool]:
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
    ) -> list[str]:
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
    ) -> list[str]:
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
        self.records: dict[str, _Record] = {}
        self.namespace = namespace

    def create_schema(self) -> None:
        """In-memory schema creation is simply ensuring the structure is initialized."""

    async def acreate_schema(self) -> None:
        """In-memory schema creation is simply ensuring the structure is initialized."""

    @override
    def get_time(self) -> float:
        return time.time()

    @override
    async def aget_time(self) -> float:
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
            msg = "Length of keys must match length of group_ids"
            raise ValueError(msg)
        for index, key in enumerate(keys):
            group_id = group_ids[index] if group_ids else None
            if time_at_least and time_at_least > self.get_time():
                msg = "time_at_least must be in the past"
                raise ValueError(msg)
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
        """
        self.update(keys, group_ids=group_ids, time_at_least=time_at_least)

    def exists(self, keys: Sequence[str]) -> list[bool]:
        """Check if the provided keys exist in the database.

        Args:
            keys: A list of keys to check.

        Returns:
            A list of boolean values indicating the existence of each key.
        """
        return [key in self.records for key in keys]

    async def aexists(self, keys: Sequence[str]) -> list[bool]:
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
    ) -> list[str]:
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
    ) -> list[str]:
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

    succeeded: list[str]
    """The IDs that were successfully indexed."""
    failed: list[str]
    """The IDs that failed to index."""


class DeleteResponse(TypedDict, total=False):
    """A generic response for delete operation.

    The fields in this response are optional and whether the vectorstore
    returns them or not is up to the implementation.
    """

    num_deleted: int
    """The number of items that were successfully deleted.

    If returned, this should only include *actual* deletions.

    If the ID did not exist to begin with,
    it should not be included in this count.
    """

    succeeded: Sequence[str]
    """The IDs that were successfully deleted.

    If returned, this should only include *actual* deletions.

    If the ID did not exist to begin with,
    it should not be included in this list.
    """

    failed: Sequence[str]
    """The IDs that failed to be deleted.

    .. warning::
        Deleting an ID that does not exist is **NOT** considered a failure.
    """

    num_failed: int
    """The number of items that failed to be deleted."""


@beta(message="Added in 0.2.29. The abstraction is subject to change.")
class DocumentIndex(BaseRetriever):
    """A document retriever that supports indexing operations.

    This indexing interface is designed to be a generic abstraction for storing and
    querying documents that has an ID and metadata associated with it.

    The interface is designed to be agnostic to the underlying implementation of the
    indexing system.

    The interface is designed to support the following operations:

    1. Storing document in the index.
    2. Fetching document by ID.
    3. Searching for document using a query.

    .. versionadded:: 0.2.29
    """

    @abc.abstractmethod
    def upsert(self, items: Sequence[Document], /, **kwargs: Any) -> UpsertResponse:
        """Upsert documents into the index.

        The upsert functionality should utilize the ID field of the content object
        if it is provided. If the ID is not provided, the upsert method is free
        to generate an ID for the content.

        When an ID is specified and the content already exists in the vectorstore,
        the upsert method should update the content with the new data. If the content
        does not exist, the upsert method should add the item to the vectorstore.

        Args:
            items: Sequence of documents to add to the vectorstore.
            **kwargs: Additional keyword arguments.

        Returns:
            UpsertResponse: A response object that contains the list of IDs that were
            successfully added or updated in the vectorstore and the list of IDs that
            failed to be added or updated.
        """

    async def aupsert(
        self, items: Sequence[Document], /, **kwargs: Any
    ) -> UpsertResponse:
        """Add or update documents in the vectorstore. Async version of upsert.

        The upsert functionality should utilize the ID field of the item
        if it is provided. If the ID is not provided, the upsert method is free
        to generate an ID for the item.

        When an ID is specified and the item already exists in the vectorstore,
        the upsert method should update the item with the new data. If the item
        does not exist, the upsert method should add the item to the vectorstore.

        Args:
            items: Sequence of documents to add to the vectorstore.
            **kwargs: Additional keyword arguments.

        Returns:
            UpsertResponse: A response object that contains the list of IDs that were
            successfully added or updated in the vectorstore and the list of IDs that
            failed to be added or updated.
        """
        return await run_in_executor(
            None,
            self.upsert,
            items,
            **kwargs,
        )

    @abc.abstractmethod
    def delete(self, ids: Optional[list[str]] = None, **kwargs: Any) -> DeleteResponse:
        """Delete by IDs or other criteria.

        Calling delete without any input parameters should raise a ValueError!

        Args:
            ids: List of ids to delete.
            kwargs: Additional keyword arguments. This is up to the implementation.
                For example, can include an option to delete the entire index,
                or else issue a non-blocking delete etc.

        Returns:
            DeleteResponse: A response object that contains the list of IDs that were
            successfully deleted and the list of IDs that failed to be deleted.
        """

    async def adelete(
        self, ids: Optional[list[str]] = None, **kwargs: Any
    ) -> DeleteResponse:
        """Delete by IDs or other criteria. Async variant.

        Calling adelete without any input parameters should raise a ValueError!

        Args:
            ids: List of ids to delete.
            kwargs: Additional keyword arguments. This is up to the implementation.
                For example, can include an option to delete the entire index.

        Returns:
            DeleteResponse: A response object that contains the list of IDs that were
            successfully deleted and the list of IDs that failed to be deleted.
        """
        return await run_in_executor(
            None,
            self.delete,
            ids,
            **kwargs,
        )

    @abc.abstractmethod
    def get(
        self,
        ids: Sequence[str],
        /,
        **kwargs: Any,
    ) -> list[Document]:
        """Get documents by id.

        Fewer documents may be returned than requested if some IDs are not found or
        if there are duplicated IDs.

        Users should not assume that the order of the returned documents matches
        the order of the input IDs. Instead, users should rely on the ID field of the
        returned documents.

        This method should **NOT** raise exceptions if no documents are found for
        some IDs.

        Args:
            ids: List of IDs to get.
            kwargs: Additional keyword arguments. These are up to the implementation.

        Returns:
            list[Document]: List of documents that were found.
        """

    async def aget(
        self,
        ids: Sequence[str],
        /,
        **kwargs: Any,
    ) -> list[Document]:
        """Get documents by id.

        Fewer documents may be returned than requested if some IDs are not found or
        if there are duplicated IDs.

        Users should not assume that the order of the returned documents matches
        the order of the input IDs. Instead, users should rely on the ID field of the
        returned documents.

        This method should **NOT** raise exceptions if no documents are found for
        some IDs.

        Args:
            ids: List of IDs to get.
            kwargs: Additional keyword arguments. These are up to the implementation.

        Returns:
            list[Document]: List of documents that were found.
        """
        return await run_in_executor(
            None,
            self.get,
            ids,
            **kwargs,
        )

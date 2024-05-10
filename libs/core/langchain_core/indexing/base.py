from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence


class RecordManager(ABC):
    """Abstract base class representing the interface for a record manager."""

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

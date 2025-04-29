"""Cassandra-based chat message history, based on cassIO."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence

from langchain_community.utilities.cassandra import SetupMode

if TYPE_CHECKING:
    from cassandra.cluster import Session
    from cassio.table.table_types import RowType

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)

DEFAULT_TABLE_NAME = "message_store"
DEFAULT_TTL_SECONDS = None


def _rows_to_messages(rows: Iterable[RowType]) -> List[BaseMessage]:
    message_blobs = [row["body_blob"] for row in rows][::-1]
    items = [json.loads(message_blob) for message_blob in message_blobs]
    messages = messages_from_dict(items)
    return messages


class CassandraChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that is backed by Cassandra."""

    def __init__(
        self,
        session_id: str,
        session: Optional[Session] = None,
        keyspace: Optional[str] = None,
        table_name: str = DEFAULT_TABLE_NAME,
        ttl_seconds: Optional[int] = DEFAULT_TTL_SECONDS,
        *,
        setup_mode: SetupMode = SetupMode.SYNC,
    ) -> None:
        """
        Initialize a new instance of CassandraChatMessageHistory.

        Args:
            session_id: arbitrary key that is used to store the messages
                of a single chat session.
            session: Cassandra driver session.
                If not provided, it is resolved from cassio.
            keyspace: Cassandra key space. If not provided, it is resolved from cassio.
            table_name: name of the table to use.
            ttl_seconds: time-to-live (seconds) for automatic expiration
                of stored entries. None (default) for no expiration.
            setup_mode: mode used to create the Cassandra table (SYNC, ASYNC or OFF).
        """
        try:
            from cassio.table import ClusteredCassandraTable
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "Could not import cassio python package. "
                "Please install it with `pip install cassio`."
            )
        self.session_id = session_id
        self.ttl_seconds = ttl_seconds
        kwargs: Dict[str, Any] = {}
        if setup_mode == SetupMode.ASYNC:
            kwargs["async_setup"] = True
        self.table = ClusteredCassandraTable(
            session=session,
            keyspace=keyspace,
            table=table_name,
            ttl_seconds=ttl_seconds,
            primary_key_type=["TEXT", "TIMEUUID"],
            ordering_in_partition="DESC",
            skip_provisioning=setup_mode == SetupMode.OFF,
            **kwargs,
        )

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore[override]
        """Retrieve all session messages from DB"""
        # The latest are returned, in chronological order
        rows = self.table.get_partition(
            partition_id=self.session_id,
        )
        return _rows_to_messages(rows)

    async def aget_messages(self) -> List[BaseMessage]:
        """Retrieve all session messages from DB"""
        # The latest are returned, in chronological order
        rows = await self.table.aget_partition(
            partition_id=self.session_id,
        )
        return _rows_to_messages(rows)

    def add_message(self, message: BaseMessage) -> None:
        """Write a message to the table

        Args:
            message: A message to write.
        """
        this_row_id = uuid.uuid4()
        self.table.put(
            partition_id=self.session_id,
            row_id=this_row_id,
            body_blob=json.dumps(message_to_dict(message)),
            ttl_seconds=self.ttl_seconds,
        )

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        for message in messages:
            this_row_id = uuid.uuid4()
            await self.table.aput(
                partition_id=self.session_id,
                row_id=this_row_id,
                body_blob=json.dumps(message_to_dict(message)),
                ttl_seconds=self.ttl_seconds,
            )

    def clear(self) -> None:
        """Clear session memory from DB"""
        self.table.delete_partition(self.session_id)

    async def aclear(self) -> None:
        """Clear session memory from DB"""
        await self.table.adelete_partition(self.session_id)

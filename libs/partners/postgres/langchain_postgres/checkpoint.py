"""Implementation of a langgraph checkpoint saver using Postgres."""
import abc
import pickle
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, AsyncIterator, Generator, Optional, Union

import psycopg
from langchain_core.runnables import ConfigurableFieldSpec, RunnableConfig
from langgraph.checkpoint import BaseCheckpointSaver
from langgraph.checkpoint.base import Checkpoint, CheckpointThreadTs, CheckpointTuple
from psycopg_pool import AsyncConnectionPool, ConnectionPool


class Serializer(abc.ABC):
    """A serializer for serializing and deserializing objects to and from bytes."""

    @staticmethod
    @abc.abstractmethod
    def dumps(obj: object) -> bytes:
        """Serialize an object to bytes."""

    @staticmethod
    @abc.abstractmethod
    def loads(data: bytes) -> object:
        """Deserialize an object from bytes."""


class PickleSerializer(Serializer):
    """Use the pickle module to serialize and deserialize objects.

    This serializer uses the pickle module to serialize and deserialize objects.

    While pickling can serialize a wide range of Python objects, it may fail
    de-serializable objects upon updates of the Python version or the python
    environment (e.g., the object's class definition changes in LangGraph).

    *Security Warning*: The pickle module can deserialize malicious payloads,
        only use this serializer with trusted data; e.g., data that you
        have serialized yourself and can guarantee the integrity of.
    """

    @staticmethod
    def dumps(obj: object) -> bytes:
        """Serialize an object to bytes."""
        return pickle.dumps(obj)

    @staticmethod
    def loads(data: bytes) -> object:
        """Deserialize an object from bytes."""
        return pickle.loads(data)


class PostgresCheckpoint(BaseCheckpointSaver):
    """Implementation of a langgraph checkpoint saver for Postgres.

    This is a reference implementation of a checkpoint saver for Postgres.

    The implementation provides some functionality to create the necessary table
    and schema for the checkpoint saver, as well as to get and put checkpoints.

    Examples:

        .. code-block:: python

            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
            from langchain_postgres.checkpoint import PostgresCheckpoint

            import psycopg
    """

    serializer: Serializer
    """The serializer for serializing and deserializing objects to and from bytes."""

    sync_connection: Optional[Union[psycopg.Connection, ConnectionPool]] = None
    """The synchronous connection or pool to the Postgres database.
    
    If providing a connection object, please ensure that the connection is open
    and remember to close the connection when done.
    """
    async_connection: Optional[
        Union[psycopg.AsyncConnection, AsyncConnectionPool]
    ] = None
    """The asynchronous connection or pool to the Postgres database.
    
    If providing a connection object, please ensure that the connection is open
    and remember to close the connection when done.
    """

    class Config:
        arbitrary_types_allowed = True

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        """Return the configuration specs for this runnable."""
        return [
            ConfigurableFieldSpec(
                id="thread_id",
                annotation=Optional[str],
                name="Thread ID",
                description=None,
                default=None,
                is_shared=True,
            ),
            CheckpointThreadTs,
        ]

    def get(self, config: RunnableConfig) -> Optional[Checkpoint]:
        """Get the checkpoint for the given configuration."""
        raise NotImplementedError

    @contextmanager
    def _get_connection(self) -> Generator[psycopg.Connection, None, None]:
        """Get the connection to the Postgres database."""
        if isinstance(self.sync_connection, psycopg.Connection):
            yield self.sync_connection
        elif isinstance(self.sync_connection, ConnectionPool):
            with self.sync_connection.connection() as conn:
                yield conn
        else:
            raise ValueError(
                "Invalid sync connection object. Please initialize the check pointer "
                f"with an appropriate sync connection object. "
                f"Got {type(self.sync_connection)}."
            )

    @asynccontextmanager
    async def _get_async_connection(self) -> AsyncGenerator[psycopg.Connection, None]:
        """Get the connection to the Postgres database."""
        if isinstance(self.async_connection, psycopg.AsyncConnection):
            yield self.async_connection
        elif isinstance(self.async_connection, AsyncConnectionPool):
            async with self.async_connection.connection() as conn:
                yield conn
        else:
            raise ValueError(
                "Invalid async connection object. Please initialize the check pointer "
                f"with an appropriate async connection object. "
                f"Got {type(self.async_connection)}."
            )

    @staticmethod
    async def acreate_schema(connection: psycopg.AsyncConnection, /) -> None:
        """Create the schema for the checkpoint saver."""
        async with connection.cursor() as cur:
            await cur.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT NOT NULL,
                    checkpoint BYTEA NOT NULL,
                    thread_ts TIMESTAMPTZ NOT NULL,
                    parent_ts TIMESTAMPTZ,
                    PRIMARY KEY (thread_id, thread_ts)
                );
                """
            )

    @staticmethod
    async def adrop_table(connection: psycopg.AsyncConnection, /) -> None:
        """Drop the table for the checkpoint saver."""
        async with connection.cursor() as cur:
            await cur.execute("DROP TABLE IF EXISTS checkpoints;")

    def put(self, config: RunnableConfig, checkpoint: Checkpoint) -> RunnableConfig:
        """Put the checkpoint for the given configuration."""
        raise NotImplementedError

    async def alist(self, config: RunnableConfig) -> AsyncIterator[CheckpointTuple]:
        """Get all the checkpoints for the given configuration."""
        async with self._get_async_connection() as conn:
            async with conn.cursor() as cur:
                thread_id = config["configurable"]["thread_id"]
                await cur.execute(
                    "SELECT checkpoint, thread_ts, parent_ts "
                    "FROM checkpoints "
                    "WHERE thread_id = %(thread_id)s "
                    "ORDER BY thread_ts DESC",
                    {
                        "thread_id": thread_id,
                    },
                )
                async for value in cur:
                    yield CheckpointTuple(
                        {
                            "configurable": {
                                "thread_id": thread_id,
                                "thread_ts": value[1],
                            }
                        },
                        pickle.loads(value[0]),
                        {
                            "configurable": {
                                "thread_id": thread_id,
                                "thread_ts": value[2],
                            }
                        }
                        if value[2]
                        else None,
                    )

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get the checkpoint tuple for the given configuration.

        Args:
            config: The configuration for the checkpoint.
                A dict with a `configurable` key which is a dict with
                a `thread_id` key and an optional `thread_ts` key.
                For example, { 'configurable': { 'thread_id': 'test_thread' } }

        Returns:
            The checkpoint tuple for the given configuration if it exists,
            otherwise None.

            If thread_ts is None, the latest checkpoint is returned if it exists.
        """
        thread_id = config["configurable"]["thread_id"]
        thread_ts = config["configurable"].get("thread_ts")
        async with self._get_async_connection() as conn:
            async with conn.cursor() as cur:
                if thread_ts:
                    await cur.execute(
                        "SELECT checkpoint, parent_ts "
                        "FROM checkpoints "
                        "WHERE thread_id = %(thread_id)s AND thread_ts = %(thread_ts)s",
                        {
                            "thread_id": thread_id,
                            "thread_ts": thread_ts,
                        },
                    )
                    value = await cur.fetchone()
                    if value:
                        return CheckpointTuple(
                            config,
                            pickle.loads(value[0]),
                            {
                                "configurable": {
                                    "thread_id": thread_id,
                                    "thread_ts": value[1].isoformat(),
                                }
                            }
                            if value[1]
                            else None,
                        )
                else:
                    await cur.execute(
                        "SELECT checkpoint, thread_ts, parent_ts "
                        "FROM checkpoints "
                        "WHERE thread_id = %(thread_id)s "
                        "ORDER BY thread_ts DESC LIMIT 1",
                        {
                            "thread_id": thread_id,
                        },
                    )
                    value = await cur.fetchone()
                    if value:
                        return CheckpointTuple(
                            config={
                                "configurable": {
                                    "thread_id": thread_id,
                                    "thread_ts": value[1].isoformat(),
                                }
                            },
                            checkpoint=pickle.loads(value[0]),
                            parent_config={
                                "configurable": {
                                    "thread_id": thread_id,
                                    "thread_ts": value[2].isoformat(),
                                }
                            }
                            if value[2]
                            else None,
                        )

    async def aput(
        self, config: RunnableConfig, checkpoint: Checkpoint
    ) -> RunnableConfig:
        """Put the checkpoint for the given configuration.

        Args:
            config: The configuration for the checkpoint.
                A dict with a `configurable` key which is a dict with
                a `thread_id` key and an optional `thread_ts` key.
                For example, { 'configurable': { 'thread_id': 'test_thread' } }
            checkpoint: That was saved.

        Returns:
            The configuration for the checkpoint.
        """
        thread_id = config["configurable"]["thread_id"]
        parent_ts = config["configurable"].get("thread_ts")
        async with self._get_async_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO checkpoints (thread_id, thread_ts, parent_ts, checkpoint)
                    VALUES (%(thread_id)s, %(thread_ts)s, %(parent_ts)s, %(checkpoint)s)
                    ON CONFLICT (thread_id, thread_ts) 
                    DO UPDATE SET checkpoint = EXCLUDED.checkpoint;
                    """,
                    {
                        "thread_id": thread_id,
                        "thread_ts": checkpoint["ts"],
                        "parent_ts": parent_ts if parent_ts else None,
                        "checkpoint": pickle.dumps(checkpoint),
                    },
                )

        return {
            **config,  # TODO(Nuno): Confirm
            "configurable": {
                "thread_id": thread_id,
                "thread_ts": checkpoint["ts"],
            },
        }

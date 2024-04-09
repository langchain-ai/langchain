"""Implementation of a langgraph checkpoint saver using Postgres."""
import abc
import pickle
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, AsyncIterator, Generator, Optional, Union, cast

import psycopg
from langchain_core.runnables import ConfigurableFieldSpec, RunnableConfig
from langgraph.checkpoint import BaseCheckpointSaver
from langgraph.checkpoint.base import Checkpoint, CheckpointThreadTs, CheckpointTuple
from psycopg_pool import AsyncConnectionPool, ConnectionPool


class CheckpointSerializer(abc.ABC):
    """A serializer for serializing and deserializing objects to and from bytes."""

    @abc.abstractmethod
    def dumps(self, obj: Checkpoint) -> bytes:
        """Serialize an object to bytes."""

    @abc.abstractmethod
    def loads(self, data: bytes) -> Checkpoint:
        """Deserialize an object from bytes."""


class PickleCheckpointSerializer(CheckpointSerializer):
    """Use the pickle module to serialize and deserialize objects.

    This serializer uses the pickle module to serialize and deserialize objects.

    While pickling can serialize a wide range of Python objects, it may fail
    de-serializable objects upon updates of the Python version or the python
    environment (e.g., the object's class definition changes in LangGraph).

    *Security Warning*: The pickle module can deserialize malicious payloads,
        only use this serializer with trusted data; e.g., data that you
        have serialized yourself and can guarantee the integrity of.
    """

    def dumps(self, obj: Checkpoint) -> bytes:
        """Serialize an object to bytes."""
        return pickle.dumps(obj)

    def loads(self, data: bytes) -> Checkpoint:
        """Deserialize an object from bytes."""
        return cast(Checkpoint, pickle.loads(data))


class PostgresCheckpoint(BaseCheckpointSaver):
    """LangGraph checkpoint saver for Postgres.

    This implementation of a checkpoint saver uses a Postgres database to save
    and retrieve checkpoints. It uses the psycopg3 package to interact with the
    Postgres database.

    The checkpoint accepts either a sync_connection in the form of a psycopg.Connection
    or a psycopg.ConnectionPool object, or an async_connection in the form of a
    psycopg.AsyncConnection or psycopg.AsyncConnectionPool object.

    Usage:

    1. First time use: create schema in the database using the `create_schema` method or
       the async version `acreate_schema` method.
    2. Create a PostgresCheckpoint object with a serializer and an appropriate
       connection object.
       It's recommended to use a connection pool object for the connection.
       If using a connection object, you are responsible for closing the connection
       when done.

    Examples:


    Sync usage with a connection pool:

        .. code-block:: python

            from psycopg_pool import ConnectionPool
            from langchain_postgres import (
                PostgresCheckpoint, PickleCheckpointSerializer
            )

            pool = ConnectionPool(
                # Example configuration
                conninfo="postgresql://user:password@localhost:5432/dbname",
                max_size=20,
            )

            # Uses the pickle module for serialization
            # Make sure that you're only de-serializing trusted data
            # (e.g., payloads that you have serialized yourself).
            # Or implement a custom serializer.
            checkpoint = PostgresCheckpoint(
                serializer=PickleCheckpointSerializer(),
                sync_connection=pool,
            )

            # Use the checkpoint object to put, get, list checkpoints, etc.


    Async usage with a connection pool:

        .. code-block:: python

            from psycopg_pool import AsyncConnectionPool
            from langchain_postgres import (
                PostgresCheckpoint, PickleCheckpointSerializer
            )

            pool = AsyncConnectionPool(
                # Example configuration
                conninfo="postgresql://user:password@localhost:5432/dbname",
                max_size=20,
            )

            # Uses the pickle module for serialization
            # Make sure that you're only de-serializing trusted data
            # (e.g., payloads that you have serialized yourself).
            # Or implement a custom serializer.
            checkpoint = PostgresCheckpoint(
                serializer=PickleCheckpointSerializer(),
                async_connection=pool,
            )

            # Use the checkpoint object to put, get, list checkpoints, etc.


    Async usage with a connection object:

        .. code-block:: python

            from psycopg import AsyncConnection
            from langchain_postgres import (
                PostgresCheckpoint, PickleCheckpointSerializer
            )

            conninfo="postgresql://user:password@localhost:5432/dbname"
            # Take care of closing the connection when done
            async with AsyncConnection(conninfo=conninfo) as conn:
                # Uses the pickle module for serialization
                # Make sure that you're only de-serializing trusted data
                # (e.g., payloads that you have serialized yourself).
                # Or implement a custom serializer.
                checkpoint = PostgresCheckpoint(
                    serializer=PickleCheckpointSerializer(),
                    async_connection=conn,
                )

                # Use the checkpoint object to put, get, list checkpoints, etc.
                ...
    """

    serializer: CheckpointSerializer
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
        extra = "forbid"

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

    @contextmanager
    def _get_sync_connection(self) -> Generator[psycopg.Connection, None, None]:
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
    async def _get_async_connection(
        self,
    ) -> AsyncGenerator[psycopg.AsyncConnection, None]:
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
    def create_schema(connection: psycopg.Connection, /) -> None:
        """Create the schema for the checkpoint saver."""
        with connection.cursor() as cur:
            cur.execute(
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
    def drop_schema(connection: psycopg.Connection, /) -> None:
        """Drop the table for the checkpoint saver."""
        with connection.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS checkpoints;")

    @staticmethod
    async def adrop_schema(connection: psycopg.AsyncConnection, /) -> None:
        """Drop the table for the checkpoint saver."""
        async with connection.cursor() as cur:
            await cur.execute("DROP TABLE IF EXISTS checkpoints;")

    def put(self, config: RunnableConfig, checkpoint: Checkpoint) -> RunnableConfig:
        """Put the checkpoint for the given configuration.

        Args:
            config: The configuration for the checkpoint.
                A dict with a `configurable` key which is a dict with
                a `thread_id` key and an optional `thread_ts` key.
                For example, { 'configurable': { 'thread_id': 'test_thread' } }
            checkpoint: The checkpoint to persist.

        Returns:
            The RunnableConfig that describes the checkpoint that was just created.
            It'll contain the `thread_id` and `thread_ts` of the checkpoint.
        """
        thread_id = config["configurable"]["thread_id"]
        parent_ts = config["configurable"].get("thread_ts")

        with self._get_sync_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO checkpoints 
                        (thread_id, thread_ts, parent_ts, checkpoint)
                    VALUES 
                        (%(thread_id)s, %(thread_ts)s, %(parent_ts)s, %(checkpoint)s)
                    ON CONFLICT (thread_id, thread_ts) 
                    DO UPDATE SET checkpoint = EXCLUDED.checkpoint;
                    """,
                    {
                        "thread_id": thread_id,
                        "thread_ts": checkpoint["ts"],
                        "parent_ts": parent_ts if parent_ts else None,
                        "checkpoint": self.serializer.dumps(checkpoint),
                    },
                )

        return {
            "configurable": {
                "thread_id": thread_id,
                "thread_ts": checkpoint["ts"],
            },
        }

    async def aput(
        self, config: RunnableConfig, checkpoint: Checkpoint
    ) -> RunnableConfig:
        """Put the checkpoint for the given configuration.

        Args:
            config: The configuration for the checkpoint.
                A dict with a `configurable` key which is a dict with
                a `thread_id` key and an optional `thread_ts` key.
                For example, { 'configurable': { 'thread_id': 'test_thread' } }
            checkpoint: The checkpoint to persist.

        Returns:
            The RunnableConfig that describes the checkpoint that was just created.
            It'll contain the `thread_id` and `thread_ts` of the checkpoint.
        """
        thread_id = config["configurable"]["thread_id"]
        parent_ts = config["configurable"].get("thread_ts")
        async with self._get_async_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO 
                        checkpoints (thread_id, thread_ts, parent_ts, checkpoint)
                    VALUES 
                        (%(thread_id)s, %(thread_ts)s, %(parent_ts)s, %(checkpoint)s)
                    ON CONFLICT (thread_id, thread_ts) 
                    DO UPDATE SET checkpoint = EXCLUDED.checkpoint;
                    """,
                    {
                        "thread_id": thread_id,
                        "thread_ts": checkpoint["ts"],
                        "parent_ts": parent_ts if parent_ts else None,
                        "checkpoint": self.serializer.dumps(checkpoint),
                    },
                )

        return {
            "configurable": {
                "thread_id": thread_id,
                "thread_ts": checkpoint["ts"],
            },
        }

    def list(self, config: RunnableConfig) -> Generator[CheckpointTuple, None, None]:
        """Get all the checkpoints for the given configuration."""
        with self._get_sync_connection() as conn:
            with conn.cursor() as cur:
                thread_id = config["configurable"]["thread_id"]
                cur.execute(
                    "SELECT checkpoint, thread_ts, parent_ts "
                    "FROM checkpoints "
                    "WHERE thread_id = %(thread_id)s "
                    "ORDER BY thread_ts DESC",
                    {
                        "thread_id": thread_id,
                    },
                )
                for value in cur:
                    yield CheckpointTuple(
                        {
                            "configurable": {
                                "thread_id": thread_id,
                                "thread_ts": value[1].isoformat(),
                            }
                        },
                        self.serializer.loads(value[0]),
                        {
                            "configurable": {
                                "thread_id": thread_id,
                                "thread_ts": value[2].isoformat(),
                            }
                        }
                        if value[2]
                        else None,
                    )

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
                                "thread_ts": value[1].isoformat(),
                            }
                        },
                        self.serializer.loads(value[0]),
                        {
                            "configurable": {
                                "thread_id": thread_id,
                                "thread_ts": value[2].isoformat(),
                            }
                        }
                        if value[2]
                        else None,
                    )

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
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
        with self._get_sync_connection() as conn:
            with conn.cursor() as cur:
                if thread_ts:
                    cur.execute(
                        "SELECT checkpoint, parent_ts "
                        "FROM checkpoints "
                        "WHERE thread_id = %(thread_id)s AND thread_ts = %(thread_ts)s",
                        {
                            "thread_id": thread_id,
                            "thread_ts": thread_ts,
                        },
                    )
                    value = cur.fetchone()
                    if value:
                        return CheckpointTuple(
                            config,
                            self.serializer.loads(value[0]),
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
                    cur.execute(
                        "SELECT checkpoint, thread_ts, parent_ts "
                        "FROM checkpoints "
                        "WHERE thread_id = %(thread_id)s "
                        "ORDER BY thread_ts DESC LIMIT 1",
                        {
                            "thread_id": thread_id,
                        },
                    )
                    value = cur.fetchone()
                    if value:
                        return CheckpointTuple(
                            config={
                                "configurable": {
                                    "thread_id": thread_id,
                                    "thread_ts": value[1].isoformat(),
                                }
                            },
                            checkpoint=self.serializer.loads(value[0]),
                            parent_config={
                                "configurable": {
                                    "thread_id": thread_id,
                                    "thread_ts": value[2].isoformat(),
                                }
                            }
                            if value[2]
                            else None,
                        )
        return None

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
                            self.serializer.loads(value[0]),
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
                            checkpoint=self.serializer.loads(value[0]),
                            parent_config={
                                "configurable": {
                                    "thread_id": thread_id,
                                    "thread_ts": value[2].isoformat(),
                                }
                            }
                            if value[2]
                            else None,
                        )

        return None

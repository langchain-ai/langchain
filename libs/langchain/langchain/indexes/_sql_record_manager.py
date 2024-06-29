"""Implementation of a record management layer in SQLAlchemy.

The management layer uses SQLAlchemy to track upserted records.

Currently, this layer only works with SQLite; hopwever, should be adaptable
to other SQL implementations with minimal effort.

Currently, includes an implementation that uses SQLAlchemy which should
allow it to work with a variety of SQL as a backend.

* Each key is associated with an updated_at field.
* This filed is updated whenever the key is updated.
* Keys can be listed based on the updated at field.
* Keys can be deleted.
"""

import contextlib
import decimal
import uuid
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Sequence, Union

from langchain_core.indexing import RecordManager
from sqlalchemy import (
    Column,
    Float,
    Index,
    String,
    UniqueConstraint,
    and_,
    create_engine,
    delete,
    select,
    text,
)
from sqlalchemy.engine import URL, Engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    create_async_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Query, Session, sessionmaker

try:
    from sqlalchemy.ext.asyncio import async_sessionmaker
except ImportError:
    # dummy for sqlalchemy < 2
    async_sessionmaker = type("async_sessionmaker", (type,), {})  # type: ignore

Base = declarative_base()


class UpsertionRecord(Base):  # type: ignore[valid-type,misc]
    """Table used to keep track of when a key was last updated."""

    # ATTENTION:
    # Prior to modifying this table, please determine whether
    # we should create migrations for this table to make sure
    # users do not experience data loss.
    __tablename__ = "upsertion_record"

    uuid = Column(
        String,
        index=True,
        default=lambda: str(uuid.uuid4()),
        primary_key=True,
        nullable=False,
    )
    key = Column(String, index=True)
    # Using a non-normalized representation to handle `namespace` attribute.
    # If the need arises, this attribute can be pulled into a separate Collection
    # table at some time later.
    namespace = Column(String, index=True, nullable=False)
    group_id = Column(String, index=True, nullable=True)

    # The timestamp associated with the last record upsertion.
    updated_at = Column(Float, index=True)

    __table_args__ = (
        UniqueConstraint("key", "namespace", name="uix_key_namespace"),
        Index("ix_key_namespace", "key", "namespace"),
    )


class SQLRecordManager(RecordManager):
    """A SQL Alchemy based implementation of the record manager."""

    def __init__(
        self,
        namespace: str,
        *,
        engine: Optional[Union[Engine, AsyncEngine]] = None,
        db_url: Union[None, str, URL] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        async_mode: bool = False,
    ) -> None:
        """Initialize the SQLRecordManager.

        This class serves as a manager persistence layer that uses an SQL
        backend to track upserted records. You should specify either a db_url
        to create an engine or provide an existing engine.

        Args:
            namespace: The namespace associated with this record manager.
            engine: An already existing SQL Alchemy engine.
                Default is None.
            db_url: A database connection string used to create
                an SQL Alchemy engine. Default is None.
            engine_kwargs: Additional keyword arguments
                to be passed when creating the engine. Default is an empty dictionary.
            async_mode: Whether to create an async engine.
                Driver should support async operations.
                It only applies if db_url is provided.
                Default is False.

        Raises:
            ValueError: If both db_url and engine are provided or neither.
            AssertionError: If something unexpected happens during engine configuration.
        """
        super().__init__(namespace=namespace)
        if db_url is None and engine is None:
            raise ValueError("Must specify either db_url or engine")

        if db_url is not None and engine is not None:
            raise ValueError("Must specify either db_url or engine, not both")

        _engine: Union[Engine, AsyncEngine]
        if db_url:
            if async_mode:
                _engine = create_async_engine(db_url, **(engine_kwargs or {}))
            else:
                _engine = create_engine(db_url, **(engine_kwargs or {}))
        elif engine:
            _engine = engine

        else:
            raise AssertionError("Something went wrong with configuration of engine.")

        _session_factory: Union[sessionmaker[Session], async_sessionmaker[AsyncSession]]
        if isinstance(_engine, AsyncEngine):
            _session_factory = async_sessionmaker(bind=_engine)
        else:
            _session_factory = sessionmaker(bind=_engine)

        self.engine = _engine
        self.dialect = _engine.dialect.name
        self.session_factory = _session_factory

    def create_schema(self) -> None:
        """Create the database schema."""
        if isinstance(self.engine, AsyncEngine):
            raise AssertionError("This method is not supported for async engines.")

        Base.metadata.create_all(self.engine)

    async def acreate_schema(self) -> None:
        """Create the database schema."""

        if not isinstance(self.engine, AsyncEngine):
            raise AssertionError("This method is not supported for sync engines.")

        async with self.engine.begin() as session:
            await session.run_sync(Base.metadata.create_all)

    @contextlib.contextmanager
    def _make_session(self) -> Generator[Session, None, None]:
        """Create a session and close it after use."""

        if isinstance(self.session_factory, async_sessionmaker):
            raise AssertionError("This method is not supported for async engines.")

        session = self.session_factory()
        try:
            yield session
        finally:
            session.close()

    @contextlib.asynccontextmanager
    async def _amake_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Create a session and close it after use."""

        if not isinstance(self.session_factory, async_sessionmaker):
            raise AssertionError("This method is not supported for sync engines.")

        async with self.session_factory() as session:
            yield session

    def get_time(self) -> float:
        """Get the current server time as a timestamp.

        Please note it's critical that time is obtained from the server since
        we want a monotonic clock.
        """
        with self._make_session() as session:
            # * SQLite specific implementation, can be changed based on dialect.
            # * For SQLite, unlike unixepoch it will work with older versions of SQLite.
            # ----
            # julianday('now'): Julian day number for the current date and time.
            # The Julian day is a continuous count of days, starting from a
            # reference date (Julian day number 0).
            # 2440587.5 - constant represents the Julian day number for January 1, 1970
            # 86400.0 - constant represents the number of seconds
            # in a day (24 hours * 60 minutes * 60 seconds)
            if self.dialect == "sqlite":
                query = text("SELECT (julianday('now') - 2440587.5) * 86400.0;")
            elif self.dialect == "postgresql":
                query = text("SELECT EXTRACT (EPOCH FROM CURRENT_TIMESTAMP);")
            else:
                raise NotImplementedError(f"Not implemented for dialect {self.dialect}")

            dt = session.execute(query).scalar()
            if isinstance(dt, decimal.Decimal):
                dt = float(dt)
            if not isinstance(dt, float):
                raise AssertionError(f"Unexpected type for datetime: {type(dt)}")
            return dt

    async def aget_time(self) -> float:
        """Get the current server time as a timestamp.

        Please note it's critical that time is obtained from the server since
        we want a monotonic clock.
        """
        async with self._amake_session() as session:
            # * SQLite specific implementation, can be changed based on dialect.
            # * For SQLite, unlike unixepoch it will work with older versions of SQLite.
            # ----
            # julianday('now'): Julian day number for the current date and time.
            # The Julian day is a continuous count of days, starting from a
            # reference date (Julian day number 0).
            # 2440587.5 - constant represents the Julian day number for January 1, 1970
            # 86400.0 - constant represents the number of seconds
            # in a day (24 hours * 60 minutes * 60 seconds)
            if self.dialect == "sqlite":
                query = text("SELECT (julianday('now') - 2440587.5) * 86400.0;")
            elif self.dialect == "postgresql":
                query = text("SELECT EXTRACT (EPOCH FROM CURRENT_TIMESTAMP);")
            else:
                raise NotImplementedError(f"Not implemented for dialect {self.dialect}")

            dt = (await session.execute(query)).scalar_one_or_none()

            if isinstance(dt, decimal.Decimal):
                dt = float(dt)
            if not isinstance(dt, float):
                raise AssertionError(f"Unexpected type for datetime: {type(dt)}")
            return dt

    def update(
        self,
        keys: Sequence[str],
        *,
        group_ids: Optional[Sequence[Optional[str]]] = None,
        time_at_least: Optional[float] = None,
    ) -> None:
        """Upsert records into the SQLite database."""
        if group_ids is None:
            group_ids = [None] * len(keys)

        if len(keys) != len(group_ids):
            raise ValueError(
                f"Number of keys ({len(keys)}) does not match number of "
                f"group_ids ({len(group_ids)})"
            )

        # Get the current time from the server.
        # This makes an extra round trip to the server, should not be a big deal
        # if the batch size is large enough.
        # Getting the time here helps us compare it against the time_at_least
        # and raise an error if there is a time sync issue.
        # Here, we're just being extra careful to minimize the chance of
        # data loss due to incorrectly deleting records.
        update_time = self.get_time()

        if time_at_least and update_time < time_at_least:
            # Safeguard against time sync issues
            raise AssertionError(f"Time sync issue: {update_time} < {time_at_least}")

        records_to_upsert = [
            {
                "key": key,
                "namespace": self.namespace,
                "updated_at": update_time,
                "group_id": group_id,
            }
            for key, group_id in zip(keys, group_ids)
        ]

        with self._make_session() as session:
            if self.dialect == "sqlite":
                from sqlalchemy.dialects.sqlite import Insert as SqliteInsertType
                from sqlalchemy.dialects.sqlite import insert as sqlite_insert

                # Note: uses SQLite insert to make on_conflict_do_update work.
                # This code needs to be generalized a bit to work with more dialects.
                sqlite_insert_stmt: SqliteInsertType = sqlite_insert(
                    UpsertionRecord
                ).values(records_to_upsert)
                stmt = sqlite_insert_stmt.on_conflict_do_update(
                    [UpsertionRecord.key, UpsertionRecord.namespace],
                    set_=dict(
                        updated_at=sqlite_insert_stmt.excluded.updated_at,
                        group_id=sqlite_insert_stmt.excluded.group_id,
                    ),
                )
            elif self.dialect == "postgresql":
                from sqlalchemy.dialects.postgresql import Insert as PgInsertType
                from sqlalchemy.dialects.postgresql import insert as pg_insert

                # Note: uses postgresql insert to make on_conflict_do_update work.
                # This code needs to be generalized a bit to work with more dialects.
                pg_insert_stmt: PgInsertType = pg_insert(UpsertionRecord).values(
                    records_to_upsert
                )
                stmt = pg_insert_stmt.on_conflict_do_update(  # type: ignore[assignment]
                    "uix_key_namespace",  # Name of constraint
                    set_=dict(
                        updated_at=pg_insert_stmt.excluded.updated_at,
                        group_id=pg_insert_stmt.excluded.group_id,
                    ),
                )
            else:
                raise NotImplementedError(f"Unsupported dialect {self.dialect}")

            session.execute(stmt)
            session.commit()

    async def aupdate(
        self,
        keys: Sequence[str],
        *,
        group_ids: Optional[Sequence[Optional[str]]] = None,
        time_at_least: Optional[float] = None,
    ) -> None:
        """Upsert records into the SQLite database."""
        if group_ids is None:
            group_ids = [None] * len(keys)

        if len(keys) != len(group_ids):
            raise ValueError(
                f"Number of keys ({len(keys)}) does not match number of "
                f"group_ids ({len(group_ids)})"
            )

        # Get the current time from the server.
        # This makes an extra round trip to the server, should not be a big deal
        # if the batch size is large enough.
        # Getting the time here helps us compare it against the time_at_least
        # and raise an error if there is a time sync issue.
        # Here, we're just being extra careful to minimize the chance of
        # data loss due to incorrectly deleting records.
        update_time = await self.aget_time()

        if time_at_least and update_time < time_at_least:
            # Safeguard against time sync issues
            raise AssertionError(f"Time sync issue: {update_time} < {time_at_least}")

        records_to_upsert = [
            {
                "key": key,
                "namespace": self.namespace,
                "updated_at": update_time,
                "group_id": group_id,
            }
            for key, group_id in zip(keys, group_ids)
        ]

        async with self._amake_session() as session:
            if self.dialect == "sqlite":
                from sqlalchemy.dialects.sqlite import Insert as SqliteInsertType
                from sqlalchemy.dialects.sqlite import insert as sqlite_insert

                # Note: uses SQLite insert to make on_conflict_do_update work.
                # This code needs to be generalized a bit to work with more dialects.
                sqlite_insert_stmt: SqliteInsertType = sqlite_insert(
                    UpsertionRecord
                ).values(records_to_upsert)
                stmt = sqlite_insert_stmt.on_conflict_do_update(
                    [UpsertionRecord.key, UpsertionRecord.namespace],
                    set_=dict(
                        updated_at=sqlite_insert_stmt.excluded.updated_at,
                        group_id=sqlite_insert_stmt.excluded.group_id,
                    ),
                )
            elif self.dialect == "postgresql":
                from sqlalchemy.dialects.postgresql import Insert as PgInsertType
                from sqlalchemy.dialects.postgresql import insert as pg_insert

                # Note: uses SQLite insert to make on_conflict_do_update work.
                # This code needs to be generalized a bit to work with more dialects.
                pg_insert_stmt: PgInsertType = pg_insert(UpsertionRecord).values(
                    records_to_upsert
                )
                stmt = pg_insert_stmt.on_conflict_do_update(  # type: ignore[assignment]
                    "uix_key_namespace",  # Name of constraint
                    set_=dict(
                        updated_at=pg_insert_stmt.excluded.updated_at,
                        group_id=pg_insert_stmt.excluded.group_id,
                    ),
                )
            else:
                raise NotImplementedError(f"Unsupported dialect {self.dialect}")

            await session.execute(stmt)
            await session.commit()

    def exists(self, keys: Sequence[str]) -> List[bool]:
        """Check if the given keys exist in the SQLite database."""
        session: Session
        with self._make_session() as session:
            filtered_query: Query = session.query(UpsertionRecord.key).filter(
                and_(
                    UpsertionRecord.key.in_(keys),
                    UpsertionRecord.namespace == self.namespace,
                )
            )
            records = filtered_query.all()
        found_keys = set(r.key for r in records)
        return [k in found_keys for k in keys]

    async def aexists(self, keys: Sequence[str]) -> List[bool]:
        """Check if the given keys exist in the SQLite database."""
        async with self._amake_session() as session:
            records = (
                (
                    await session.execute(
                        select(UpsertionRecord.key).where(
                            and_(
                                UpsertionRecord.key.in_(keys),
                                UpsertionRecord.namespace == self.namespace,
                            )
                        )
                    )
                )
                .scalars()
                .all()
            )
        found_keys = set(records)
        return [k in found_keys for k in keys]

    def list_keys(
        self,
        *,
        before: Optional[float] = None,
        after: Optional[float] = None,
        group_ids: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        """List records in the SQLite database based on the provided date range."""
        session: Session
        with self._make_session() as session:
            query: Query = session.query(UpsertionRecord).filter(
                UpsertionRecord.namespace == self.namespace
            )

            if after:
                query = query.filter(UpsertionRecord.updated_at > after)
            if before:
                query = query.filter(UpsertionRecord.updated_at < before)
            if group_ids:
                query = query.filter(UpsertionRecord.group_id.in_(group_ids))

            if limit:
                query = query.limit(limit)
            records = query.all()
        return [r.key for r in records]

    async def alist_keys(
        self,
        *,
        before: Optional[float] = None,
        after: Optional[float] = None,
        group_ids: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        """List records in the SQLite database based on the provided date range."""
        session: AsyncSession
        async with self._amake_session() as session:
            query: Query = select(UpsertionRecord.key).filter(  # type: ignore[assignment]
                UpsertionRecord.namespace == self.namespace
            )

            # mypy does not recognize .all() or .filter()
            if after:
                query = query.filter(UpsertionRecord.updated_at > after)
            if before:
                query = query.filter(UpsertionRecord.updated_at < before)
            if group_ids:
                query = query.filter(UpsertionRecord.group_id.in_(group_ids))

            if limit:
                query = query.limit(limit)
            records = (await session.execute(query)).scalars().all()
        return list(records)

    def delete_keys(self, keys: Sequence[str]) -> None:
        """Delete records from the SQLite database."""
        session: Session
        with self._make_session() as session:
            filtered_query: Query = session.query(UpsertionRecord).filter(
                and_(
                    UpsertionRecord.key.in_(keys),
                    UpsertionRecord.namespace == self.namespace,
                )
            )

            filtered_query.delete()
            session.commit()

    async def adelete_keys(self, keys: Sequence[str]) -> None:
        """Delete records from the SQLite database."""
        async with self._amake_session() as session:
            await session.execute(
                delete(UpsertionRecord).where(
                    and_(
                        UpsertionRecord.key.in_(keys),
                        UpsertionRecord.namespace == self.namespace,
                    )
                )
            )

            await session.commit()

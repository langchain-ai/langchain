"""Implementation of a record management layer.

Currently, includes an implementation that uses SQLAlchemy which should
allow it to work with a variety of SQL as a backend.

* Each key is associated with an updated_at field.
* This filed is updated whenever the key is updated.
* Keys can be listed based on the updated at field.
* Keys can be deleted.
"""
import contextlib
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Sequence, Dict, Any, Generator

from sqlalchemy import (
    and_,
    create_engine,
    select,
    Column,
    String,
    DateTime,
    func,
    Engine,
    UniqueConstraint,
    Index,
)
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

Base = declarative_base()


class UpsertionRecord(Base):  # type: ignore[valid-type,misc]
    """Table used to keep track of when a key was last updated."""

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

    # Created at and updated at should be using the server time to make sure
    # that time is incremented monotonically.
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(
        DateTime, server_default=func.now(), server_onupdate=func.now(), index=True
    )

    __table_args__ = (
        UniqueConstraint("key", "namespace", name="uix_key_namespace"),
        Index("ix_key_namespace", "key", "namespace"),
    )


class AbstractRecordManager(ABC):
    """An abstract base class representing the interface for a record manager."""

    def __init__(
        self,
        namespace: str,
    ) -> None:
        """
        Initialize the record manager.

        Args:
            namespace (str): The namespace for the record manager.
        """
        self.namespace = namespace

    @abstractmethod
    def create_schema(self) -> None:
        """Create the database schema for the record manager."""
        pass

    @abstractmethod
    def get_time(self) -> datetime:
        """Get the current server time.

        It's important to get this from the server to ensure a monotonic clock,
        otherwise there may be data loss when cleaning up old documents!

        Returns:
            The current server time.
        """

    @abstractmethod
    def update(
        self,
        keys: Sequence[str],
        *,
        group_ids: Optional[Sequence[Optional[str]]] = None,
        time_at_least: Optional[datetime] = None,
    ) -> None:
        """Upsert records into the database.

        Args:
            keys: A list of record keys to upsert.
            group_ids: A list of group IDs corresponding to the keys.
            time_at_least: if provided, updates should only happen if the
              updated_at field is at least this time.

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
    def list_keys(
        self,
        *,
        before: Optional[datetime] = None,
        after: Optional[datetime] = None,
        group_ids: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """List records in the database based on the provided filters.

        Args:
            before: Filter to list records updated before this time.
            after: Filter to list records updated after this time.
            group_ids: Filter to list records with specific group IDs.

        Returns:
            A list of keys for the matching records.
        """

    @abstractmethod
    def delete_keys(self, keys: Sequence[str]) -> None:
        """Delete specified records from the database.

        Args:
            keys: A list of keys to delete.
        """


class SQLRecordManager(AbstractRecordManager):
    """A manager persistence layer used to keep track of upserted records."""

    def __init__(
        self,
        namespace: str,
        *,
        engine: Optional[Engine] = None,
        db_url: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the record store."""
        super().__init__(namespace=namespace)
        if db_url is None and engine is None:
            raise ValueError("Must specify either db_url or engine")
        if db_url is not None and engine is not None:
            raise ValueError("Must specify either db_url or engine, not both")

        if db_url:
            _kwargs = engine_kwargs or {}
            _engine = create_engine(db_url, **_kwargs)
        elif engine:
            _engine = engine
        else:
            raise AssertionError("Something went wrong with configuration of engine.")

        self.engine = _engine
        self.session_factory = sessionmaker(bind=self.engine)

    def create_schema(self) -> None:
        """Create the database schema."""
        Base.metadata.create_all(self.engine)

    @contextlib.contextmanager
    def _make_session(self) -> Generator[Session, None, None]:
        """Create a session and close it after use."""
        session = self.session_factory()
        try:
            yield session
        finally:
            session.close()

    def get_time(self) -> datetime:
        """Get the current server time.

        Please note it's critical that time is obtained from the server since
        we want a monotonic clock.
        """
        with self._make_session() as session:
            return session.execute(select(func.now())).scalar()

    def update(
        self,
        keys: Sequence[str],
        *,
        group_ids: Optional[Sequence[Optional[str]]] = None,
        time_at_least: Optional[datetime] = None,
    ) -> None:
        """Upsert records into the SQLite database."""
        if group_ids is None:
            group_ids = [None] * len(keys)

        if len(keys) != len(group_ids):
            raise ValueError(
                f"Number of keys ({len(keys)}) does not match number of "
                f"group_ids ({len(group_ids)})"
            )

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
            insert_stmt = insert(UpsertionRecord).values(records_to_upsert)
            stmt = insert_stmt.on_conflict_do_update(
                [UpsertionRecord.key, UpsertionRecord.namespace],
                set_=dict(
                    updated_at=insert_stmt.excluded.updated_at,
                ),
            )
            session.execute(stmt)
            session.commit()

    def exists(self, keys: Sequence[str]) -> List[bool]:
        """Check if the given keys exist in the SQLite database."""
        with self._make_session() as session:
            records = (
                session.query(UpsertionRecord.key)
                .filter(
                    and_(
                        UpsertionRecord.key.in_(keys),
                        UpsertionRecord.namespace == self.namespace,
                    )
                )
                .all()
            )
        found_keys = set(r.key for r in records)
        return [k in found_keys for k in keys]

    def list_keys(
        self,
        *,
        before: Optional[datetime] = None,
        after: Optional[datetime] = None,
        group_ids: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """List records in the SQLite database based on the provided date range."""
        with self._make_session() as session:
            query = session.query(UpsertionRecord).filter(
                UpsertionRecord.namespace == self.namespace
            )

            if after:
                query = query.filter(UpsertionRecord.updated_at > after)
            if before:
                query = query.filter(UpsertionRecord.updated_at < before)
            if group_ids:
                query = query.filter(UpsertionRecord.group_id.in_(group_ids))
            records = query.all()
        return [r.key for r in records]

    def delete_keys(self, keys: Sequence[str]) -> None:
        """Delete records from the SQLite database."""
        with self._make_session() as session:
            session.query(UpsertionRecord).filter(
                and_(
                    UpsertionRecord.key.in_(keys),
                    UpsertionRecord.namespace == self.namespace,
                )
            ).delete()
            session.commit()

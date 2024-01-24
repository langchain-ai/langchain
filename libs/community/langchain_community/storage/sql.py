"""SQL storage that persists data in a SQL database
and supports data isolation using collections."""
from __future__ import annotations

import uuid
from typing import Any, Generic, Iterator, List, Optional, Sequence, Tuple, TypeVar

import sqlalchemy
from sqlalchemy import JSON, UUID
from sqlalchemy.orm import Session, relationship

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

from langchain_core.documents import Document
from langchain_core.load import Serializable, dumps, loads
from langchain_core.stores import BaseStore

V = TypeVar("V")

ITERATOR_WINDOW_SIZE = 1000

Base = declarative_base()  # type: Any


_LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"


class BaseModel(Base):
    """Base model for the SQL stores."""

    __abstract__ = True
    uuid = sqlalchemy.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


_classes: Any = None


def _get_storage_stores() -> Any:
    global _classes
    if _classes is not None:
        return _classes

    class CollectionStore(BaseModel):
        """Collection store."""

        __tablename__ = "langchain_storage_collection"

        name = sqlalchemy.Column(sqlalchemy.String)
        cmetadata = sqlalchemy.Column(JSON)

        items = relationship(
            "ItemStore",
            back_populates="collection",
            passive_deletes=True,
        )

        @classmethod
        def get_by_name(
            cls, session: Session, name: str
        ) -> Optional["CollectionStore"]:
            # type: ignore
            return session.query(cls).filter(cls.name == name).first()

        @classmethod
        def get_or_create(
            cls,
            session: Session,
            name: str,
            cmetadata: Optional[dict] = None,
        ) -> Tuple["CollectionStore", bool]:
            """
            Get or create a collection.
            Returns [Collection, bool] where the bool is True if the collection was created.
            """  # noqa: E501
            created = False
            collection = cls.get_by_name(session, name)
            if collection:
                return collection, created

            collection = cls(name=name, cmetadata=cmetadata)
            session.add(collection)
            session.commit()
            created = True
            return collection, created

    class ItemStore(BaseModel):
        """Item store."""

        __tablename__ = "langchain_storage_items"

        collection_id = sqlalchemy.Column(
            UUID(as_uuid=True),
            sqlalchemy.ForeignKey(
                f"{CollectionStore.__tablename__}.uuid",
                ondelete="CASCADE",
            ),
        )
        collection = relationship(CollectionStore, back_populates="items")

        content = sqlalchemy.Column(sqlalchemy.String, nullable=True)

        # custom_id : any user defined id
        custom_id = sqlalchemy.Column(sqlalchemy.String, nullable=True)

    _classes = (ItemStore, CollectionStore)

    return _classes


class SQLBaseStore(BaseStore[str, V], Generic[V]):
    """SQL storage

    Args:
        connection_string: SQL connection string that will be passed to SQLAlchemy.
        collection_name: The name of the collection to use. (default: langchain)
            NOTE: Collections are useful to isolate your data in a given a database.
            This is not the name of the table, but the name of the collection.
            The tables will be created when initializing the store (if not exists)
            So, make sure the user has the right permissions to create tables.
        pre_delete_collection: If True, will delete the collection if it exists.
            (default: False). Useful for testing.
        engine_args: SQLAlchemy's create engine arguments.

    Example:
        .. code-block:: python

            from langchain_community.storage import SQLDocStore
            from langchain_community.embeddings.openai import OpenAIEmbeddings

            # example using an SQLDocStore to store Document objects for
            # a ParentDocumentRetriever
            CONNECTION_STRING = "postgresql+psycopg2://user:pass@localhost:5432/db"
            COLLECTION_NAME = "state_of_the_union_test"
            docstore = SQLDocStore(
                collection_name=COLLECTION_NAME,
                connection_string=CONNECTION_STRING,
            )
            child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
            vectorstore = ...

            retriever = ParentDocumentRetriever(
                vectorstore=vectorstore,
                docstore=docstore,
                child_splitter=child_splitter,
            )

            # example using an SQLStrStore to store strings
            # same example as in "InMemoryStore" but using SQL persistence
            store = SQLDocStore(
                collection_name=COLLECTION_NAME,
                connection_string=CONNECTION_STRING,
            )
            store.mset([('key1', 'value1'), ('key2', 'value2')])
            store.mget(['key1', 'key2'])
            # ['value1', 'value2']
            store.mdelete(['key1'])
            list(store.yield_keys())
            # ['key2']
            list(store.yield_keys(prefix='k'))
            # ['key2']

            # delete the COLLECTION_NAME collection
            docstore.delete_collection()
    """

    def __init__(
        self,
        connection_string: str,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        collection_metadata: Optional[dict] = None,
        pre_delete_collection: bool = False,
        connection: Optional[sqlalchemy.engine.Connection] = None,
        engine_args: Optional[dict[str, Any]] = None,
    ) -> None:
        self.connection_string = connection_string
        self.collection_name = collection_name
        self.collection_metadata = collection_metadata
        self.pre_delete_collection = pre_delete_collection
        self.engine_args = engine_args or {}
        # Create a connection if not provided, otherwise use the provided connection
        self._conn = connection if connection else self.__connect()
        self.__post_init__()

    def __post_init__(
        self,
    ) -> None:
        """Initialize the store."""
        ItemStore, CollectionStore = _get_storage_stores()
        self.CollectionStore = CollectionStore
        self.ItemStore = ItemStore
        self.__create_tables_if_not_exists()
        self.__create_collection()

    def __connect(self) -> sqlalchemy.engine.Connection:
        engine = sqlalchemy.create_engine(self.connection_string, **self.engine_args)
        conn = engine.connect()
        return conn

    def __create_tables_if_not_exists(self) -> None:
        with self._conn.begin():
            Base.metadata.create_all(self._conn)

    def __create_collection(self) -> None:
        if self.pre_delete_collection:
            self.delete_collection()
        with Session(self._conn) as session:
            self.CollectionStore.get_or_create(
                session, self.collection_name, cmetadata=self.collection_metadata
            )

    def delete_collection(self) -> None:
        with Session(self._conn) as session:
            collection = self.__get_collection(session)
            if not collection:
                return
            session.delete(collection)
            session.commit()

    def __get_collection(self, session: Session) -> Any:
        return self.CollectionStore.get_by_name(session, self.collection_name)

    def __del__(self) -> None:
        if self._conn:
            self._conn.close()

    def __serialize_value(self, obj: V) -> str:
        if isinstance(obj, Serializable):
            return dumps(obj)
        return obj

    def __deserialize_value(self, obj: V) -> str:
        try:
            return loads(obj)
        except Exception:
            return obj

    def mget(self, keys: Sequence[str]) -> List[Optional[V]]:
        """Get the values associated with the given keys.

        Args:
            keys (Sequence[str]): A sequence of keys.

        Returns:
            A sequence of optional values associated with the keys.
            If a key is not found, the corresponding value will be None.
        """
        with Session(self._conn) as session:
            collection = self.__get_collection(session)

            items = (
                session.query(self.ItemStore.content, self.ItemStore.custom_id)
                .where(
                    sqlalchemy.and_(
                        self.ItemStore.custom_id.in_(keys),
                        self.ItemStore.collection_id == (collection.uuid),
                    )
                )
                .all()
            )

        ordered_values = {key: None for key in keys}
        for item in items:
            v = item[0]
            val = self.__deserialize_value(v) if v is not None else v
            k = item[1]
            ordered_values[k] = val

        return [ordered_values[key] for key in keys]

    def mset(self, key_value_pairs: Sequence[Tuple[str, V]]) -> None:
        """Set the values for the given keys.

        Args:
            key_value_pairs (Sequence[Tuple[str, V]]): A sequence of key-value pairs.

        Returns:
            None
        """
        with Session(self._conn) as session:
            collection = self.__get_collection(session)
            if not collection:
                raise ValueError("Collection not found")
            for id, item in key_value_pairs:
                content = self.__serialize_value(item)
                item_store = self.ItemStore(
                    content=content,
                    custom_id=id,
                    collection_id=collection.uuid,
                )
                session.add(item_store)
            session.commit()

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete the given keys and their associated values.

        Args:
            keys (Sequence[str]): A sequence of keys to delete.
        """
        with Session(self._conn) as session:
            collection = self.__get_collection(session)
            if not collection:
                raise ValueError("Collection not found")
            if keys is not None:
                stmt = sqlalchemy.delete(self.ItemStore).where(
                    sqlalchemy.and_(
                        self.ItemStore.custom_id.in_(keys),
                        self.ItemStore.collection_id == (collection.uuid),
                    )
                )
                session.execute(stmt)
            session.commit()

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        """Get an iterator over keys that match the given prefix.

        Args:
            prefix (str, optional): The prefix to match. Defaults to None.

        Returns:
            Iterator[str]: An iterator over keys that match the given prefix.
        """
        with Session(self._conn) as session:
            collection = self.__get_collection(session)
            start = 0
            while True:
                stop = start + ITERATOR_WINDOW_SIZE
                query = session.query(self.ItemStore.custom_id).where(
                    self.ItemStore.collection_id == (collection.uuid)
                )
                if prefix is not None:
                    query = query.filter(self.ItemStore.custom_id.startswith(prefix))
                items = query.slice(start, stop).all()

                if len(items) == 0:
                    break
                for item in items:
                    yield item[0]
                start += ITERATOR_WINDOW_SIZE


SQLDocStore = SQLBaseStore[Document]
SQLStrStore = SQLBaseStore[str]

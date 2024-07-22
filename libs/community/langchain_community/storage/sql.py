import contextlib
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

from langchain_core.stores import BaseStore
from sqlalchemy import (
    LargeBinary,
    Text,
    and_,
    create_engine,
    delete,
    select,
)
from sqlalchemy.engine.base import Engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    create_async_engine,
)
from sqlalchemy.orm import (
    Mapped,
    Session,
    declarative_base,
    sessionmaker,
)

try:
    from sqlalchemy.ext.asyncio import async_sessionmaker
except ImportError:
    # dummy for sqlalchemy < 2
    async_sessionmaker = type("async_sessionmaker", (type,), {})  # type: ignore

Base = declarative_base()

try:
    from sqlalchemy.orm import mapped_column

    class LangchainKeyValueStores(Base):  # type: ignore[valid-type,misc]
        """Table used to save values."""

        # ATTENTION:
        # Prior to modifying this table, please determine whether
        # we should create migrations for this table to make sure
        # users do not experience data loss.
        __tablename__ = "langchain_key_value_stores"

        namespace: Mapped[str] = mapped_column(
            primary_key=True, index=True, nullable=False
        )
        key: Mapped[str] = mapped_column(primary_key=True, index=True, nullable=False)
        value = mapped_column(LargeBinary, index=False, nullable=False)

except ImportError:
    # dummy for sqlalchemy < 2
    from sqlalchemy import Column

    class LangchainKeyValueStores(Base):  # type: ignore[valid-type,misc,no-redef]
        """Table used to save values."""

        # ATTENTION:
        # Prior to modifying this table, please determine whether
        # we should create migrations for this table to make sure
        # users do not experience data loss.
        __tablename__ = "langchain_key_value_stores"

        namespace = Column(Text(), primary_key=True, index=True, nullable=False)
        key = Column(Text(), primary_key=True, index=True, nullable=False)
        value = Column(LargeBinary, index=False, nullable=False)


def items_equal(x: Any, y: Any) -> bool:
    return x == y


# This is a fix of original SQLStore.
# This can will be removed when a PR will be merged.
class SQLStore(BaseStore[str, bytes]):
    """BaseStore interface that works on an SQL database.

    Examples:
        Create a SQLStore instance and perform operations on it:

        .. code-block:: python

            from langchain_rag.storage import SQLStore

            # Instantiate the SQLStore with the root path
            sql_store = SQLStore(namespace="test", db_url="sqlite://:memory:")

            # Set values for keys
            sql_store.mset([("key1", b"value1"), ("key2", b"value2")])

            # Get values for keys
            values = sql_store.mget(["key1", "key2"])  # Returns [b"value1", b"value2"]

            # Delete keys
            sql_store.mdelete(["key1"])

            # Iterate over keys
            for key in sql_store.yield_keys():
                print(key)

    """

    def __init__(
        self,
        *,
        namespace: str,
        db_url: Optional[Union[str, Path]] = None,
        engine: Optional[Union[Engine, AsyncEngine]] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        async_mode: Optional[bool] = None,
    ):
        if db_url is None and engine is None:
            raise ValueError("Must specify either db_url or engine")

        if db_url is not None and engine is not None:
            raise ValueError("Must specify either db_url or engine, not both")

        _engine: Union[Engine, AsyncEngine]
        if db_url:
            if async_mode is None:
                async_mode = False
            if async_mode:
                _engine = create_async_engine(
                    url=str(db_url),
                    **(engine_kwargs or {}),
                )
            else:
                _engine = create_engine(url=str(db_url), **(engine_kwargs or {}))
        elif engine:
            _engine = engine

        else:
            raise AssertionError("Something went wrong with configuration of engine.")

        _session_maker: Union[sessionmaker[Session], async_sessionmaker[AsyncSession]]
        if isinstance(_engine, AsyncEngine):
            self.async_mode = True
            _session_maker = async_sessionmaker(bind=_engine)
        else:
            self.async_mode = False
            _session_maker = sessionmaker(bind=_engine)

        self.engine = _engine
        self.dialect = _engine.dialect.name
        self.session_maker = _session_maker
        self.namespace = namespace

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)  # problem in sqlalchemy v1
        # sqlalchemy.exc.CompileError: (in table 'langchain_key_value_stores',
        # column 'namespace'): Can't generate DDL for NullType(); did you forget
        # to specify a type on this Column?

    async def acreate_schema(self) -> None:
        assert isinstance(self.engine, AsyncEngine)
        async with self.engine.begin() as session:
            await session.run_sync(Base.metadata.create_all)

    def drop(self) -> None:
        Base.metadata.drop_all(bind=self.engine.connect())

    async def amget(self, keys: Sequence[str]) -> List[Optional[bytes]]:
        assert isinstance(self.engine, AsyncEngine)
        result: Dict[str, bytes] = {}
        async with self._make_async_session() as session:
            stmt = select(LangchainKeyValueStores).filter(
                and_(
                    LangchainKeyValueStores.key.in_(keys),
                    LangchainKeyValueStores.namespace == self.namespace,
                )
            )
            for v in await session.scalars(stmt):
                result[v.key] = v.value
        return [result.get(key) for key in keys]

    def mget(self, keys: Sequence[str]) -> List[Optional[bytes]]:
        result = {}

        with self._make_sync_session() as session:
            stmt = select(LangchainKeyValueStores).filter(
                and_(
                    LangchainKeyValueStores.key.in_(keys),
                    LangchainKeyValueStores.namespace == self.namespace,
                )
            )
            for v in session.scalars(stmt):
                result[v.key] = v.value
        return [result.get(key) for key in keys]

    async def amset(self, key_value_pairs: Sequence[Tuple[str, bytes]]) -> None:
        async with self._make_async_session() as session:
            await self._amdelete([key for key, _ in key_value_pairs], session)
            session.add_all(
                [
                    LangchainKeyValueStores(namespace=self.namespace, key=k, value=v)
                    for k, v in key_value_pairs
                ]
            )
            await session.commit()

    def mset(self, key_value_pairs: Sequence[Tuple[str, bytes]]) -> None:
        values: Dict[str, bytes] = dict(key_value_pairs)
        with self._make_sync_session() as session:
            self._mdelete(list(values.keys()), session)
            session.add_all(
                [
                    LangchainKeyValueStores(namespace=self.namespace, key=k, value=v)
                    for k, v in values.items()
                ]
            )
            session.commit()

    def _mdelete(self, keys: Sequence[str], session: Session) -> None:
        stmt = delete(LangchainKeyValueStores).filter(
            and_(
                LangchainKeyValueStores.key.in_(keys),
                LangchainKeyValueStores.namespace == self.namespace,
            )
        )
        session.execute(stmt)

    async def _amdelete(self, keys: Sequence[str], session: AsyncSession) -> None:
        stmt = delete(LangchainKeyValueStores).filter(
            and_(
                LangchainKeyValueStores.key.in_(keys),
                LangchainKeyValueStores.namespace == self.namespace,
            )
        )
        await session.execute(stmt)

    def mdelete(self, keys: Sequence[str]) -> None:
        with self._make_sync_session() as session:
            self._mdelete(keys, session)
            session.commit()

    async def amdelete(self, keys: Sequence[str]) -> None:
        async with self._make_async_session() as session:
            await self._amdelete(keys, session)
            await session.commit()

    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        with self._make_sync_session() as session:
            for v in session.query(LangchainKeyValueStores).filter(  # type: ignore
                LangchainKeyValueStores.namespace == self.namespace
            ):
                if str(v.key).startswith(prefix or ""):
                    yield str(v.key)
            session.close()

    async def ayield_keys(self, *, prefix: Optional[str] = None) -> AsyncIterator[str]:
        async with self._make_async_session() as session:
            stmt = select(LangchainKeyValueStores).filter(
                LangchainKeyValueStores.namespace == self.namespace
            )
            for v in await session.scalars(stmt):
                if str(v.key).startswith(prefix or ""):
                    yield str(v.key)
            await session.close()

    @contextlib.contextmanager
    def _make_sync_session(self) -> Generator[Session, None, None]:
        """Make an async session."""
        if self.async_mode:
            raise ValueError(
                "Attempting to use a sync method in when async mode is turned on. "
                "Please use the corresponding async method instead."
            )
        with cast(Session, self.session_maker()) as session:
            yield cast(Session, session)

    @contextlib.asynccontextmanager
    async def _make_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Make an async session."""
        if not self.async_mode:
            raise ValueError(
                "Attempting to use an async method in when sync mode is turned on. "
                "Please use the corresponding async method instead."
            )
        async with cast(AsyncSession, self.session_maker()) as session:
            yield cast(AsyncSession, session)

"""Implement integration tests for Redis storage."""

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from langchain_community.storage import SQLStore

pytest.importorskip("sqlalchemy")


@pytest.fixture
def sql_engine() -> Engine:
    """Yield redis client."""
    return create_engine(url="sqlite://", echo=True)


@pytest.fixture
def sql_aengine() -> AsyncEngine:
    """Yield redis client."""
    return create_async_engine(url="sqlite+aiosqlite:///:memory:", echo=True)


def test_mget(sql_engine: Engine) -> None:
    """Test mget method."""
    store = SQLStore(engine=sql_engine, namespace="test")
    store.create_schema()
    keys = ["key1", "key2"]
    with sql_engine.connect() as session:
        session.execute(
            text(
                "insert into langchain_key_value_stores ('namespace', 'key', 'value') "
                "values('test','key1',:value)"
            ).bindparams(value=b"value1"),
        )
        session.execute(
            text(
                "insert into langchain_key_value_stores ('namespace', 'key', 'value') "
                "values('test','key2',:value)"
            ).bindparams(value=b"value2"),
        )
        session.commit()

    result = store.mget(keys)
    assert result == [b"value1", b"value2"]


@pytest.mark.asyncio
async def test_amget(sql_aengine: AsyncEngine) -> None:
    """Test mget method."""
    store = SQLStore(engine=sql_aengine, namespace="test")
    await store.acreate_schema()
    keys = ["key1", "key2"]
    async with sql_aengine.connect() as session:
        await session.execute(
            text(
                "insert into langchain_key_value_stores ('namespace', 'key', 'value') "
                "values('test','key1',:value)"
            ).bindparams(value=b"value1"),
        )
        await session.execute(
            text(
                "insert into langchain_key_value_stores ('namespace', 'key', 'value') "
                "values('test','key2',:value)"
            ).bindparams(value=b"value2"),
        )
        await session.commit()

    result = await store.amget(keys)
    assert result == [b"value1", b"value2"]


def test_mset(sql_engine: Engine) -> None:
    """Test that multiple keys can be set."""
    store = SQLStore(engine=sql_engine, namespace="test")
    store.create_schema()
    key_value_pairs = [("key1", b"value1"), ("key2", b"value2")]
    store.mset(key_value_pairs)

    with sql_engine.connect() as session:
        result = session.exec_driver_sql("select * from langchain_key_value_stores")
        assert result.keys() == ["namespace", "key", "value"]
        data = [(row[0], row[1]) for row in result]
        assert data == [("test", "key1"), ("test", "key2")]
        session.commit()


@pytest.mark.asyncio
async def test_amset(sql_aengine: AsyncEngine) -> None:
    """Test that multiple keys can be set."""
    store = SQLStore(engine=sql_aengine, namespace="test")
    await store.acreate_schema()
    key_value_pairs = [("key1", b"value1"), ("key2", b"value2")]
    await store.amset(key_value_pairs)

    async with sql_aengine.connect() as session:
        result = await session.exec_driver_sql(
            "select * from langchain_key_value_stores"
        )
        assert result.keys() == ["namespace", "key", "value"]
        data = [(row[0], row[1]) for row in result]
        assert data == [("test", "key1"), ("test", "key2")]
        await session.commit()


def test_mdelete(sql_engine: Engine) -> None:
    """Test that deletion works as expected."""
    store = SQLStore(engine=sql_engine, namespace="test")
    store.create_schema()
    keys = ["key1", "key2"]
    with sql_engine.connect() as session:
        session.execute(
            text(
                "insert into langchain_key_value_stores ('namespace', 'key', 'value') "
                "values('test','key1',:value)"
            ).bindparams(value=b"value1"),
        )
        session.execute(
            text(
                "insert into langchain_key_value_stores ('namespace', 'key', 'value') "
                "values('test','key2',:value)"
            ).bindparams(value=b"value2"),
        )
        session.commit()
    store.mdelete(keys)
    with sql_engine.connect() as session:
        result = session.exec_driver_sql("select * from langchain_key_value_stores")
        assert result.keys() == ["namespace", "key", "value"]
        data = [row for row in result]
        assert data == []
        session.commit()


@pytest.mark.asyncio
async def test_amdelete(sql_aengine: AsyncEngine) -> None:
    """Test that deletion works as expected."""
    store = SQLStore(engine=sql_aengine, namespace="test")
    await store.acreate_schema()
    keys = ["key1", "key2"]
    async with sql_aengine.connect() as session:
        await session.execute(
            text(
                "insert into langchain_key_value_stores ('namespace', 'key', 'value') "
                "values('test','key1',:value)"
            ).bindparams(value=b"value1"),
        )
        await session.execute(
            text(
                "insert into langchain_key_value_stores ('namespace', 'key', 'value') "
                "values('test','key2',:value)"
            ).bindparams(value=b"value2"),
        )
        await session.commit()
    await store.amdelete(keys)
    async with sql_aengine.connect() as session:
        result = await session.exec_driver_sql(
            "select * from langchain_key_value_stores"
        )
        assert result.keys() == ["namespace", "key", "value"]
        data = [row for row in result]
        assert data == []
        await session.commit()


def test_yield_keys(sql_engine: Engine) -> None:
    store = SQLStore(engine=sql_engine, namespace="test")
    store.create_schema()
    key_value_pairs = [("key1", b"value1"), ("key2", b"value2")]
    store.mset(key_value_pairs)
    assert sorted(store.yield_keys()) == ["key1", "key2"]
    assert sorted(store.yield_keys(prefix="key")) == ["key1", "key2"]
    assert sorted(store.yield_keys(prefix="lang")) == []


@pytest.mark.asyncio
async def test_ayield_keys(sql_aengine: AsyncEngine) -> None:
    store = SQLStore(engine=sql_aengine, namespace="test")
    await store.acreate_schema()
    key_value_pairs = [("key1", b"value1"), ("key2", b"value2")]
    await store.amset(key_value_pairs)
    assert sorted([k async for k in store.ayield_keys()]) == ["key1", "key2"]
    assert sorted([k async for k in store.ayield_keys(prefix="key")]) == [
        "key1",
        "key2",
    ]
    assert sorted([k async for k in store.ayield_keys(prefix="lang")]) == []

from typing import AsyncGenerator, Generator, cast

import pytest
import sqlalchemy as sa
from langchain.storage._lc_store import create_kv_docstore, create_lc_store
from langchain_core.documents import Document
from langchain_core.stores import BaseStore
from packaging import version

from langchain_community.storage.sql import SQLStore

is_sqlalchemy_v1 = version.parse(sa.__version__).major == 1


@pytest.fixture
def sql_store() -> Generator[SQLStore, None, None]:
    store = SQLStore(namespace="test", db_url="sqlite://")
    store.create_schema()
    yield store


@pytest.fixture
async def async_sql_store() -> AsyncGenerator[SQLStore, None]:
    store = SQLStore(namespace="test", db_url="sqlite+aiosqlite://", async_mode=True)
    await store.acreate_schema()
    yield store


@pytest.mark.xfail(is_sqlalchemy_v1, reason="SQLAlchemy 1.x issues")
def test_create_lc_store(sql_store: SQLStore) -> None:
    """Test that a docstore is created from a base store."""
    docstore: BaseStore[str, Document] = cast(
        BaseStore[str, Document], create_lc_store(sql_store)
    )
    docstore.mset([("key1", Document(page_content="hello", metadata={"key": "value"}))])
    fetched_doc = docstore.mget(["key1"])[0]
    assert fetched_doc is not None
    assert fetched_doc.page_content == "hello"
    assert fetched_doc.metadata == {"key": "value"}


@pytest.mark.xfail(is_sqlalchemy_v1, reason="SQLAlchemy 1.x issues")
def test_create_kv_store(sql_store: SQLStore) -> None:
    """Test that a docstore is created from a base store."""
    docstore = create_kv_docstore(sql_store)
    docstore.mset([("key1", Document(page_content="hello", metadata={"key": "value"}))])
    fetched_doc = docstore.mget(["key1"])[0]
    assert isinstance(fetched_doc, Document)
    assert fetched_doc.page_content == "hello"
    assert fetched_doc.metadata == {"key": "value"}


@pytest.mark.requires("aiosqlite")
async def test_async_create_kv_store(async_sql_store: SQLStore) -> None:
    """Test that a docstore is created from a base store."""
    docstore = create_kv_docstore(async_sql_store)
    await docstore.amset(
        [("key1", Document(page_content="hello", metadata={"key": "value"}))]
    )
    fetched_doc = (await docstore.amget(["key1"]))[0]
    assert isinstance(fetched_doc, Document)
    assert fetched_doc.page_content == "hello"
    assert fetched_doc.metadata == {"key": "value"}


@pytest.mark.xfail(is_sqlalchemy_v1, reason="SQLAlchemy 1.x issues")
def test_sample_sql_docstore(sql_store: SQLStore) -> None:
    # Set values for keys
    sql_store.mset([("key1", b"value1"), ("key2", b"value2")])

    # Get values for keys
    values = sql_store.mget(["key1", "key2"])  # Returns [b"value1", b"value2"]
    assert values == [b"value1", b"value2"]
    # Delete keys
    sql_store.mdelete(["key1"])

    # Iterate over keys
    assert [key for key in sql_store.yield_keys()] == ["key2"]


@pytest.mark.requires("aiosqlite")
async def test_async_sample_sql_docstore(async_sql_store: SQLStore) -> None:
    # Set values for keys
    await async_sql_store.amset([("key1", b"value1"), ("key2", b"value2")])
    # sql_store.mset([("key1", "value1"), ("key2", "value2")])

    # Get values for keys
    values = await async_sql_store.amget(
        ["key1", "key2"]
    )  # Returns [b"value1", b"value2"]
    assert values == [b"value1", b"value2"]
    # Delete keys
    await async_sql_store.amdelete(["key1"])

    # Iterate over keys
    assert [key async for key in async_sql_store.ayield_keys()] == ["key2"]

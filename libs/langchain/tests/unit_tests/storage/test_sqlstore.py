from typing import Generator, cast

import pytest

from langchain.schema import Document
from langchain.storage._lc_store import create_kv_docstore, create_lc_store
from langchain.storage.sql_docstore import SQLStore


@pytest.fixture
def sql_store() -> Generator[SQLStore, None, None]:
    store = SQLStore(namespace="test", db_url="sqlite://")
    store.create_schema()
    yield store


def test_create_lc_store(sql_store: SQLStore) -> None:
    """Test that a docstore is created from a base store."""
    docstore = create_lc_store(sql_store)
    docstore.mset([("key1", Document(page_content="hello", metadata={"key": "value"}))])
    fetched_doc = cast(Document, docstore.mget(["key1"])[0])
    assert fetched_doc.page_content == "hello"
    assert fetched_doc.metadata == {"key": "value"}


def test_create_kv_store(sql_store: SQLStore) -> None:
    """Test that a docstore is created from a base store."""
    docstore = create_kv_docstore(sql_store)
    docstore.mset([("key1", Document(page_content="hello", metadata={"key": "value"}))])
    fetched_doc = docstore.mget(["key1"])[0]
    assert isinstance(fetched_doc, Document)
    assert fetched_doc.page_content == "hello"
    assert fetched_doc.metadata == {"key": "value"}


def test_sample_sql_docstore() -> None:
    # Instantiate the SQLStore with the root path
    sql_store = SQLStore(namespace="test", db_url="sqlite://")
    # sql_store = SQLStore[str, Any](namespace="test", db_url="sqlite:////tmp/test.db")
    sql_store.create_schema()

    # Set values for keys
    sql_store.mset([("key1", b"value1"), ("key2", b"value2")])
    # sql_store.mset([("key1", "value1"), ("key2", "value2")])

    # Get values for keys
    values = sql_store.mget(["key1", "key2"])  # Returns [b"value1", b"value2"]
    assert values == [b"value1", b"value2"]
    # Delete keys
    sql_store.mdelete(["key1"])

    # Iterate over keys
    assert [key for key in sql_store.yield_keys()] == ["key2"]

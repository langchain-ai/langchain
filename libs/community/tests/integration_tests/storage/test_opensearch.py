"""Implement integration tests for OpenSearch storage."""

import os
from typing import Generator, TYPE_CHECKING
import uuid

import pytest

from langchain_community.storage.opensearch import OpenSearchStore
from langchain_core.documents import Document

if TYPE_CHECKING:
    from opensearchpy import OpenSearch

pytest.importorskip("opensearchpy")


@pytest.fixture
def opensearch_client() -> OpenSearch:
    """Yield OpenSearch client."""
    from opensearchpy import OpenSearch

    host = os.environ.get("OPENSEARCH_HOST", "localhost")
    index_name = os.environ.get("OPENSEARCH_INDEX", f"test_index_{uuid.uuid4().hex}")
    username = os.environ.get("OPENSEARCH_USERNAME")
    password = os.environ.get("OPENSEARCH_PASSWORD")

    client = OpenSearch(
        [host],
        http_auth=(username, password) if username else None,
        use_ssl=False,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
    )

    yield client

    client.indices.delete(index=index_name, ignore=[400, 404])


@pytest.fixture
def store(opensearch_client: OpenSearch) -> Generator[OpenSearchStore, None, None]:
    """Yield an OpenSearchStore instance."""
    index_name = os.environ.get("OPENSEARCH_INDEX", f"test_index_{uuid.uuid4().hex}")
    store = OpenSearchStore(
        host=os.environ.get("OPENSEARCH_HOST", "localhost"),
        index_name=index_name,
        username=os.environ.get("OPENSEARCH_USERNAME"),
        password=os.environ.get("OPENSEARCH_PASSWORD"),
    )
    yield store
    opensearch_client.indices.delete(index=index_name, ignore=[400, 404])


def test_mget(store: OpenSearchStore) -> None:
    """Test OpenSearchStore mget method."""
    store.mset([
        ("key1", Document(page_content="content1", metadata={"author": "Alice"})),
        ("key2", Document(page_content="content2", metadata={"author": "Bob"}))
    ])
    result = store.mget(["key1", "key2"])
    assert result[0].page_content == "content1"
    assert result[1].page_content == "content2"


def test_mset(store: OpenSearchStore) -> None:
    """Test OpenSearchStore mset method."""
    store.mset([
        ("key1", Document(page_content="content1", metadata={"author": "Alice"}))
    ])
    result = store._get_document("key1")
    assert result["page_content"] == "content1"
    assert result["metadata"]["author"] == "Alice"


def test_mdelete(store: OpenSearchStore) -> None:
    """Test OpenSearchStore mdelete method."""
    store.mset([
        ("key1", Document(page_content="content1", metadata={})),
        ("key2", Document(page_content="content2", metadata={}))
    ])
    store.mdelete(["key1", "key2"])
    result = store.mget(["key1", "key2"])
    assert result == [None, None]


def test_yield_keys(store: OpenSearchStore) -> None:
    """Test OpenSearchStore yield_keys method."""
    store.mset([
        ("key1", Document(page_content="content1", metadata={})),
        ("key2", Document(page_content="content2", metadata={}))
    ])
    assert sorted(store.yield_keys()) == ["key1", "key2"]
    assert sorted(store.yield_keys(prefix="key")) == ["key1", "key2"]
    assert sorted(store.yield_keys(prefix="unknown")) == []
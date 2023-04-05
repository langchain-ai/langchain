"""Test Qdrant functionality."""
import pytest

from langchain.docstore.document import Document
from langchain.vectorstores import Qdrant
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


@pytest.mark.parametrize(
    ["content_payload_key", "metadata_payload_key"],
    [
        (Qdrant.CONTENT_KEY, Qdrant.METADATA_KEY),
        ("foo", "bar"),
        (Qdrant.CONTENT_KEY, "bar"),
        ("foo", Qdrant.METADATA_KEY),
    ],
)
def test_qdrant(content_payload_key: str, metadata_payload_key: str) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = Qdrant.from_texts(
        texts,
        FakeEmbeddings(),
        location=":memory:",
        content_payload_key=content_payload_key,
        metadata_payload_key=metadata_payload_key,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


@pytest.mark.parametrize(
    ["content_payload_key", "metadata_payload_key"],
    [
        (Qdrant.CONTENT_KEY, Qdrant.METADATA_KEY),
        ("test_content", "test_payload"),
        (Qdrant.CONTENT_KEY, "payload_test"),
        ("content_test", Qdrant.METADATA_KEY),
    ],
)
def test_qdrant_with_metadatas(
    content_payload_key: str, metadata_payload_key: str
) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = Qdrant.from_texts(
        texts,
        FakeEmbeddings(),
        metadatas=metadatas,
        location=":memory:",
        content_payload_key=content_payload_key,
        metadata_payload_key=metadata_payload_key,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": 0})]


def test_qdrant_similarity_search_filters() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = Qdrant.from_texts(
        texts,
        FakeEmbeddings(),
        metadatas=metadatas,
        location=":memory:",
    )
    output = docsearch.similarity_search("foo", k=1, filter={"page": 1})
    assert output == [Document(page_content="bar", metadata={"page": 1})]


@pytest.mark.parametrize(
    ["content_payload_key", "metadata_payload_key"],
    [
        (Qdrant.CONTENT_KEY, Qdrant.METADATA_KEY),
        ("test_content", "test_payload"),
        (Qdrant.CONTENT_KEY, "payload_test"),
        ("content_test", Qdrant.METADATA_KEY),
    ],
)
def test_qdrant_max_marginal_relevance_search(
    content_payload_key: str, metadata_payload_key: str
) -> None:
    """Test end to end construction and MRR search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = Qdrant.from_texts(
        texts,
        FakeEmbeddings(),
        metadatas=metadatas,
        location=":memory:",
        content_payload_key=content_payload_key,
        metadata_payload_key=metadata_payload_key,
    )
    output = docsearch.max_marginal_relevance_search("foo", k=2, fetch_k=3)
    assert output == [
        Document(page_content="foo", metadata={"page": 0}),
        Document(page_content="bar", metadata={"page": 1}),
    ]

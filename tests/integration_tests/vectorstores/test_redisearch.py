"""Test RediSearch functionality."""

from langchain.docstore.document import Document
from langchain.vectorstores.redisearch import RediSearch
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


def test_redisearch() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = RediSearch.from_texts(
        texts, FakeEmbeddings(), redisearch_url="redis://localhost:6379"
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

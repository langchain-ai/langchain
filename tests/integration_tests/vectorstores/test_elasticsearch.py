"""Test ElasticSearch functionality."""
from typing import List

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch


class FakeEmbeddings(Embeddings):
    """Fake embeddings functionality for testing."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings."""
        return [[1.0] * 9 + [i] for i in range(len(texts))]

    def embed_query(self, text: str) -> List[float]:
        """Return simple embeddings."""
        return [1.0] * 9 + [0.0]


def test_elasticsearch() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = ElasticVectorSearch.from_texts(
        texts, FakeEmbeddings(), elasticsearch_url="http://localhost:9200"
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_elasticsearch_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = ElasticVectorSearch.from_texts(
        texts,
        FakeEmbeddings(),
        metadatas=metadatas,
        elasticsearch_url="http://localhost:9200",
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": 0})]

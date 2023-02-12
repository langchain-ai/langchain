"""Test ElasticSearch functionality."""

from langchain.docstore.document import Document
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


def test_elasticsearch() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = ElasticVectorSearch.from_texts(
        texts, FakeEmbeddings(), elasticsearch_url="http://localhost:9200"
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_elasticsearch_vector_sim() -> None:
    """Test end to end construction and search by vector."""
    texts = ["foo", "bar", "baz"]
    docsearch = ElasticVectorSearch.from_texts(
        texts, FakeEmbeddings(), elasticsearch_url="http://localhost:9200"
    )
    query_vec = FakeEmbeddings.embed_query("foo")
    output = docsearch.similarity_search_by_vector(query_vec, k=1)
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

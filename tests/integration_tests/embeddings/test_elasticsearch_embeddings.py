import pytest
from elasticsearch import Elasticsearch
from elasticsearch.client import MlClient
from elasticsearch_embeddings import ElasticsearchEmbeddings


@pytest.fixture
def es_connection() -> Elasticsearch:
    # Replace with your actual Elasticsearch connection details
    return Elasticsearch("http://localhost:9200")


@pytest.fixture
def model_id() -> str:
    # Replace with your actual model_id
    return "your_model_id"


def test_elasticsearch_embedding_documents(es_connection: Elasticsearch, model_id: str) -> None:
    """Test Elasticsearch embeddings."""
    documents = ["foo bar"]
    embedding = ElasticsearchEmbeddings(es_connection, model_id)
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 768  # Change 768 to the expected embedding size


def test_elasticsearch_embedding_documents_multiple(es_connection: Elasticsearch, model_id: str) -> None:
    """Test Elasticsearch embeddings."""
    documents = ["foo bar", "bar foo", "foo"]
    embedding = ElasticsearchEmbeddings(es_connection, model_id)
    output = embedding.embed_documents(documents)
    assert len(output) == 3
    assert len(output[0]) == 768  # Change 768 to the expected embedding size
    assert len(output[1]) == 768  # Change 768 to the expected embedding size
    assert len(output[2]) == 768  # Change 768 to the expected embedding size


def test_elasticsearch_embedding_query(es_connection: Elasticsearch, model_id: str) -> None:
    """Test Elasticsearch embeddings."""
    document = "foo bar"
    embedding = ElasticsearchEmbeddings(es_connection, model_id)
    output = embedding.embed_query(document)
    assert len(output) == 768  # Change 768 to the expected embedding size

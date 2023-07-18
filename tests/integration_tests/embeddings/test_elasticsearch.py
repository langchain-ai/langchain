"""Test elasticsearch_embeddings embeddings."""

import pytest

from langchain.embeddings.elasticsearch import ElasticsearchEmbeddings


@pytest.fixture
def model_id() -> str:
    # Replace with your actual model_id
    return "your_model_id"


def test_elasticsearch_embedding_documents(model_id: str) -> None:
    """Test Elasticsearch embedding documents."""
    documents = ["foo bar", "bar foo", "foo"]
    embedding = ElasticsearchEmbeddings.from_credentials(model_id)
    output = embedding.embed_documents(documents)
    assert len(output) == 3
    assert len(output[0]) == 768  # Change 768 to the expected embedding size
    assert len(output[1]) == 768  # Change 768 to the expected embedding size
    assert len(output[2]) == 768  # Change 768 to the expected embedding size


def test_elasticsearch_embedding_query(model_id: str) -> None:
    """Test Elasticsearch embedding query."""
    document = "foo bar"
    embedding = ElasticsearchEmbeddings.from_credentials(model_id)
    output = embedding.embed_query(document)
    assert len(output) == 768  # Change 768 to the expected embedding size

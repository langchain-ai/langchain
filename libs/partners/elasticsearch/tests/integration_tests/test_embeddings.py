"""Test elasticsearch_embeddings embeddings."""

import pytest
from langchain_core.utils import get_from_env

from langchain_elasticsearch.embeddings import ElasticsearchEmbeddings

# deployed with
# https://www.elastic.co/guide/en/machine-learning/current/ml-nlp-text-emb-vector-search-example.html
DEFAULT_MODEL = "sentence-transformers__msmarco-minilm-l-12-v3"
DEFAULT_NUM_DIMENSIONS = "384"


@pytest.fixture
def model_id() -> str:
    return get_from_env("model_id", "MODEL_ID", DEFAULT_MODEL)


@pytest.fixture
def expected_num_dimensions() -> int:
    return int(
        get_from_env(
            "expected_num_dimensions", "EXPECTED_NUM_DIMENSIONS", DEFAULT_NUM_DIMENSIONS
        )
    )


def test_elasticsearch_embedding_documents(
    model_id: str, expected_num_dimensions: int
) -> None:
    """Test Elasticsearch embedding documents."""
    documents = ["foo bar", "bar foo", "foo"]
    embedding = ElasticsearchEmbeddings.from_credentials(model_id)
    output = embedding.embed_documents(documents)
    assert len(output) == 3
    assert len(output[0]) == expected_num_dimensions
    assert len(output[1]) == expected_num_dimensions
    assert len(output[2]) == expected_num_dimensions


def test_elasticsearch_embedding_query(
    model_id: str, expected_num_dimensions: int
) -> None:
    """Test Elasticsearch embedding query."""
    document = "foo bar"
    embedding = ElasticsearchEmbeddings.from_credentials(model_id)
    output = embedding.embed_query(document)
    assert len(output) == expected_num_dimensions

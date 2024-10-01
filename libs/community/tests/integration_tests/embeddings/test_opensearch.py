import pytest
from opensearchpy import OpenSearch
from langchain_community.embeddings.opensearch import OpenSearchEmbedding


@pytest.fixture
def model_id() -> str:
    """Fixture to provide model ID."""
    return "some-model-id"


@pytest.fixture
def opensearch_client() -> OpenSearch:
    """Fixture to provide OpenSearch client connection."""
    return OpenSearch(
        hosts=[{'host': "localhost", 'port': 9200}],  # Remove sensitive info
        http_auth=("username", "password"),  # Remove sensitive info
        use_ssl=True,
        verify_certs=False
    )


@pytest.fixture
def opensearch_embedding(opensearch_client, model_id) -> OpenSearchEmbedding:
    return OpenSearchEmbedding.from_opensearch_connection(opensearch_client, model_id)


def test_opensearch_embedding_documents(opensearch_embedding: OpenSearchEmbedding) -> None:
    """
    Test OpenSearch embedding documents.
    Convert a list of strings, into a list of floats with the shape of its element and its
    embedding vector dimensions.
    """
    documents = ["foo bar", "bar foo", "foo"]
    output = opensearch_embedding.embed_documents(documents)
    assert len(output) == 3
    assert len(output[0]) == 768  # Change 768 to the expected embedding size
    assert len(output[1]) == 768  # Change 768 to the expected embedding size
    assert len(output[2]) == 768  # Change 768 to the expected embedding size


def test_opensearch_embedding_query(opensearch_embedding: OpenSearchEmbedding) -> None:
    """
    Test OpenSearch embedding documents.
    Convert strings, into floats with the shape of its embedding vector dimensions.
    """
    document = "foo bar"
    output = opensearch_embedding.embed_query(document)
    assert len(output) == 768

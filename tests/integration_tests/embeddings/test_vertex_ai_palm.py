"""Test Google PaLM Chat API wrapper.

To use you must have the google-cloud-aiplatform Python package installed and
either:

    1. Have credentials configured for your environment - 
        (gcloud, workload identity, etc...)
    2. Pass your service account key json using the 
        google_application_credentials kwarg to the ChatGoogle constructor.

"""
import pytest

from langchain.embeddings.vertex_ai_palm import GoogleCloudVertexAIPalmEmbeddings

try:
    from vertexai.preview.language_models import TextEmbeddingModel  # noqa: F401

    vertexai_installed = True
except ImportError:
    vertexai_installed = False


@pytest.mark.skipif(
    not vertexai_installed,
    reason="google-cloud-aiplatform>=1.25.0 package not installed",
)
def test_google_palm_embedding_documents() -> None:
    """Test Google PaLM embeddings."""
    documents = ["foo bar"]
    embedding = GoogleCloudVertexAIPalmEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 768


@pytest.mark.skipif(
    not vertexai_installed,
    reason="google-cloud-aiplatform>=1.25.0 package not installed",
)
def test_google_palm_embedding_documents_multiple() -> None:
    """Test Google PaLM embeddings."""
    documents = ["foo bar", "bar foo", "foo"]
    embedding = GoogleCloudVertexAIPalmEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 3
    assert len(output[0]) == 768
    assert len(output[1]) == 768
    assert len(output[2]) == 768


@pytest.mark.skipif(
    not vertexai_installed,
    reason="google-cloud-aiplatform>=1.25.0 package not installed",
)
def test_google_palm_embedding_query() -> None:
    """Test Google PaLM embeddings."""
    document = "foo bar"
    embedding = GoogleCloudVertexAIPalmEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) == 768

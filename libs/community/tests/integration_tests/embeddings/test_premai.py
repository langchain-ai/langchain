"""Test PremAIEmbeddings from PremAI API wrapper.

Note: This test must be run with the PREMAI_API_KEY environment variable set to a valid
API key and a valid project_id. This needs to setup a project in PremAI's platform.
You can check it out here: https://app.premai.io
"""

import pytest

from langchain_community.embeddings.premai import PremAIEmbeddings


@pytest.fixture
def embedder() -> PremAIEmbeddings:
    return PremAIEmbeddings(project_id=8, model="text-embedding-3-small")  # type: ignore[call-arg]


def test_prem_embedding_documents(embedder: PremAIEmbeddings) -> None:
    """Test Prem embeddings."""
    documents = ["foo bar"]
    output = embedder.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 1536


def test_prem_embedding_documents_multiple(embedder: PremAIEmbeddings) -> None:
    """Test prem embeddings for multiple queries or documents."""
    documents = ["foo bar", "bar foo", "foo"]
    output = embedder.embed_documents(documents)
    assert len(output) == 3
    assert len(output[0]) == 1536
    assert len(output[1]) == 1536
    assert len(output[2]) == 1536


def test_prem_embedding_query(embedder: PremAIEmbeddings) -> None:
    """Test Prem embeddings for single query"""
    document = "foo bar"
    output = embedder.embed_query(document)
    assert len(output) == 1536

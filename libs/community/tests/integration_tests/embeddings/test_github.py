"""Tests for the GithubEmbeddings.

Note: This test mmust be run with the GITHUB_TOKEN environment variable set to a
        valid API key.
"""

from langchain_community.embeddings.github import GithubEmbeddings


def test_github_embedding_documents() -> None:
    """Test Github embeddings."""
    documents = ["foo", "bar"]
    embedding = GithubEmbeddings(model="cohere-embed-v3-english")
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) == 1024

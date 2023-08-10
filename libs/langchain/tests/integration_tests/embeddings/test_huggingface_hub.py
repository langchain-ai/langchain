"""Test HuggingFaceHub embeddings."""
import pytest

from langchain.embeddings import HuggingFaceHubEmbeddings


def test_huggingfacehub_embedding_documents() -> None:
    """Test huggingfacehub embeddings."""
    documents = ["foo bar"]
    embedding = HuggingFaceHubEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 768


def test_huggingfacehub_embedding_query() -> None:
    """Test huggingfacehub embeddings."""
    document = "foo bar"
    embedding = HuggingFaceHubEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) == 768


def test_huggingfacehub_embedding_invalid_repo() -> None:
    """Test huggingfacehub embedding repo id validation."""
    # Only sentence-transformers models are currently supported.
    with pytest.raises(ValueError):
        HuggingFaceHubEmbeddings(repo_id="allenai/specter")

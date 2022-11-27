"""Test HuggingFaceHub embeddings."""
import pytest

from langchain.embeddings import HuggingFaceHubEmbeddings


def test_huggingfacehub_embedding_invalid_repo() -> None:
    """Test huggingfacehub embedding repo id validation."""
    # Only sentence-transformers models are currently supported.
    with pytest.raises(ValueError):
        HuggingFaceHubEmbeddings(repo_id="allenai/specter")

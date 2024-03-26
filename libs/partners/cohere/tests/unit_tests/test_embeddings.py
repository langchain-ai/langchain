"""Test embedding model integration."""


from langchain_cohere.embeddings import CohereEmbeddings


def test_initialization() -> None:
    """Test embedding model initialization."""
    CohereEmbeddings(cohere_api_key="test")

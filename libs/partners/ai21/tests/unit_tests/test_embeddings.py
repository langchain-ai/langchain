"""Test embedding model integration."""


from langchain_ai21.embeddings import AI21Embeddings


def test_initialization() -> None:
    """Test embedding model initialization."""
    AI21Embeddings()

"""Test embedding model integration."""


from langchain_together.embeddings import TogetherEmbeddings


def test_initialization() -> None:
    """Test embedding model initialization."""
    TogetherEmbeddings()

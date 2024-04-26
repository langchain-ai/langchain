"""Test embedding model integration."""


from langchain_prompty.embeddings import PromptyEmbeddings


def test_initialization() -> None:
    """Test embedding model initialization."""
    PromptyEmbeddings()

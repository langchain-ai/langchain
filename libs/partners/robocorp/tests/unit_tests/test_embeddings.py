"""Test embedding model integration."""


from langchain_robocorp.embeddings import ActionServerEmbeddings


def test_initialization() -> None:
    """Test embedding model initialization."""
    ActionServerEmbeddings()

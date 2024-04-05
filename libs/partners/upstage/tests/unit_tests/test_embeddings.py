"""Test embedding model integration."""


from langchain_upstage.embeddings import UpstageEmbeddings


def test_initialization() -> None:
    """Test embedding model initialization."""
    UpstageEmbeddings()

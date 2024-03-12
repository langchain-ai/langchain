"""Test embedding model integration."""


from langchain_unify.embeddings import UnifyEmbeddings


def test_initialization() -> None:
    """Test embedding model initialization."""
    UnifyEmbeddings()

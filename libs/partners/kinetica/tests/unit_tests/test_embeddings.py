"""Test embedding model integration."""


from langchain_kinetica.embeddings import KineticaEmbeddings


def test_initialization() -> None:
    """Test embedding model initialization."""
    KineticaEmbeddings()

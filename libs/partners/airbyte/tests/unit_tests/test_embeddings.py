"""Test embedding model integration."""


from langchain_airbyte.embeddings import AirbyteEmbeddings


def test_initialization() -> None:
    """Test embedding model initialization."""
    AirbyteEmbeddings()

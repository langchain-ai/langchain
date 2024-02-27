from langchain_airbyte.vectorstores import AirbyteVectorStore


def test_initialization() -> None:
    """Test integration vectorstore initialization."""
    AirbyteVectorStore()

from langchain_airbyte import AirbyteLoader


def test_initialization() -> None:
    """Test integration vectorstore initialization."""
    AirbyteLoader(source="source-github", stream="pull_requests")

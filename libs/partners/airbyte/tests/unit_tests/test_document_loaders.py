from langchain_airbyte import AirbyteLoader


def test_initialization() -> None:
    """Test integration vectorstore initialization."""
    AirbyteLoader(
        source="source-faker",
        stream="users",
        config={"count": 100},
    )

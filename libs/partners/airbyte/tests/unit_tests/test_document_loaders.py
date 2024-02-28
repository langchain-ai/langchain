from langchain_airbyte import AirbyteLoader


def test_initialization() -> None:
    """Test integration loader initialization."""
    AirbyteLoader(
        source="source-faker",
        stream="users",
        config={"count": 100},
    )

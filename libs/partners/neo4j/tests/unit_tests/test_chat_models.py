"""Test chat model integration."""

from langchain_neo4j.chat_models import ChatNeo4j


def test_initialization() -> None:
    """Test chat model initialization."""
    ChatNeo4j()

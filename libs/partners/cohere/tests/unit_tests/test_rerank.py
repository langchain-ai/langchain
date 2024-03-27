"""Test chat model integration."""

from langchain_cohere import CohereRerank


def test_initialization() -> None:
    """Test chat model initialization."""
    CohereRerank(cohere_api_key="test")

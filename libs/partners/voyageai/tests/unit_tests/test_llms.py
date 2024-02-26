"""Test VoyageAI Chat API wrapper."""
from langchain_voyageai import VoyageAILLM


def test_initialization() -> None:
    """Test integration initialization."""
    VoyageAILLM()

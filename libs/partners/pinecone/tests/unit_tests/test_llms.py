"""Test Pinecone Chat API wrapper."""
from langchain_pinecone import PineconeLLM


def test_initialization() -> None:
    """Test integration initialization."""
    PineconeLLM()

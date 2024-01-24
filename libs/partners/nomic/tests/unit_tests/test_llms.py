"""Test Nomic Chat API wrapper."""
from langchain_nomic import NomicLLM


def test_initialization() -> None:
    """Test integration initialization."""
    NomicLLM()

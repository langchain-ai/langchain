"""Test Cohere Chat API wrapper."""
from langchain_cohere import CohereLLM


def test_initialization() -> None:
    """Test integration initialization."""
    CohereLLM()

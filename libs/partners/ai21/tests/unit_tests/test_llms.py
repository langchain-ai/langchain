"""Test AI21 Chat API wrapper."""
from langchain_ai21 import AI21LLM


def test_initialization() -> None:
    """Test integration initialization."""
    AI21LLM()

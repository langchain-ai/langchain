"""Test Prompty Chat API wrapper."""
from langchain_prompty import PromptyLLM


def test_initialization() -> None:
    """Test integration initialization."""
    PromptyLLM()

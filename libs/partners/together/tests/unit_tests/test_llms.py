"""Test Together Chat API wrapper."""
from langchain_together import TogetherLLM


def test_initialization() -> None:
    """Test integration initialization."""
    TogetherLLM()

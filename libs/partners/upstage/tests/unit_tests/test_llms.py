"""Test Upstage Chat API wrapper."""
from langchain_upstage import UpstageLLM


def test_initialization() -> None:
    """Test integration initialization."""
    UpstageLLM()

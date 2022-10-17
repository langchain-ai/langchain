"""Test Cohere API wrapper."""

from langchain.llms.cohere import Cohere


def test_cohere_call() -> None:
    """Test valid call to cohere."""
    llm = Cohere(max_tokens=10)
    output = llm("Say foo:")
    assert isinstance(output, str)

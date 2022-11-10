"""Test AI21 API wrapper."""

from langchain.llms.ai21 import AI21


def test_ai21_call() -> None:
    """Test valid call to ai21."""
    llm = AI21(maxTokens=10)
    output = llm("Say foo:")
    assert isinstance(output, str)

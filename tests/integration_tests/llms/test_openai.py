"""Test OpenAI API wrapper."""

from langchain.llms.openai import OpenAI


def test_cohere_call() -> None:
    """Test valid call to cohere."""
    llm = OpenAI(max_tokens=10)
    output = llm("Say foo:")
    assert isinstance(output, str)

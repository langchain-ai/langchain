"""Test Aleph Alpha API wrapper."""

from langchain.llms.aleph_alpha import AlephAlpha


def test_aleph_alpha_call() -> None:
    """Test valid call to cohere."""
    llm = AlephAlpha(maximum_tokens=10)
    output = llm("Say foo:")
    assert isinstance(output, str)

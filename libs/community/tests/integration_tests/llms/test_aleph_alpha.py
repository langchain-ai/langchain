"""Test Aleph Alpha API wrapper."""

from langchain_community.llms.aleph_alpha import AlephAlpha


def test_aleph_alpha_call() -> None:
    """Test valid call to cohere."""
    llm = AlephAlpha(maximum_tokens=10)  # type: ignore[call-arg]
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

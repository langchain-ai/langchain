"""Test Anyscale API wrapper."""

from langchain.llms.anyscale import AnyscaleLLM


def test_anyscale_call() -> None:
    """Test valid call to Anyscale."""
    llm = AnyscaleLLM()
    output = llm("Say foo:")
    assert isinstance(output, str)

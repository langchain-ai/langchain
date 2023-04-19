"""Test Camel API wrapper."""

from langchain.llms.camel import Camel


def test_camel_call() -> None:
    """Test valid call to Camel API."""
    llm = Camel()
    output = llm("Say foo:")
    assert isinstance(output, str)

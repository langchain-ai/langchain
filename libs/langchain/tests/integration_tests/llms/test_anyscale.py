"""Test Anyscale API wrapper."""

from langchain.llms.anyscale import Anyscale


def test_anyscale_call() -> None:
    """Test valid call to Anyscale."""
    llm = Anyscale()
    output = llm("Say foo:")
    assert isinstance(output, str)

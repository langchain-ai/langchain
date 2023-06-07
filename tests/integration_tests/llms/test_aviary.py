"""Test Anyscale API wrapper."""

from langchain.llms.aviary import Aviary


def test_aviary_call() -> None:
    """Test valid call to Anyscale."""
    llm = Aviary(model="test/model")
    output = llm("Say bar:")
    assert isinstance(output, str)

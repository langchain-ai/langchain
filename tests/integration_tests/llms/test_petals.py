"""Test Petals API wrapper."""

from langchain.llms.petals import Petals


def test_gooseai_call() -> None:
    """Test valid call to gooseai."""
    llm = Petals(max_new_tokens=10)
    output = llm("Say foo:")
    assert isinstance(output, str)

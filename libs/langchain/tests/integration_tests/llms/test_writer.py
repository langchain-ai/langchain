"""Test Writer API wrapper."""

from langchain.llms.writer import Writer


def test_writer_call() -> None:
    """Test valid call to Writer."""
    llm = Writer()
    output = llm("Say foo:")
    assert isinstance(output, str)

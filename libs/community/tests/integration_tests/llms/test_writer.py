"""Test Writer API wrapper."""

from langchain_community.llms.writer import Writer


def test_writer_call() -> None:
    """Test valid call to Writer."""
    llm = Writer()
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

"""Test NLPCloud API wrapper."""

from langchain.llms.nlpcloud import NLPCloud


def test_nlpcloud_call() -> None:
    """Test valid call to nlpcloud."""
    llm = NLPCloud(max_length=10)
    output = llm("Say foo:")
    assert isinstance(output, str)

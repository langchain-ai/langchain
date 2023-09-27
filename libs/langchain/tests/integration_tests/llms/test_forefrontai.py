"""Test ForefrontAI API wrapper."""

from langchain.llms.forefrontai import ForefrontAI


def test_forefrontai_call() -> None:
    """Test valid call to forefrontai."""
    llm = ForefrontAI(length=10)
    output = llm("Say foo:")
    assert isinstance(output, str)

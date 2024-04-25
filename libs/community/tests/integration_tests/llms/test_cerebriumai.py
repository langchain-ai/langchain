"""Test CerebriumAI API wrapper."""

from langchain_community.llms.cerebriumai import CerebriumAI


def test_cerebriumai_call() -> None:
    """Test valid call to cerebriumai."""
    llm = CerebriumAI(max_length=10)
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

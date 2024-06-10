"""Test ForefrontAI API wrapper."""

from langchain_community.llms.forefrontai import ForefrontAI


def test_forefrontai_call() -> None:
    """Test valid call to forefrontai."""
    llm = ForefrontAI(length=10)  # type: ignore[call-arg]
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

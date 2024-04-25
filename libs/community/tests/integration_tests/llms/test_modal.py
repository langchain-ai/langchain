"""Test Modal API wrapper."""

from langchain_community.llms.modal import Modal


def test_modal_call() -> None:
    """Test valid call to Modal."""
    llm = Modal()
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)

"""Test Modal API wrapper."""

from langchain.llms.modal import Modal


def test_modal_call() -> None:
    """Test valid call to Modal."""
    llm = Modal()
    output = llm("Say foo:")
    assert isinstance(output, str)

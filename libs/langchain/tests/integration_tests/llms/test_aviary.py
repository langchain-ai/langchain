"""Test Anyscale API wrapper."""

from langchain.llms.aviary import Aviary


def test_aviary_call() -> None:
    """Test valid call to Anyscale."""
    llm = Aviary()
    output = llm("Say bar:")
    print(f"llm answer:\n{output}")
    assert isinstance(output, str)

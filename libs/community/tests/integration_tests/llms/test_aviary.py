"""Test Anyscale API wrapper."""

from langchain_community.llms.aviary import Aviary


def test_aviary_call() -> None:
    """Test valid call to Anyscale."""
    llm = Aviary()
    output = llm.invoke("Say bar:")
    print(f"llm answer:\n{output}")  # noqa: T201
    assert isinstance(output, str)

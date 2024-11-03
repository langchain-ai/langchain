"""Integration test for SerpAPI."""

from langchain_community.utilities import SerpAPIWrapper


def test_call() -> None:
    """Test that call gives the correct answer."""
    chain = SerpAPIWrapper()  # type: ignore[call-arg]
    output = chain.run("What was Obama's first name?")
    assert output == "Barack Hussein Obama II"

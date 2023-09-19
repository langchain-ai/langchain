"""Integration test for SerpAPI."""
from langchain.utilities import SerpAPIWrapper


def test_call() -> None:
    """Test that call gives the correct answer."""
    chain = SerpAPIWrapper()
    output = chain.run("What was Obama's first name?")
    assert output == "Barack Hussein Obama II"

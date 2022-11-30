"""Integration test for SerpAPI."""
from langchain.serpapi import SerpAPIWrapper


def test_call() -> None:
    """Test that call gives the correct answer."""
    chain = SerpAPIWrapper()
    output = chain.run("What was Obama's first name?")
    assert output == "Barack Hussein Obama II"

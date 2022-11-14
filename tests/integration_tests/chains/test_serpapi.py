"""Integration test for SerpAPI."""
from langchain.chains.serpapi import SerpAPIChain


def test_call() -> None:
    """Test that call gives the correct answer."""
    chain = SerpAPIChain()
    output = chain.run("What was Obama's first name?")
    assert output == "Barack Hussein Obama II"

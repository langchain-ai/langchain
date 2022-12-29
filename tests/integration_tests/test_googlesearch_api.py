"""Integration test for SerpAPI."""
from googleapiclient.discovery import build

from langchain.google_search import GoogleSearchAPIWrapper


def test_call() -> None:
    """Test that call gives the correct answer."""
    chain = GoogleSearchAPIWrapper()
    output = chain.run("What was Obama's first name?")
    assert "Barack Hussein Obama II" in output

"""Integration test for Google Search API Wrapper."""
from langchain.utilities.google_search import GoogleSearchAPIWrapper


def test_call() -> None:
    """Test that call gives the correct answer."""
    search = GoogleSearchAPIWrapper()
    output = search.run("What was Obama's first name?")
    assert "Barack Hussein Obama II" in output

"""Integration test for Serper.dev's Google Search API Wrapper."""
from langchain.utilities.google_serper import GoogleSerperAPIWrapper


def test_call() -> None:
    """Test that call gives the correct answer."""
    search = GoogleSerperAPIWrapper()
    output = search.run("What was Obama's first name?")
    assert "Barack Hussein Obama II" in output

"""Integration test for Google Search API Wrapper."""
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper


def test_call() -> None:
    """Test that call gives the correct answer."""
    search = WolframAlphaAPIWrapper()
    output = search.run("What was Obama's first name?")
    assert "Barack Hussein Obama II" in output

"""Integration test for Google Search API Wrapper."""
from langchain.utilities.dalle_image_generator import DallEAPIWrapper


def test_call() -> None:
    """Test that call returns a URL in the output."""
    search = DallEAPIWrapper()
    output = search.run("volcano island")
    assert "https://oaidalleapi" in output
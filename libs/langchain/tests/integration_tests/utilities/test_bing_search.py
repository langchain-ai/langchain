"""Integration test for Bing Search API Wrapper."""
from langchain.utilities.bing_search import BingSearchAPIWrapper


def test_call() -> None:
    """Test that call gives the correct answer."""
    search = BingSearchAPIWrapper()
    output = search.run("Obama's first name")
    assert "Barack Hussein Obama" in output


def test_results() -> None:
    """Test that call gives the correct answer."""
    search = BingSearchAPIWrapper()
    results = search.results("Obama's first name", num_results=5)
    result_contents = "\n".join(
        f"{result['title']}: {result['snippet']}" for result in results
    )
    assert "Barack Hussein Obama" in result_contents

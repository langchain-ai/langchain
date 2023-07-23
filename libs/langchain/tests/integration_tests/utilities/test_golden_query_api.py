"""Integration test for Golden API Wrapper."""
import json

from langchain.utilities.golden_query import GoldenQueryAPIWrapper


def test_call() -> None:
    """Test that call gives the correct answer."""
    search = GoldenQueryAPIWrapper()
    output = json.loads(search.run("companies in nanotech"))
    assert len(output.get("results", [])) > 0

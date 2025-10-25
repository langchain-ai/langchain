"""Integration tests for SERPEX Search Tool."""

import os

import pytest

from langchain_serpex import SerpexSearchResults


@pytest.mark.compile
def test_import() -> None:
    """Test that the module can be imported."""
    from langchain_serpex import SerpexSearchResults  # noqa: F401


@pytest.mark.skipif("SERPEX_API_KEY" not in os.environ, reason="SERPEX_API_KEY not set")
def test_serpex_search_integration() -> None:
    """Integration test with real API."""
    api_key = os.getenv("SERPEX_API_KEY")
    if not api_key:
        pytest.skip("SERPEX_API_KEY not set")

    tool = SerpexSearchResults(api_key=api_key)
    result = tool._run("Python programming language")

    assert result
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.skipif("SERPEX_API_KEY" not in os.environ, reason="SERPEX_API_KEY not set")
def test_serpex_different_engines() -> None:
    """Test search with different engines."""
    api_key = os.getenv("SERPEX_API_KEY")
    if not api_key:
        pytest.skip("SERPEX_API_KEY not set")

    engines = ["google", "bing", "duckduckgo"]

    for engine in engines:
        tool = SerpexSearchResults(api_key=api_key, engine=engine)
        result = tool._run("test query")
        assert result
        assert isinstance(result, str)

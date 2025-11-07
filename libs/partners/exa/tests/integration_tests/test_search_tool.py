"""Integration tests for Exa search tool."""

from langchain_exa import (
    ExaSearchResults,  # type: ignore[import-not-found, import-not-found]
)


def test_search_tool() -> None:
    """Test that the Exa search tool works."""
    tool = ExaSearchResults()
    res = tool.invoke({"query": "best time to visit japan", "num_results": 5})
    print(res)  # noqa: T201
    assert not isinstance(res, str)  # str means error for this tool\


def test_search_tool_advanced_features() -> None:
    """Test advanced features of the Exa search tool."""
    tool = ExaSearchResults()
    res = tool.invoke(
        {
            "query": "best time to visit japan",
            "num_results": 3,
            "text_contents_options": {"max_characters": 1000},
            "summary": True,
            "type": "auto",
        }
    )
    print(res)  # noqa: T201
    assert not isinstance(res, str)  # str means error for this tool
    assert len(res.results) == 3
    # Verify summary exists
    assert hasattr(res.results[0], "summary")
    # Verify text was limited
    assert len(res.results[0].text) <= 1000

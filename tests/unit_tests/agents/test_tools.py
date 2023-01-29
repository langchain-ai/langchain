"""Test tool utils."""
import pytest

from langchain.agents.tools import Tool, tool


def test_unnamed_decorator() -> None:
    """Test functionality with unnamed decorator."""

    @tool
    def search_api(query: str) -> str:
        """Search the API for the query."""
        return "API result"

    assert isinstance(search_api, Tool)
    assert search_api.name == "search_api"
    assert not search_api.return_direct
    assert search_api("test") == "API result"


def test_named_tool_decorator() -> None:
    """Test functionality when arguments are provided as input to decorator."""

    @tool("search")
    def search_api(query: str) -> str:
        """Search the API for the query."""
        return "API result"

    assert isinstance(search_api, Tool)
    assert search_api.name == "search"
    assert not search_api.return_direct


def test_named_tool_decorator_return_direct() -> None:
    """Test functionality when arguments and return direct are provided as input."""

    @tool("search", return_direct=True)
    def search_api(query: str) -> str:
        """Search the API for the query."""
        return "API result"

    assert isinstance(search_api, Tool)
    assert search_api.name == "search"
    assert search_api.return_direct


def test_unnamed_tool_decorator_return_direct() -> None:
    """Test functionality when only return direct is provided."""

    @tool(return_direct=True)
    def search_api(query: str) -> str:
        """Search the API for the query."""
        return "API result"

    assert isinstance(search_api, Tool)
    assert search_api.name == "search_api"
    assert search_api.return_direct


def test_missing_docstring() -> None:
    """Test error is raised when docstring is missing."""
    # expect to throw a value error if theres no docstring
    with pytest.raises(AssertionError):

        @tool
        def search_api(query: str) -> str:
            return "API result"

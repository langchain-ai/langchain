import pytest

from langchain.agents.tools import Tool, tool


def test_unnamed_decorator() -> None:
    @tool
    def search_api(query: str) -> str:
        """Searches the API for the query."""
        return "API result"

    assert isinstance(search_api, Tool)
    assert search_api.name == "search_api"
    assert search_api.return_direct == False
    assert search_api("test") == "API result"


def test_named_tool_decorator() -> None:
    @tool("search")
    def search_api(query: str) -> str:
        """Searches the API for the query."""
        return "API result"

    assert isinstance(search_api, Tool)
    assert search_api.name == "search"
    assert search_api.return_direct == False


def test_named_tool_decorator_return_direct() -> None:
    @tool("search", return_direct=True)
    def search_api(query: str) -> str:
        """Searches the API for the query."""
        return "API result"

    assert isinstance(search_api, Tool)
    assert search_api.name == "search"
    assert search_api.return_direct == True


def test_unnamed_tool_decorator_return_direct() -> None:
    @tool(return_direct=True)
    def search_api(query: str) -> str:
        """Searches the API for the query."""
        return "API result"

    assert isinstance(search_api, Tool)
    assert search_api.name == "search_api"
    assert search_api.return_direct == True


def test_missing_docstring() -> None:
    # except to throw a value error if theres too many arguments
    with pytest.raises(AssertionError):

        @tool
        def search_api(query: str, arg2: str) -> str:
            return "API result"

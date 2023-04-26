"""Test tool utils."""

import pytest

from langchain.agents.tools import Tool, tool
from langchain.tools.base import BaseTool, StringSchema


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
    assert search_api.args_schema == StringSchema


def test_base_tool_inheritance_base_schema() -> None:
    """Test schema is correctly inferred when inheriting from BaseTool."""

    class _MockSimpleTool(BaseTool):
        name = "simple_tool"
        description = "A Simple Tool"

        def _run(self, tool_input: str) -> str:
            return f"{tool_input}"

        async def _arun(self, tool_input: str) -> str:
            raise NotImplementedError

    simple_tool = _MockSimpleTool()
    assert simple_tool.args_schema == StringSchema
    expected_args = {"tool_input": {"title": "Tool Input", "type": "string"}}
    assert simple_tool.args == expected_args


def test_tool_lambda_args_schema() -> None:
    """Test args schema inference when the tool argument is a lambda function."""

    tool = Tool(
        name="tool",
        description="A tool",
        func=lambda tool_input: tool_input,
    )
    assert tool.args_schema == StringSchema
    expected_args = {"tool_input": {"title": "Tool Input", "type": "string"}}
    assert tool.args == expected_args


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


def test_create_tool_positional_args() -> None:
    """Test that positional arguments are allowed."""
    test_tool = Tool("test_name", lambda x: x, "test_description")
    assert test_tool("foo") == "foo"
    assert test_tool.name == "test_name"
    assert test_tool.description == "test_description"


def test_create_tool_keyword_args() -> None:
    """Test that keyword arguments are allowed."""
    test_tool = Tool(name="test_name", func=lambda x: x, description="test_description")
    assert test_tool("foo") == "foo"
    assert test_tool.name == "test_name"
    assert test_tool.description == "test_description"


@pytest.mark.asyncio
async def test_create_async_tool() -> None:
    """Test that async tools are allowed."""

    async def _test_func(x: str) -> str:
        return x

    test_tool = Tool(
        name="test_name",
        func=lambda x: x,
        description="test_description",
        coroutine=_test_func,
    )
    assert test_tool("foo") == "foo"
    assert test_tool.name == "test_name"
    assert test_tool.description == "test_description"
    assert test_tool.coroutine is not None
    assert await test_tool.arun("foo") == "foo"

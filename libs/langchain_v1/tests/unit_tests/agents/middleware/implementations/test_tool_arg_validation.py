"""Tests for ToolArgValidationMiddleware."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from langchain.agents.factory import create_agent
from langchain.agents.middleware.tool_arg_validation import (
    ToolArgValidationMiddleware,
    _extract_ai_message,
    _format_validation_errors,
    _strip_empty_values,
    _validate_with_pydantic,
    _ValidationError,
)
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from tests.unit_tests.agents.model import FakeToolCallingModel

# ------------------------------------------------------------------ #
# Fixtures & helpers
# ------------------------------------------------------------------ #


class SearchInput(BaseModel):
    """Input schema for search tool."""

    query: str = Field(description="Search query")
    max_results: int = Field(default=10, description="Maximum results to return")


class FilterInput(BaseModel):
    """Input schema with required and optional fields."""

    name: str
    age: int
    email: str | None = None


class NestedInput(BaseModel):
    """Input schema with nested object."""

    query: str
    options: dict[str, Any] | None = None


@tool(args_schema=SearchInput)
def search_tool(query: str, max_results: int = 10) -> str:
    """Search for information."""
    return f"Results for {query}"


@tool(args_schema=FilterInput)
def filter_tool(name: str, age: int, email: str | None = None) -> str:
    """Filter records."""
    return f"Filtered: {name}"


@tool
def auto_schema_tool(x: str, count: int = 5) -> str:
    """A tool with auto-generated schema."""
    return f"{x}: {count}"


def _make_request(
    tools: list[Any] | None = None,
    messages: list[Any] | None = None,
) -> ModelRequest[None]:
    """Create a minimal ModelRequest for testing."""
    model = MagicMock()
    return ModelRequest(
        model=model,
        messages=messages or [HumanMessage("test")],
        tools=tools or [],
    )


def _make_response(ai_msg: AIMessage) -> ModelResponse[Any]:
    """Wrap an AIMessage in a ModelResponse."""
    return ModelResponse(result=[ai_msg])


def _make_tool_call(name: str, args: dict[str, Any], call_id: str = "call_1") -> ToolCall:
    """Create a ToolCall dict."""
    return ToolCall(name=name, args=args, id=call_id)


@pytest.fixture
def json_schema_tool() -> Any:
    """Create a mock tool with a dict-based JSON Schema (MCP-style)."""
    mock_tool = MagicMock()
    mock_tool.name = "mcp_search"
    mock_tool.args_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Max results"},
        },
        "required": ["query"],
    }
    return mock_tool


# ------------------------------------------------------------------ #
# Initialization tests
# ------------------------------------------------------------------ #


def test_initialization_defaults() -> None:
    """Test default initialization values."""
    mw = ToolArgValidationMiddleware()
    assert mw.max_retries == 2
    assert mw.strip_empty_values is True
    assert mw.tools == []


def test_initialization_custom() -> None:
    """Test custom initialization values."""
    mw = ToolArgValidationMiddleware(max_retries=5, strip_empty_values=False)
    assert mw.max_retries == 5
    assert mw.strip_empty_values is False


def test_initialization_invalid_max_retries() -> None:
    """Test that max_retries < 1 raises ValueError."""
    with pytest.raises(ValueError, match="max_retries must be >= 1"):
        ToolArgValidationMiddleware(max_retries=0)

    with pytest.raises(ValueError, match="max_retries must be >= 1"):
        ToolArgValidationMiddleware(max_retries=-1)


def test_initialization_keyword_only() -> None:
    """Test that __init__ only accepts keyword arguments."""
    with pytest.raises(TypeError):
        ToolArgValidationMiddleware(3)  # type: ignore[misc]


# ------------------------------------------------------------------ #
# _strip_empty_values tests
# ------------------------------------------------------------------ #


def test_strip_empty_values_removes_none() -> None:
    """Test that None values are stripped."""
    assert _strip_empty_values({"a": 1, "b": None}) == {"a": 1}


def test_strip_empty_values_removes_empty_dict() -> None:
    """Test that empty dict values are stripped."""
    assert _strip_empty_values({"a": 1, "b": {}}) == {"a": 1}


def test_strip_empty_values_removes_empty_list() -> None:
    """Test that empty list values are stripped."""
    assert _strip_empty_values({"a": 1, "b": []}) == {"a": 1}


def test_strip_empty_values_recursive() -> None:
    """Test recursive stripping of nested dicts."""
    result = _strip_empty_values({"a": {"b": None, "c": 1}, "d": None})
    assert result == {"a": {"c": 1}}


def test_strip_empty_values_preserves_valid() -> None:
    """Test that valid values are preserved."""
    data = {"a": 1, "b": "hello", "c": [1, 2], "d": {"e": True}}
    assert _strip_empty_values(data) == data


def test_strip_empty_values_nested_list() -> None:
    """Test stripping inside list items."""
    result = _strip_empty_values({"items": [{"a": 1, "b": None}, {"c": 2}]})
    assert result == {"items": [{"a": 1}, {"c": 2}]}


def test_strip_empty_values_preserves_falsy_non_empty() -> None:
    """Test that falsy but non-empty values (0, False, '') are preserved."""
    data = {"a": 0, "b": False, "c": ""}
    assert _strip_empty_values(data) == data


def test_strip_empty_values_empty_input() -> None:
    """Test stripping from an empty dict returns empty dict."""
    assert _strip_empty_values({}) == {}


def test_strip_empty_values_all_empty() -> None:
    """Test stripping when all values are empty."""
    assert _strip_empty_values({"a": None, "b": {}, "c": []}) == {}


def test_strip_empty_values_deeply_nested() -> None:
    """Test stripping deeply nested empty values."""
    result = _strip_empty_values(
        {
            "a": {
                "b": {
                    "c": None,
                    "d": {"e": []},
                    "f": 42,
                }
            }
        }
    )
    assert result == {"a": {"b": {"d": {}, "f": 42}}}


# ------------------------------------------------------------------ #
# _validate_with_pydantic tests
# ------------------------------------------------------------------ #


def test_validate_pydantic_valid() -> None:
    """Test Pydantic validation with valid args."""
    errors = _validate_with_pydantic(SearchInput, {"query": "test"})
    assert errors == []


def test_validate_pydantic_valid_all_fields() -> None:
    """Test Pydantic validation with all fields provided."""
    errors = _validate_with_pydantic(SearchInput, {"query": "test", "max_results": 20})
    assert errors == []


def test_validate_pydantic_missing_required() -> None:
    """Test Pydantic validation with missing required field."""
    errors = _validate_with_pydantic(SearchInput, {})
    assert len(errors) >= 1
    assert any("query" in e.path for e in errors)


def test_validate_pydantic_wrong_type() -> None:
    """Test Pydantic validation with wrong type."""
    errors = _validate_with_pydantic(SearchInput, {"query": 123})
    assert len(errors) >= 1


def test_validate_pydantic_extra_optional_ok() -> None:
    """Test Pydantic validation passes with optional fields omitted."""
    errors = _validate_with_pydantic(FilterInput, {"name": "Alice", "age": 30})
    assert errors == []


def test_validate_pydantic_multiple_errors() -> None:
    """Test Pydantic validation returns multiple errors."""
    errors = _validate_with_pydantic(FilterInput, {})
    # Must report both 'name' and 'age' as missing
    assert len(errors) >= 2
    paths = {e.path for e in errors}
    assert "name" in paths
    assert "age" in paths


def test_validate_pydantic_empty_args() -> None:
    """Test Pydantic validation with empty args dict."""
    errors = _validate_with_pydantic(SearchInput, {})
    assert len(errors) >= 1


# ------------------------------------------------------------------ #
# _format_validation_errors tests
# ------------------------------------------------------------------ #


def test_format_validation_errors() -> None:
    """Test error message formatting."""
    errors = [
        _ValidationError(path="query", message="Field required"),
        _ValidationError(path="max_results", message="Input should be a valid integer"),
    ]
    result = _format_validation_errors("search", errors)
    assert "Tool 'search' argument validation failed" in result
    assert "[query] Field required" in result
    assert "[max_results] Input should be a valid integer" in result
    assert "Hint:" in result


def test_format_validation_errors_single() -> None:
    """Test error message formatting with a single error."""
    errors = [_ValidationError(path="(root)", message="object is not valid")]
    result = _format_validation_errors("my_tool", errors)
    assert "Tool 'my_tool'" in result
    assert "[(root)]" in result


# ------------------------------------------------------------------ #
# _extract_ai_message tests
# ------------------------------------------------------------------ #


def test_extract_ai_message_present() -> None:
    """Test extracting AIMessage from response."""
    ai_msg = AIMessage(content="hello")
    response = ModelResponse(result=[ai_msg])
    assert _extract_ai_message(response) is ai_msg


def test_extract_ai_message_absent() -> None:
    """Test extracting AIMessage when none present."""
    response = ModelResponse(result=[HumanMessage("hello")])
    assert _extract_ai_message(response) is None


def test_extract_ai_message_empty() -> None:
    """Test extracting AIMessage from empty response."""
    response = ModelResponse(result=[])
    assert _extract_ai_message(response) is None


def test_extract_ai_message_returns_first() -> None:
    """Test that first AIMessage is returned when multiple present."""
    first = AIMessage(content="first")
    second = AIMessage(content="second")
    response = ModelResponse(result=[HumanMessage("q"), first, second])
    assert _extract_ai_message(response) is first


# ------------------------------------------------------------------ #
# wrap_model_call — sync tests
# ------------------------------------------------------------------ #


def test_valid_tool_call_passes_through() -> None:
    """Test that valid tool calls pass through without retry."""
    mw = ToolArgValidationMiddleware()

    ai_msg = AIMessage(
        content="",
        tool_calls=[_make_tool_call("search_tool", {"query": "test"})],
    )
    request = _make_request(tools=[search_tool])
    call_count = {"n": 0}

    def handler(req: ModelRequest[None]) -> ModelResponse[Any]:
        call_count["n"] += 1
        return _make_response(ai_msg)

    result = mw.wrap_model_call(request, handler)
    assert call_count["n"] == 1
    assert result.result[0] is ai_msg


def test_invalid_tool_call_triggers_retry() -> None:
    """Test that invalid tool calls trigger a retry with corrected args."""
    mw = ToolArgValidationMiddleware(max_retries=2)

    bad_msg = AIMessage(
        content="",
        id="bad",
        tool_calls=[_make_tool_call("search_tool", {})],  # missing 'query'
    )
    good_msg = AIMessage(
        content="",
        id="good",
        tool_calls=[_make_tool_call("search_tool", {"query": "fixed"})],
    )

    request = _make_request(tools=[search_tool])
    call_count = {"n": 0}

    def handler(req: ModelRequest[None]) -> ModelResponse[Any]:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _make_response(bad_msg)
        return _make_response(good_msg)

    result = mw.wrap_model_call(request, handler)
    assert call_count["n"] == 2
    assert result.result[0] is good_msg


def test_retry_messages_include_errors() -> None:
    """Test that retry request contains original messages, failed AI, and error ToolMessages."""
    mw = ToolArgValidationMiddleware(max_retries=1)

    bad_msg = AIMessage(
        content="",
        id="bad",
        tool_calls=[_make_tool_call("search_tool", {})],
    )
    good_msg = AIMessage(
        content="",
        id="good",
        tool_calls=[_make_tool_call("search_tool", {"query": "ok"})],
    )

    original_messages = [HumanMessage("find something")]
    request = _make_request(tools=[search_tool], messages=original_messages)
    retry_requests: list[ModelRequest[None]] = []

    def handler(req: ModelRequest[None]) -> ModelResponse[Any]:
        retry_requests.append(req)
        if len(retry_requests) == 1:
            return _make_response(bad_msg)
        return _make_response(good_msg)

    mw.wrap_model_call(request, handler)

    assert len(retry_requests) == 2
    retry_msgs = retry_requests[1].messages

    # Original messages preserved at the start
    assert retry_msgs[0] is original_messages[0]
    # Failed AIMessage appended
    assert isinstance(retry_msgs[1], AIMessage)
    assert retry_msgs[1] is bad_msg
    # Error ToolMessage appended
    tool_messages = [m for m in retry_msgs if isinstance(m, ToolMessage)]
    assert len(tool_messages) >= 1
    assert "validation failed" in tool_messages[0].content.lower()


def test_max_retries_exhausted_passes_through() -> None:
    """Test that after max retries, last response is returned."""
    mw = ToolArgValidationMiddleware(max_retries=1)

    bad_msg = AIMessage(
        content="",
        id="bad",
        tool_calls=[_make_tool_call("search_tool", {})],
    )

    request = _make_request(tools=[search_tool])
    call_count = {"n": 0}

    def handler(req: ModelRequest[None]) -> ModelResponse[Any]:
        call_count["n"] += 1
        return _make_response(bad_msg)

    result = mw.wrap_model_call(request, handler)
    # 1 initial + 1 retry = 2 calls
    assert call_count["n"] == 2
    # Last response (still invalid) is passed through
    assert result.result[0] is bad_msg


def test_multiple_retries_succeed_on_last() -> None:
    """Test that validation succeeds on the last allowed retry."""
    mw = ToolArgValidationMiddleware(max_retries=3)

    bad_msg = AIMessage(
        content="",
        id="bad",
        tool_calls=[_make_tool_call("search_tool", {})],
    )
    good_msg = AIMessage(
        content="",
        id="good",
        tool_calls=[_make_tool_call("search_tool", {"query": "finally"})],
    )

    request = _make_request(tools=[search_tool])
    call_count = {"n": 0}

    def handler(req: ModelRequest[None]) -> ModelResponse[Any]:
        call_count["n"] += 1
        if call_count["n"] <= 3:  # fails on attempts 1, 2, 3
            return _make_response(bad_msg)
        return _make_response(good_msg)  # succeeds on attempt 4 (retry 3)

    result = mw.wrap_model_call(request, handler)
    assert call_count["n"] == 4  # 1 initial + 3 retries
    assert result.result[0] is good_msg


def test_no_tool_calls_passes_through() -> None:
    """Test that messages without tool calls pass through."""
    mw = ToolArgValidationMiddleware()

    ai_msg = AIMessage(content="Just text, no tools")
    request = _make_request(tools=[search_tool])

    def handler(req: ModelRequest[None]) -> ModelResponse[Any]:
        return _make_response(ai_msg)

    result = mw.wrap_model_call(request, handler)
    assert result.result[0] is ai_msg


def test_empty_tool_calls_list_passes_through() -> None:
    """Test that an AIMessage with empty tool_calls list passes through."""
    mw = ToolArgValidationMiddleware()

    ai_msg = AIMessage(content="Done", tool_calls=[])
    request = _make_request(tools=[search_tool])

    def handler(req: ModelRequest[None]) -> ModelResponse[Any]:
        return _make_response(ai_msg)

    result = mw.wrap_model_call(request, handler)
    assert result.result[0] is ai_msg


def test_unknown_tool_passes_through() -> None:
    """Test that tool calls for unknown tools are not validated."""
    mw = ToolArgValidationMiddleware()

    ai_msg = AIMessage(
        content="",
        tool_calls=[_make_tool_call("unknown_tool", {"whatever": True})],
    )
    request = _make_request(tools=[search_tool])

    def handler(req: ModelRequest[None]) -> ModelResponse[Any]:
        return _make_response(ai_msg)

    result = mw.wrap_model_call(request, handler)
    assert result.result[0] is ai_msg


def test_auto_schema_tool_valid() -> None:
    """Test validation with auto-generated schema (no explicit args_schema)."""
    mw = ToolArgValidationMiddleware()

    ai_msg = AIMessage(
        content="",
        tool_calls=[_make_tool_call("auto_schema_tool", {"x": "hello"})],
    )
    request = _make_request(tools=[auto_schema_tool])
    call_count = {"n": 0}

    def handler(req: ModelRequest[None]) -> ModelResponse[Any]:
        call_count["n"] += 1
        return _make_response(ai_msg)

    result = mw.wrap_model_call(request, handler)
    assert call_count["n"] == 1
    assert result.result[0] is ai_msg


def test_strip_empty_values_before_validation() -> None:
    """Test that empty values are stripped before validation."""
    mw = ToolArgValidationMiddleware(strip_empty_values=True)

    # 'email' is optional — stripping None means it's just omitted
    ai_msg = AIMessage(
        content="",
        tool_calls=[_make_tool_call("filter_tool", {"name": "Alice", "age": 30, "email": None})],
    )
    request = _make_request(tools=[filter_tool])
    call_count = {"n": 0}

    def handler(req: ModelRequest[None]) -> ModelResponse[Any]:
        call_count["n"] += 1
        return _make_response(ai_msg)

    mw.wrap_model_call(request, handler)
    assert call_count["n"] == 1  # no retry needed


def test_strip_empty_values_updates_args_in_place() -> None:
    """Test that stripping updates the tool_call args dict in place."""
    mw = ToolArgValidationMiddleware(strip_empty_values=True)

    tc = _make_tool_call("filter_tool", {"name": "Bob", "age": 25, "email": None})
    ai_msg = AIMessage(content="", tool_calls=[tc])
    request = _make_request(tools=[filter_tool])

    def handler(req: ModelRequest[None]) -> ModelResponse[Any]:
        return _make_response(ai_msg)

    mw.wrap_model_call(request, handler)
    # After stripping, 'email' should be removed from args
    assert "email" not in ai_msg.tool_calls[0]["args"]


def test_no_strip_empty_values() -> None:
    """Test that stripping can be disabled."""
    mw = ToolArgValidationMiddleware(strip_empty_values=False)

    # email=None should still pass since it's Optional
    ai_msg = AIMessage(
        content="",
        tool_calls=[_make_tool_call("filter_tool", {"name": "Alice", "age": 30, "email": None})],
    )
    request = _make_request(tools=[filter_tool])
    call_count = {"n": 0}

    def handler(req: ModelRequest[None]) -> ModelResponse[Any]:
        call_count["n"] += 1
        return _make_response(ai_msg)

    mw.wrap_model_call(request, handler)
    assert call_count["n"] == 1  # None is valid for Optional[str]


def test_strip_required_field_causes_validation_error() -> None:
    """Test that stripping a required field (set to None) exposes it as missing."""
    mw = ToolArgValidationMiddleware(strip_empty_values=True, max_retries=1)

    # 'query' is required but set to None — stripping makes it absent → validation error
    bad_msg = AIMessage(
        content="",
        id="bad",
        tool_calls=[_make_tool_call("search_tool", {"query": None})],
    )
    good_msg = AIMessage(
        content="",
        id="good",
        tool_calls=[_make_tool_call("search_tool", {"query": "fixed"})],
    )

    request = _make_request(tools=[search_tool])
    call_count = {"n": 0}

    def handler(req: ModelRequest[None]) -> ModelResponse[Any]:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _make_response(bad_msg)
        return _make_response(good_msg)

    result = mw.wrap_model_call(request, handler)
    assert call_count["n"] == 2  # retry triggered
    assert result.result[0] is good_msg


# ------------------------------------------------------------------ #
# Mixed valid/invalid batch tests
# ------------------------------------------------------------------ #


def test_mixed_batch_all_get_messages() -> None:
    """Test that when one tool call fails, all get ToolMessages."""
    mw = ToolArgValidationMiddleware(max_retries=1)

    ai_msg = AIMessage(
        content="",
        id="mixed",
        tool_calls=[
            _make_tool_call("search_tool", {"query": "valid"}, call_id="valid_1"),
            _make_tool_call("search_tool", {}, call_id="invalid_1"),  # missing query
        ],
    )

    good_msg = AIMessage(
        content="",
        id="good",
        tool_calls=[
            _make_tool_call("search_tool", {"query": "a"}, call_id="ok_1"),
            _make_tool_call("search_tool", {"query": "b"}, call_id="ok_2"),
        ],
    )

    request = _make_request(tools=[search_tool])
    retry_requests: list[ModelRequest[None]] = []

    def handler(req: ModelRequest[None]) -> ModelResponse[Any]:
        retry_requests.append(req)
        if len(retry_requests) == 1:
            return _make_response(ai_msg)
        return _make_response(good_msg)

    mw.wrap_model_call(request, handler)

    # Check retry messages contain both error and "not executed" messages
    retry_msgs = retry_requests[1].messages
    tool_messages = [m for m in retry_msgs if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 2  # one error + one "not executed"

    contents = [m.content for m in tool_messages]
    assert any("validation failed" in c.lower() for c in contents)
    assert any("not executed" in c.lower() for c in contents)


def test_mixed_batch_tool_call_ids_match() -> None:
    """Test that each ToolMessage has the correct tool_call_id."""
    mw = ToolArgValidationMiddleware(max_retries=1)

    ai_msg = AIMessage(
        content="",
        id="mixed",
        tool_calls=[
            _make_tool_call("search_tool", {"query": "ok"}, call_id="good_id"),
            _make_tool_call("search_tool", {}, call_id="bad_id"),
        ],
    )

    request = _make_request(tools=[search_tool])
    retry_requests: list[ModelRequest[None]] = []

    def handler(req: ModelRequest[None]) -> ModelResponse[Any]:
        retry_requests.append(req)
        if len(retry_requests) == 1:
            return _make_response(ai_msg)
        # Return valid on second call
        return _make_response(
            AIMessage(
                content="",
                tool_calls=[_make_tool_call("search_tool", {"query": "x"}, call_id="c1")],
            )
        )

    mw.wrap_model_call(request, handler)

    tool_messages = [m for m in retry_requests[1].messages if isinstance(m, ToolMessage)]
    call_ids = {m.tool_call_id for m in tool_messages}
    assert call_ids == {"good_id", "bad_id"}


def test_all_tool_calls_invalid_in_batch() -> None:
    """Test that when all tool calls fail, all get error ToolMessages (no 'not executed')."""
    mw = ToolArgValidationMiddleware(max_retries=1)

    ai_msg = AIMessage(
        content="",
        id="all_bad",
        tool_calls=[
            _make_tool_call("search_tool", {}, call_id="bad_1"),
            _make_tool_call("search_tool", {}, call_id="bad_2"),
        ],
    )

    request = _make_request(tools=[search_tool])
    retry_requests: list[ModelRequest[None]] = []

    def handler(req: ModelRequest[None]) -> ModelResponse[Any]:
        retry_requests.append(req)
        if len(retry_requests) == 1:
            return _make_response(ai_msg)
        return _make_response(
            AIMessage(
                content="",
                tool_calls=[_make_tool_call("search_tool", {"query": "x"}, call_id="c1")],
            )
        )

    mw.wrap_model_call(request, handler)

    tool_messages = [m for m in retry_requests[1].messages if isinstance(m, ToolMessage)]
    # Both should be error messages, none should say "not executed"
    assert len(tool_messages) == 2
    for msg in tool_messages:
        assert "validation failed" in msg.content.lower()
        assert "not executed" not in msg.content.lower()


def test_mixed_known_and_unknown_tools() -> None:
    """Test batch with known (validated) and unknown (skipped) tools."""
    mw = ToolArgValidationMiddleware(max_retries=1)

    ai_msg = AIMessage(
        content="",
        id="mixed",
        tool_calls=[
            _make_tool_call("search_tool", {}, call_id="known_bad"),  # invalid
            _make_tool_call("unknown_tool", {"x": 1}, call_id="unknown_ok"),  # skipped
        ],
    )

    request = _make_request(tools=[search_tool])
    retry_requests: list[ModelRequest[None]] = []

    def handler(req: ModelRequest[None]) -> ModelResponse[Any]:
        retry_requests.append(req)
        if len(retry_requests) == 1:
            return _make_response(ai_msg)
        return _make_response(AIMessage(content="fixed"))

    mw.wrap_model_call(request, handler)

    tool_messages = [m for m in retry_requests[1].messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 2

    ids_to_content = {m.tool_call_id: m.content for m in tool_messages}
    assert "validation failed" in ids_to_content["known_bad"].lower()
    assert "not executed" in ids_to_content["unknown_ok"].lower()


# ------------------------------------------------------------------ #
# awrap_model_call — async tests
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_async_valid_tool_call_passes_through() -> None:
    """Test async: valid tool calls pass through without retry."""
    mw = ToolArgValidationMiddleware()

    ai_msg = AIMessage(
        content="",
        tool_calls=[_make_tool_call("search_tool", {"query": "test"})],
    )
    request = _make_request(tools=[search_tool])
    call_count = {"n": 0}

    async def handler(req: ModelRequest[None]) -> ModelResponse[Any]:
        call_count["n"] += 1
        return _make_response(ai_msg)

    result = await mw.awrap_model_call(request, handler)
    assert call_count["n"] == 1
    assert result.result[0] is ai_msg


@pytest.mark.asyncio
async def test_async_invalid_triggers_retry() -> None:
    """Test async: invalid tool calls trigger retry."""
    mw = ToolArgValidationMiddleware(max_retries=2)

    bad_msg = AIMessage(
        content="",
        id="bad",
        tool_calls=[_make_tool_call("search_tool", {})],
    )
    good_msg = AIMessage(
        content="",
        id="good",
        tool_calls=[_make_tool_call("search_tool", {"query": "fixed"})],
    )

    request = _make_request(tools=[search_tool])
    call_count = {"n": 0}

    async def handler(req: ModelRequest[None]) -> ModelResponse[Any]:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _make_response(bad_msg)
        return _make_response(good_msg)

    result = await mw.awrap_model_call(request, handler)
    assert call_count["n"] == 2
    assert result.result[0] is good_msg


@pytest.mark.asyncio
async def test_async_max_retries_exhausted() -> None:
    """Test async: after max retries, last response is passed through."""
    mw = ToolArgValidationMiddleware(max_retries=1)

    bad_msg = AIMessage(
        content="",
        id="bad",
        tool_calls=[_make_tool_call("search_tool", {})],
    )
    request = _make_request(tools=[search_tool])

    async def handler(req: ModelRequest[None]) -> ModelResponse[Any]:
        return _make_response(bad_msg)

    result = await mw.awrap_model_call(request, handler)
    assert result.result[0] is bad_msg


@pytest.mark.asyncio
async def test_async_retry_preserves_original_messages() -> None:
    """Test async: retry request preserves original messages."""
    mw = ToolArgValidationMiddleware(max_retries=1)

    bad_msg = AIMessage(content="", id="bad", tool_calls=[_make_tool_call("search_tool", {})])
    good_msg = AIMessage(
        content="",
        id="good",
        tool_calls=[_make_tool_call("search_tool", {"query": "ok"})],
    )

    original_messages = [HumanMessage("original")]
    request = _make_request(tools=[search_tool], messages=original_messages)
    retry_requests: list[ModelRequest[None]] = []

    async def handler(req: ModelRequest[None]) -> ModelResponse[Any]:
        retry_requests.append(req)
        if len(retry_requests) == 1:
            return _make_response(bad_msg)
        return _make_response(good_msg)

    await mw.awrap_model_call(request, handler)

    assert len(retry_requests) == 2
    # Original message preserved
    assert retry_requests[1].messages[0] is original_messages[0]


# ------------------------------------------------------------------ #
# JSON Schema (dict-based) validation tests
# ------------------------------------------------------------------ #


def test_json_schema_valid(json_schema_tool: Any) -> None:
    """Test dict-schema validation with valid args."""
    pytest.importorskip("jsonschema")

    mw = ToolArgValidationMiddleware()

    ai_msg = AIMessage(
        content="",
        tool_calls=[_make_tool_call("mcp_search", {"query": "test"})],
    )
    request = _make_request(tools=[json_schema_tool])

    def handler(req: ModelRequest[None]) -> ModelResponse[Any]:
        return _make_response(ai_msg)

    result = mw.wrap_model_call(request, handler)
    assert result.result[0] is ai_msg


def test_json_schema_invalid_triggers_retry(json_schema_tool: Any) -> None:
    """Test dict-schema validation triggers retry on invalid args."""
    pytest.importorskip("jsonschema")

    mw = ToolArgValidationMiddleware(max_retries=1)

    bad_msg = AIMessage(
        content="",
        id="bad",
        tool_calls=[_make_tool_call("mcp_search", {"limit": 5})],  # missing 'query'
    )
    good_msg = AIMessage(
        content="",
        id="good",
        tool_calls=[_make_tool_call("mcp_search", {"query": "fixed"})],
    )

    request = _make_request(tools=[json_schema_tool])
    call_count = {"n": 0}

    def handler(req: ModelRequest[None]) -> ModelResponse[Any]:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _make_response(bad_msg)
        return _make_response(good_msg)

    result = mw.wrap_model_call(request, handler)
    assert call_count["n"] == 2
    assert result.result[0] is good_msg


def test_json_schema_wrong_type(json_schema_tool: Any) -> None:
    """Test dict-schema validation catches type errors."""
    pytest.importorskip("jsonschema")

    mw = ToolArgValidationMiddleware(max_retries=1)

    bad_msg = AIMessage(
        content="",
        id="bad",
        tool_calls=[_make_tool_call("mcp_search", {"query": "ok", "limit": "not_int"})],
    )
    good_msg = AIMessage(
        content="",
        id="good",
        tool_calls=[_make_tool_call("mcp_search", {"query": "ok", "limit": 5})],
    )

    request = _make_request(tools=[json_schema_tool])
    call_count = {"n": 0}

    def handler(req: ModelRequest[None]) -> ModelResponse[Any]:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _make_response(bad_msg)
        return _make_response(good_msg)

    mw.wrap_model_call(request, handler)
    assert call_count["n"] == 2


def test_json_schema_custom_validator_class(json_schema_tool: Any) -> None:
    """Test that a custom jsonschema validator class is used when provided."""
    jsonschema = pytest.importorskip("jsonschema")

    mw = ToolArgValidationMiddleware(
        max_retries=1,
        json_schema_validator_class=jsonschema.Draft202012Validator,
    )

    good_msg = AIMessage(
        content="",
        id="ok",
        tool_calls=[_make_tool_call("mcp_search", {"query": "hello", "limit": 5})],
    )
    request = _make_request(tools=[json_schema_tool])

    def handler(req: ModelRequest[None]) -> ModelResponse[Any]:
        return _make_response(good_msg)

    resp = mw.wrap_model_call(request, handler)
    assert resp.result[0] == good_msg


# ------------------------------------------------------------------ #
# Schema resolution tests
# ------------------------------------------------------------------ #


def test_schema_resolved_once() -> None:
    """Test that schemas are only resolved on first call."""
    mw = ToolArgValidationMiddleware()

    ai_msg = AIMessage(content="no tools")
    request = _make_request(tools=[search_tool])

    def handler(req: ModelRequest[None]) -> ModelResponse[Any]:
        return _make_response(ai_msg)

    mw.wrap_model_call(request, handler)
    assert mw._schemas_resolved is True
    assert "search_tool" in mw._schemas

    # Second call should not re-resolve
    original_schemas = mw._schemas
    mw.wrap_model_call(request, handler)
    assert mw._schemas is original_schemas


def test_raw_dict_tools_skipped() -> None:
    """Test that raw dict tool specs are skipped during schema resolution."""
    mw = ToolArgValidationMiddleware()

    ai_msg = AIMessage(content="no tools")
    dict_tool = {"type": "function", "function": {"name": "test"}}
    request = _make_request(tools=[dict_tool, search_tool])

    def handler(req: ModelRequest[None]) -> ModelResponse[Any]:
        return _make_response(ai_msg)

    mw.wrap_model_call(request, handler)
    assert "search_tool" in mw._schemas
    assert "test" not in mw._schemas


def test_no_tools_produces_empty_schemas() -> None:
    """Test that empty tools list results in empty schemas."""
    mw = ToolArgValidationMiddleware()

    ai_msg = AIMessage(content="no tools")
    request = _make_request(tools=[])

    def handler(req: ModelRequest[None]) -> ModelResponse[Any]:
        return _make_response(ai_msg)

    mw.wrap_model_call(request, handler)
    assert mw._schemas == {}


# ------------------------------------------------------------------ #
# End-to-end with create_agent
# ------------------------------------------------------------------ #


def test_end_to_end_valid_tool_call() -> None:
    """Test middleware works end-to-end with create_agent and valid tool calls."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search_tool", args={"query": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search_tool],
        middleware=[ToolArgValidationMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search for test")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "Results for test" in tool_messages[0].content


def test_end_to_end_no_tool_calls() -> None:
    """Test middleware passes through when model makes no tool calls."""
    model = FakeToolCallingModel(tool_calls=[[]])

    agent = create_agent(
        model=model,
        tools=[search_tool],
        middleware=[ToolArgValidationMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Hello")]},
        {"configurable": {"thread_id": "test"}},
    )

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 1
    # No tool calls, so no tool messages
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 0

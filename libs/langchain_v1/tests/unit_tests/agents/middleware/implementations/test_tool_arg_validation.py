"""Tests for ToolArgValidationMiddleware."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from pydantic import BaseModel
from pydantic.v1 import BaseModel as BaseModelV1

from langchain.agents import create_agent
from langchain.agents.middleware import ToolArgValidationMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain.agents.structured_output import ToolStrategy
from tests.unit_tests.agents.model import FakeToolCallingModel

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


def _build_request(
    *,
    tools: list[BaseTool | dict[str, Any]] | None = None,
    messages: list[Any] | None = None,
) -> ModelRequest[None]:
    """Create a model request for middleware unit tests."""
    return ModelRequest(
        model=FakeToolCallingModel(),
        tools=tools or [],
        messages=messages or [HumanMessage("Hello")],
        runtime=Runtime(),
    )


def _build_response(
    tool_calls: list[dict[str, Any]] | None = None,
    *,
    content: str = "assistant",
    extra_messages: list[Any] | None = None,
) -> ModelResponse[Any]:
    """Create a model response with a single AI message by default."""
    result: list[Any] = [AIMessage(content=content, tool_calls=tool_calls or [])]
    if extra_messages:
        result.extend(extra_messages)
    return ModelResponse(result=result)


def _build_sync_handler(
    responses: list[ModelResponse[Any]],
    captured_requests: list[ModelRequest[Any]],
) -> Callable[[ModelRequest[Any]], ModelResponse[Any]]:
    """Build a sequential synchronous handler."""

    def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        captured_requests.append(request)
        return responses[len(captured_requests) - 1]

    return handler


def _build_async_handler(
    responses: list[ModelResponse[Any]],
    captured_requests: list[ModelRequest[Any]],
) -> Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]]:
    """Build a sequential asynchronous handler."""

    async def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        captured_requests.append(request)
        return responses[len(captured_requests) - 1]

    return handler


class SearchArgs(BaseModel):
    """Schema for search tool tests."""

    query: str
    limit: int
    metadata: dict[str, Any] | None = None
    items: list[Any] | None = None


@tool(args_schema=SearchArgs)
def search_tool(
    query: str,
    limit: int,
    metadata: dict[str, Any] | None = None,
    items: list[Any] | None = None,
) -> str:
    """Search for content."""
    return f"{query}:{limit}:{metadata}:{items}"


class LegacySearchArgs(BaseModelV1):
    """Legacy schema for async validation tests."""

    city: str
    count: int


@tool(args_schema=LegacySearchArgs)
async def legacy_search_tool(city: str, count: int) -> str:
    """Search using a legacy pydantic v1 schema."""
    return f"{city}:{count}"


JSON_SCHEMA_TOOL_ARGS = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "limit": {"type": "integer"},
    },
    "required": ["query", "limit"],
    "additionalProperties": False,
}


@tool(args_schema=JSON_SCHEMA_TOOL_ARGS)
def json_schema_tool(query: str, limit: int) -> str:
    """Search with a dict schema."""
    return f"{query}:{limit}"


class CountingArgs(BaseModel):
    """Schema for cache tests."""

    value: str


class CountingTool(BaseTool):
    """Tool whose schema access can be counted."""

    name: str = "counting_tool"
    description: str = "A counting tool."
    args_schema: type[BaseModel] = CountingArgs
    schema_accesses: int = 0

    @property
    def tool_call_schema(self) -> type[BaseModel]:
        """Count how many times the schema is resolved."""
        self.schema_accesses += 1
        return super().tool_call_schema

    def _run(self, value: str) -> str:
        """Run the counting tool."""
        return value


def test_tool_arg_validation_initialization() -> None:
    """Test initialization defaults and validation."""
    middleware = ToolArgValidationMiddleware()

    assert middleware.max_retries == 2
    assert middleware.strip_empty_values is True
    assert middleware.json_schema_validator_class is None
    assert middleware.tools == []

    with pytest.raises(ValueError, match="max_retries must be >= 0"):
        ToolArgValidationMiddleware(max_retries=-1)

    with pytest.raises(TypeError, match="json_schema_validator_class must be a class or None"):
        ToolArgValidationMiddleware(json_schema_validator_class=object())  # type: ignore[arg-type]


def test_wrap_model_call_passes_through_without_tool_calls() -> None:
    """Test sync pass-through when the model does not emit tool calls."""
    middleware = ToolArgValidationMiddleware()
    captured_requests: list[ModelRequest[Any]] = []
    response = _build_response(content="No tools")

    result = middleware.wrap_model_call(
        _build_request(tools=[search_tool]),
        _build_sync_handler([response], captured_requests),
    )

    assert result is response
    assert len(captured_requests) == 1


def test_wrap_model_call_retries_pydantic_v2_and_sanitizes_args() -> None:
    """Test sync retries with Pydantic v2 validation and sanitized args."""
    middleware = ToolArgValidationMiddleware()
    captured_requests: list[ModelRequest[Any]] = []
    responses = [
        _build_response(
            [
                {
                    "name": "search_tool",
                    "id": "call_invalid",
                    "args": {"query": "python", "limit": "oops"},
                }
            ],
            content="invalid",
        ),
        _build_response(
            [
                {
                    "name": "search_tool",
                    "id": "call_valid",
                    "args": {
                        "query": "python",
                        "limit": 3,
                        "metadata": {"keep": "yes", "drop": None, "nested": {}},
                    },
                }
            ],
            content="valid",
        ),
    ]

    result = middleware.wrap_model_call(
        _build_request(tools=[search_tool]),
        _build_sync_handler(responses, captured_requests),
    )

    assert len(captured_requests) == 2
    retry_request = captured_requests[1]
    assert isinstance(retry_request.messages[-2], AIMessage)
    assert isinstance(retry_request.messages[-1], ToolMessage)
    assert "Invalid arguments for tool 'search_tool'" in retry_request.messages[-1].content

    final_message = result.result[0]
    assert isinstance(final_message, AIMessage)
    assert final_message.tool_calls[0]["args"] == {
        "query": "python",
        "limit": 3,
        "metadata": {"keep": "yes"},
    }


@pytest.mark.asyncio
async def test_awrap_model_call_retries_pydantic_v1() -> None:
    """Test async retries with a Pydantic v1 args schema."""
    middleware = ToolArgValidationMiddleware()
    captured_requests: list[ModelRequest[Any]] = []
    responses = [
        _build_response(
            [
                {
                    "name": "legacy_search_tool",
                    "id": "legacy_invalid",
                    "args": {"city": "Tokyo", "count": "nope"},
                }
            ]
        ),
        _build_response(
            [
                {
                    "name": "legacy_search_tool",
                    "id": "legacy_valid",
                    "args": {"city": "Tokyo", "count": 2},
                }
            ]
        ),
    ]

    result = await middleware.awrap_model_call(
        _build_request(tools=[legacy_search_tool]),
        _build_async_handler(responses, captured_requests),
    )

    assert len(captured_requests) == 2
    retry_request = captured_requests[1]
    assert isinstance(retry_request.messages[-1], ToolMessage)
    assert "legacy_search_tool" in retry_request.messages[-1].content

    final_message = result.result[0]
    assert isinstance(final_message, AIMessage)
    assert final_message.tool_calls[0]["args"] == {"city": "Tokyo", "count": 2}


def test_strip_empty_values_recursively_preserves_list_shape() -> None:
    """Test recursive stripping and list-shape preservation."""
    middleware = ToolArgValidationMiddleware()
    response = _build_response(
        [
            {
                "name": "search_tool",
                "id": "call_sanitized",
                "args": {
                    "query": "python",
                    "limit": 5,
                    "metadata": {"drop": None, "nested": {}},
                    "items": [{"keep": 1, "drop": None}, None, {}, []],
                },
            }
        ]
    )

    result = middleware.wrap_model_call(
        _build_request(tools=[search_tool]),
        _build_sync_handler([response], []),
    )

    final_message = result.result[0]
    assert isinstance(final_message, AIMessage)
    assert final_message.tool_calls[0]["args"] == {
        "query": "python",
        "limit": 5,
        "items": [{"keep": 1}, None, {}, []],
    }


def test_retry_exhaustion_returns_last_invalid_response_unchanged() -> None:
    """Test fail-open behavior after validation retries are exhausted."""
    middleware = ToolArgValidationMiddleware(max_retries=1)
    captured_requests: list[ModelRequest[Any]] = []
    responses = [
        _build_response(
            [
                {
                    "name": "search_tool",
                    "id": "call_1",
                    "args": {"query": "python", "limit": "bad"},
                }
            ]
        ),
        _build_response(
            [
                {
                    "name": "search_tool",
                    "id": "call_2",
                    "args": {
                        "query": "python",
                        "limit": "still-bad",
                        "metadata": {"empty": None},
                    },
                }
            ]
        ),
    ]

    result = middleware.wrap_model_call(
        _build_request(tools=[search_tool]),
        _build_sync_handler(responses, captured_requests),
    )

    assert len(captured_requests) == 2
    assert result is responses[-1]
    last_message = result.result[0]
    assert isinstance(last_message, AIMessage)
    assert last_message.tool_calls[0]["args"] == {
        "query": "python",
        "limit": "still-bad",
        "metadata": {"empty": None},
    }


def test_unknown_tool_calls_pass_through_without_validation() -> None:
    """Test that tool calls not present in the request are ignored."""
    middleware = ToolArgValidationMiddleware()
    captured_requests: list[ModelRequest[Any]] = []
    response = _build_response(
        [{"name": "unknown_tool", "id": "call_unknown", "args": {"bad": "value"}}]
    )

    result = middleware.wrap_model_call(
        _build_request(tools=[search_tool]),
        _build_sync_handler([response], captured_requests),
    )

    assert result is not response
    final_message = result.result[0]
    assert isinstance(final_message, AIMessage)
    assert final_message.tool_calls[0]["name"] == "unknown_tool"
    assert len(captured_requests) == 1


def test_mixed_valid_invalid_batch_retries_all_tool_calls() -> None:
    """Test retry-all behavior when a batch contains valid and invalid tool calls."""

    @tool
    def calculator_tool(expression: str) -> str:
        """Evaluate an expression."""
        return expression

    middleware = ToolArgValidationMiddleware()
    captured_requests: list[ModelRequest[Any]] = []
    responses = [
        _build_response(
            [
                {
                    "name": "search_tool",
                    "id": "invalid_search",
                    "args": {"query": "python", "limit": "bad"},
                },
                {
                    "name": "calculator_tool",
                    "id": "valid_calc",
                    "args": {"expression": "1 + 1"},
                },
            ]
        ),
        _build_response(
            [
                {
                    "name": "search_tool",
                    "id": "fixed_search",
                    "args": {"query": "python", "limit": 2},
                },
                {
                    "name": "calculator_tool",
                    "id": "fixed_calc",
                    "args": {"expression": "1 + 1"},
                },
            ]
        ),
    ]

    result = middleware.wrap_model_call(
        _build_request(tools=[search_tool, calculator_tool]),
        _build_sync_handler(responses, captured_requests),
    )

    assert len(captured_requests) == 2
    retry_messages = captured_requests[1].messages[-3:]
    assert isinstance(retry_messages[0], AIMessage)
    assert isinstance(retry_messages[1], ToolMessage)
    assert isinstance(retry_messages[2], ToolMessage)
    assert "Invalid arguments for tool 'search_tool'" in retry_messages[1].content
    assert "batch rejected" in retry_messages[2].content

    final_message = result.result[0]
    assert isinstance(final_message, AIMessage)
    assert [tool_call["id"] for tool_call in final_message.tool_calls] == [
        "fixed_search",
        "fixed_calc",
    ]


@pytest.mark.requires("jsonschema")
def test_dict_schema_validation_retries_with_jsonschema() -> None:
    """Test dict-schema validation and retry behavior."""
    middleware = ToolArgValidationMiddleware()
    captured_requests: list[ModelRequest[Any]] = []
    responses = [
        _build_response(
            [
                {
                    "name": "json_schema_tool",
                    "id": "json_invalid",
                    "args": {"query": "python", "limit": 2, "extra": "boom"},
                }
            ]
        ),
        _build_response(
            [
                {
                    "name": "json_schema_tool",
                    "id": "json_valid",
                    "args": {"query": "python", "limit": 2},
                }
            ]
        ),
    ]

    result = middleware.wrap_model_call(
        _build_request(tools=[json_schema_tool]),
        _build_sync_handler(responses, captured_requests),
    )

    assert len(captured_requests) == 2
    retry_message = captured_requests[1].messages[-1]
    assert isinstance(retry_message, ToolMessage)
    assert "extra" in retry_message.content

    final_message = result.result[0]
    assert isinstance(final_message, AIMessage)
    assert final_message.tool_calls[0]["args"] == {"query": "python", "limit": 2}


def test_missing_jsonschema_raises_clear_error() -> None:
    """Test the ImportError path for dict-schema tools."""
    middleware = ToolArgValidationMiddleware()
    request = _build_request(tools=[json_schema_tool])
    response = _build_response(
        [{"name": "json_schema_tool", "id": "json_call", "args": {"query": "python"}}]
    )

    with (
        patch(
            "langchain.agents.middleware.tool_arg_validation.importlib.import_module",
            side_effect=ImportError("missing jsonschema"),
        ),
        pytest.raises(ImportError, match="requires the jsonschema package"),
    ):
        middleware.wrap_model_call(request, _build_sync_handler([response], []))


def test_validator_cache_reuses_tools_and_refreshes_when_tool_object_changes() -> None:
    """Test validator caching across requests."""
    middleware = ToolArgValidationMiddleware()
    first_tool = CountingTool()
    second_tool = CountingTool()

    first_request = _build_request(tools=[first_tool])
    second_request = _build_request(tools=[second_tool])

    middleware.wrap_model_call(
        first_request,
        _build_sync_handler(
            [
                _build_response(
                    [{"name": "counting_tool", "id": "call_1", "args": {"value": "first"}}]
                )
            ],
            [],
        ),
    )
    middleware.wrap_model_call(
        first_request,
        _build_sync_handler(
            [
                _build_response(
                    [{"name": "counting_tool", "id": "call_2", "args": {"value": "again"}}]
                )
            ],
            [],
        ),
    )
    middleware.wrap_model_call(
        second_request,
        _build_sync_handler(
            [
                _build_response(
                    [{"name": "counting_tool", "id": "call_3", "args": {"value": "fresh"}}]
                )
            ],
            [],
        ),
    )

    assert first_tool.schema_accesses == 1
    assert second_tool.schema_accesses == 1
    assert middleware._validator_cache["counting_tool"].tool is second_tool


def test_create_agent_retries_inside_model_node_sync() -> None:
    """Test end-to-end sync retry behavior with create_agent."""

    @tool
    def uppercase_tool(value: str) -> str:
        """Uppercase a string."""
        return value.upper()

    model = FakeToolCallingModel(
        tool_calls=[
            [{"name": "uppercase_tool", "id": "call_invalid", "args": {"wrong": "x"}}],
            [{"name": "uppercase_tool", "id": "call_valid", "args": {"value": "hello"}}],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[uppercase_tool],
        middleware=[ToolArgValidationMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Please uppercase hello")]},
        {"configurable": {"thread_id": "tool-arg-validation-sync"}},
    )

    assert model.index == 3
    assert len(result["messages"]) == 4
    assert isinstance(result["messages"][1], AIMessage)
    assert result["messages"][1].tool_calls[0]["id"] == "call_valid"

    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "HELLO"
    assert tool_messages[0].status == "success"


@pytest.mark.asyncio
async def test_create_agent_retries_inside_model_node_async() -> None:
    """Test end-to-end async retry behavior with create_agent."""

    @tool
    async def async_uppercase_tool(value: str) -> str:
        """Uppercase a string asynchronously."""
        return value.upper()

    model = FakeToolCallingModel(
        tool_calls=[
            [{"name": "async_uppercase_tool", "id": "call_invalid", "args": {"oops": "x"}}],
            [
                {
                    "name": "async_uppercase_tool",
                    "id": "call_valid",
                    "args": {"value": "hello"},
                }
            ],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[async_uppercase_tool],
        middleware=[ToolArgValidationMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage("Please uppercase hello")]},
        {"configurable": {"thread_id": "tool-arg-validation-async"}},
    )

    assert model.index == 3
    assert len(result["messages"]) == 4
    assert isinstance(result["messages"][1], AIMessage)
    assert result["messages"][1].tool_calls[0]["id"] == "call_valid"

    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "HELLO"
    assert tool_messages[0].status == "success"


def test_structured_output_flow_is_unchanged() -> None:
    """Test that structured output retries are not intercepted."""

    class WeatherReport(BaseModel):
        """Structured weather response."""

        temperature: float
        conditions: str

    model = FakeToolCallingModel(
        tool_calls=[
            [{"name": "WeatherReport", "id": "sr_invalid", "args": {"temperature": "bad"}}],
            [
                {
                    "name": "WeatherReport",
                    "id": "sr_valid",
                    "args": {"temperature": 72.0, "conditions": "sunny"},
                }
            ],
        ]
    )

    agent = create_agent(
        model=model,
        middleware=[ToolArgValidationMiddleware()],
        response_format=ToolStrategy(schema=WeatherReport),
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("What's the weather?")]},
        {"configurable": {"thread_id": "tool-arg-structured-output"}},
    )

    assert model.index == 2
    assert isinstance(result["structured_response"], WeatherReport)
    assert result["structured_response"].temperature == 72.0
    assert result["structured_response"].conditions == "sunny"

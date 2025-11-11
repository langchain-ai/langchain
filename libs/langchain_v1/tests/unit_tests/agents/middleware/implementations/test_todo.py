"""Unit tests for TodoListMiddleware."""

from __future__ import annotations

from typing import cast

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from langchain.agents.factory import create_agent
from langchain.agents.middleware.todo import (
    PlanningState,
    TodoListMiddleware,
    WRITE_TODOS_SYSTEM_PROMPT,
    WRITE_TODOS_TOOL_DESCRIPTION,
    write_todos,
)
from langchain.agents.middleware.types import ModelRequest, ModelResponse

from ...model import FakeToolCallingModel


def _fake_runtime() -> Runtime:
    return cast(Runtime, object())


def _make_request(system_prompt: str | None = None) -> ModelRequest:
    """Create a minimal ModelRequest for testing."""
    model = GenericFakeChatModel(messages=iter([AIMessage(content="response")]))
    return ModelRequest(
        model=model,
        system_prompt=system_prompt,
        messages=[],
        tool_choice=None,
        tools=[],
        response_format=None,
        state=cast("AgentState", {}),  # type: ignore[name-defined]
        runtime=_fake_runtime(),
        model_settings={},
    )


# ==============================================================================
# Synchronous Tests
# ==============================================================================


def test_todo_middleware_initialization() -> None:
    """Test that TodoListMiddleware initializes correctly."""
    middleware = TodoListMiddleware()
    assert middleware.state_schema == PlanningState
    assert len(middleware.tools) == 1
    assert middleware.tools[0].name == "write_todos"


def test_has_write_todos_tool() -> None:
    """Test that middleware registers the write_todos tool."""
    middleware = TodoListMiddleware()

    # Should have one tool registered
    assert len(middleware.tools) == 1
    assert middleware.tools[0].name == "write_todos"


def test_todo_middleware_default_prompts() -> None:
    """Test that TodoListMiddleware uses default prompts when none provided."""
    middleware = TodoListMiddleware()

    # Verify default system prompt
    assert middleware.system_prompt == WRITE_TODOS_SYSTEM_PROMPT

    # Verify default tool description
    assert middleware.tool_description == WRITE_TODOS_TOOL_DESCRIPTION
    assert len(middleware.tools) == 1
    tool = middleware.tools[0]
    assert tool.description == WRITE_TODOS_TOOL_DESCRIPTION


def test_adds_system_prompt_when_none_exists() -> None:
    """Test that middleware adds system prompt when request has none."""
    middleware = TodoListMiddleware()
    request = _make_request(system_prompt=None)

    def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="response")])

    middleware.wrap_model_call(request, mock_handler)

    # System prompt should be set
    assert request.system_prompt is not None
    assert "write_todos" in request.system_prompt


def test_appends_to_existing_system_prompt() -> None:
    """Test that middleware appends to existing system prompt."""
    existing_prompt = "You are a helpful assistant."
    middleware = TodoListMiddleware()
    request = _make_request(system_prompt=existing_prompt)

    def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="response")])

    middleware.wrap_model_call(request, mock_handler)

    # System prompt should contain both
    assert request.system_prompt is not None
    assert existing_prompt in request.system_prompt
    assert "write_todos" in request.system_prompt
    assert request.system_prompt.startswith(existing_prompt)


@pytest.mark.parametrize(
    "original_prompt,expected_prompt_prefix",
    [
        ("Original prompt", "Original prompt\n\n## `write_todos`"),
        (None, "## `write_todos`"),
    ],
)
def test_todo_middleware_on_model_call(original_prompt, expected_prompt_prefix) -> None:
    """Test that wrap_model_call handles system prompts correctly."""
    middleware = TodoListMiddleware()
    model = FakeToolCallingModel()

    state: PlanningState = {"messages": [HumanMessage(content="Hello")]}

    request = ModelRequest(
        model=model,
        system_prompt=original_prompt,
        messages=[HumanMessage(content="Hello")],
        tool_choice=None,
        tools=[],
        response_format=None,
        state=state,
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    def mock_handler(req: ModelRequest) -> AIMessage:
        return AIMessage(content="mock response")

    # Call wrap_model_call to trigger the middleware logic
    middleware.wrap_model_call(request, mock_handler)
    # Check that the request was modified in place
    assert request.system_prompt.startswith(expected_prompt_prefix)


def test_custom_system_prompt() -> None:
    """Test that middleware uses custom system prompt."""
    custom_prompt = "Custom planning instructions"
    middleware = TodoListMiddleware(system_prompt=custom_prompt)
    request = _make_request(system_prompt=None)

    def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="response")])

    middleware.wrap_model_call(request, mock_handler)

    # Should use custom prompt
    assert request.system_prompt == custom_prompt


def test_todo_middleware_custom_system_prompt() -> None:
    """Test that TodoListMiddleware can be initialized with custom system prompt."""
    custom_system_prompt = "Custom todo system prompt for testing"
    middleware = TodoListMiddleware(system_prompt=custom_system_prompt)
    model = FakeToolCallingModel()

    state: PlanningState = {"messages": [HumanMessage(content="Hello")]}

    request = ModelRequest(
        model=model,
        system_prompt="Original prompt",
        messages=[HumanMessage(content="Hello")],
        tool_choice=None,
        tools=[],
        response_format=None,
        model_settings={},
        state=state,
        runtime=cast(Runtime, object()),
    )

    def mock_handler(req: ModelRequest) -> AIMessage:
        return AIMessage(content="mock response")

    # Call wrap_model_call to trigger the middleware logic
    middleware.wrap_model_call(request, mock_handler)
    # Check that the request was modified in place
    assert request.system_prompt == f"Original prompt\n\n{custom_system_prompt}"


def test_custom_tool_description() -> None:
    """Test that middleware uses custom tool description."""
    custom_description = "Custom todo tool description"
    middleware = TodoListMiddleware(tool_description=custom_description)

    # Tool should use custom description
    assert len(middleware.tools) == 1
    assert middleware.tools[0].description == custom_description


def test_todo_middleware_custom_tool_description() -> None:
    """Test that TodoListMiddleware can be initialized with custom tool description."""
    custom_tool_description = "Custom tool description for testing"
    middleware = TodoListMiddleware(tool_description=custom_tool_description)

    assert len(middleware.tools) == 1
    tool = middleware.tools[0]
    assert tool.description == custom_tool_description


def test_todo_middleware_custom_system_prompt_and_tool_description() -> None:
    """Test that TodoListMiddleware can be initialized with both custom prompts."""
    custom_system_prompt = "Custom system prompt"
    custom_tool_description = "Custom tool description"
    middleware = TodoListMiddleware(
        system_prompt=custom_system_prompt,
        tool_description=custom_tool_description,
    )

    # Verify system prompt
    model = FakeToolCallingModel()
    state: PlanningState = {"messages": [HumanMessage(content="Hello")]}

    request = ModelRequest(
        model=model,
        system_prompt=None,
        messages=[HumanMessage(content="Hello")],
        tool_choice=None,
        tools=[],
        response_format=None,
        state=state,
        runtime=cast(Runtime, object()),
        model_settings={},
    )

    def mock_handler(req: ModelRequest) -> AIMessage:
        return AIMessage(content="mock response")

    # Call wrap_model_call to trigger the middleware logic
    middleware.wrap_model_call(request, mock_handler)
    # Check that the request was modified in place
    assert request.system_prompt == custom_system_prompt

    # Verify tool description
    assert len(middleware.tools) == 1
    tool = middleware.tools[0]
    assert tool.description == custom_tool_description


@pytest.mark.parametrize(
    "todos,expected_message",
    [
        ([], "Updated todo list to []"),
        (
            [{"content": "Task 1", "status": "pending"}],
            "Updated todo list to [{'content': 'Task 1', 'status': 'pending'}]",
        ),
        (
            [
                {"content": "Task 1", "status": "pending"},
                {"content": "Task 2", "status": "in_progress"},
            ],
            "Updated todo list to [{'content': 'Task 1', 'status': 'pending'}, {'content': 'Task 2', 'status': 'in_progress'}]",
        ),
        (
            [
                {"content": "Task 1", "status": "pending"},
                {"content": "Task 2", "status": "in_progress"},
                {"content": "Task 3", "status": "completed"},
            ],
            "Updated todo list to [{'content': 'Task 1', 'status': 'pending'}, {'content': 'Task 2', 'status': 'in_progress'}, {'content': 'Task 3', 'status': 'completed'}]",
        ),
    ],
)
def test_todo_middleware_write_todos_tool_execution(todos, expected_message) -> None:
    """Test that the write_todos tool executes correctly."""
    tool_call = {
        "args": {"todos": todos},
        "name": "write_todos",
        "type": "tool_call",
        "id": "test_call",
    }
    result = write_todos.invoke(tool_call)
    assert result.update["todos"] == todos
    assert result.update["messages"][0].content == expected_message


@pytest.mark.parametrize(
    "invalid_todos",
    [
        [{"content": "Task 1", "status": "invalid_status"}],
        [{"status": "pending"}],
    ],
)
def test_todo_middleware_write_todos_tool_validation_errors(invalid_todos) -> None:
    """Test that the write_todos tool rejects invalid input."""
    tool_call = {
        "args": {"todos": invalid_todos},
        "name": "write_todos",
        "type": "tool_call",
        "id": "test_call",
    }
    with pytest.raises(Exception):
        write_todos.invoke(tool_call)


def test_todo_middleware_agent_creation_with_middleware() -> None:
    """Test that an agent can be created with the planning middleware."""
    model = FakeToolCallingModel(
        tool_calls=[
            [
                {
                    "args": {"todos": [{"content": "Task 1", "status": "pending"}]},
                    "name": "write_todos",
                    "type": "tool_call",
                    "id": "test_call",
                }
            ],
            [
                {
                    "args": {"todos": [{"content": "Task 1", "status": "in_progress"}]},
                    "name": "write_todos",
                    "type": "tool_call",
                    "id": "test_call",
                }
            ],
            [
                {
                    "args": {"todos": [{"content": "Task 1", "status": "completed"}]},
                    "name": "write_todos",
                    "type": "tool_call",
                    "id": "test_call",
                }
            ],
            [],
        ]
    )
    middleware = TodoListMiddleware()
    agent = create_agent(model=model, middleware=[middleware])

    result = agent.invoke({"messages": [HumanMessage("Hello")]})
    assert result["todos"] == [{"content": "Task 1", "status": "completed"}]

    # human message (1)
    # ai message (2) - initial todo
    # tool message (3)
    # ai message (4) - updated todo
    # tool message (5)
    # ai message (6) - complete todo
    # tool message (7)
    # ai message (8) - no tool calls
    assert len(result["messages"]) == 8


def test_todo_middleware_custom_system_prompt_in_agent() -> None:
    """Test that custom tool executes correctly in an agent."""
    middleware = TodoListMiddleware(system_prompt="call the write_todos tool")

    model = FakeToolCallingModel(
        tool_calls=[
            [
                {
                    "args": {"todos": [{"content": "Custom task", "status": "pending"}]},
                    "name": "write_todos",
                    "type": "tool_call",
                    "id": "test_call",
                }
            ],
            [],
        ]
    )

    agent = create_agent(model=model, middleware=[middleware])

    result = agent.invoke({"messages": [HumanMessage("Hello")]})
    assert result["todos"] == [{"content": "Custom task", "status": "pending"}]
    # assert custom system prompt is in the first AI message
    assert "call the write_todos tool" in result["messages"][1].content


# ==============================================================================
# Async Tests
# ==============================================================================


async def test_adds_system_prompt_when_none_exists_async() -> None:
    """Test async version - middleware adds system prompt when request has none."""
    middleware = TodoListMiddleware()
    request = _make_request(system_prompt=None)

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="response")])

    await middleware.awrap_model_call(request, mock_handler)

    # System prompt should be set
    assert request.system_prompt is not None
    assert "write_todos" in request.system_prompt


async def test_appends_to_existing_system_prompt_async() -> None:
    """Test async version - middleware appends to existing system prompt."""
    existing_prompt = "You are a helpful assistant."
    middleware = TodoListMiddleware()
    request = _make_request(system_prompt=existing_prompt)

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="response")])

    await middleware.awrap_model_call(request, mock_handler)

    # System prompt should contain both
    assert request.system_prompt is not None
    assert existing_prompt in request.system_prompt
    assert "write_todos" in request.system_prompt
    assert request.system_prompt.startswith(existing_prompt)


async def test_custom_system_prompt_async() -> None:
    """Test async version - middleware uses custom system prompt."""
    custom_prompt = "Custom planning instructions"
    middleware = TodoListMiddleware(system_prompt=custom_prompt)
    request = _make_request(system_prompt=None)

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="response")])

    await middleware.awrap_model_call(request, mock_handler)

    # Should use custom prompt
    assert request.system_prompt == custom_prompt


async def test_handler_called_with_modified_request_async() -> None:
    """Test async version - handler receives the modified request."""
    middleware = TodoListMiddleware()
    request = _make_request(system_prompt="Original")
    handler_called = {"value": False}
    received_prompt = {"value": None}

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        handler_called["value"] = True
        received_prompt["value"] = req.system_prompt
        return ModelResponse(result=[AIMessage(content="response")])

    await middleware.awrap_model_call(request, mock_handler)

    assert handler_called["value"]
    assert received_prompt["value"] is not None
    assert "Original" in received_prompt["value"]
    assert "write_todos" in received_prompt["value"]

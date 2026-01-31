"""Unit tests for TodoListMiddleware."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from langchain.agents.factory import create_agent
from langchain.agents.middleware.todo import (
    WRITE_TODOS_SYSTEM_PROMPT,
    WRITE_TODOS_TOOL_DESCRIPTION,
    PlanningState,
    TodoListMiddleware,
    write_todos,
)
from langchain.agents.middleware.types import AgentState, ModelRequest, ModelResponse
from tests.unit_tests.agents.model import FakeToolCallingModel

if TYPE_CHECKING:
    from langgraph.runtime import Runtime


def _fake_runtime() -> Runtime:
    return cast("Runtime", object())


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
        state=AgentState(messages=[]),
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

    captured_request = None

    def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal captured_request
        captured_request = req
        return ModelResponse(result=[AIMessage(content="response")])

    middleware.wrap_model_call(request, mock_handler)

    # System prompt should be set in the modified request passed to handler
    assert captured_request is not None
    assert captured_request.system_prompt is not None
    assert "write_todos" in captured_request.system_prompt
    # Original request should be unchanged
    assert request.system_prompt is None


def test_appends_to_existing_system_prompt() -> None:
    """Test that middleware appends to existing system prompt."""
    existing_prompt = "You are a helpful assistant."
    middleware = TodoListMiddleware()
    request = _make_request(system_prompt=existing_prompt)

    captured_request = None

    def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal captured_request
        captured_request = req
        return ModelResponse(result=[AIMessage(content="response")])

    middleware.wrap_model_call(request, mock_handler)

    # System prompt should contain both in the modified request passed to handler
    assert captured_request is not None
    assert captured_request.system_prompt is not None
    assert existing_prompt in captured_request.system_prompt
    assert "write_todos" in captured_request.system_prompt
    assert captured_request.system_prompt.startswith(existing_prompt)
    # Original request should be unchanged
    assert request.system_prompt == existing_prompt


@pytest.mark.parametrize(
    ("original_prompt", "expected_prompt_prefix"),
    [
        ("Original prompt", "Original prompt\n\n## `write_todos`"),
        (None, "## `write_todos`"),
    ],
)
def test_todo_middleware_on_model_call(
    original_prompt: str | None, expected_prompt_prefix: str
) -> None:
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
        runtime=cast("Runtime", object()),
        model_settings={},
    )

    captured_request = None

    def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal captured_request
        captured_request = req
        return ModelResponse(result=[AIMessage(content="mock response")])

    # Call wrap_model_call to trigger the middleware logic
    middleware.wrap_model_call(request, mock_handler)
    # Check that the modified request passed to handler has the expected prompt
    assert captured_request is not None
    assert captured_request.system_prompt is not None
    assert captured_request.system_prompt.startswith(expected_prompt_prefix)
    # Original request should be unchanged
    assert request.system_prompt == original_prompt


def test_custom_system_prompt() -> None:
    """Test that middleware uses custom system prompt."""
    custom_prompt = "Custom planning instructions"
    middleware = TodoListMiddleware(system_prompt=custom_prompt)
    request = _make_request(system_prompt=None)

    captured_request = None

    def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal captured_request
        captured_request = req
        return ModelResponse(result=[AIMessage(content="response")])

    middleware.wrap_model_call(request, mock_handler)

    # Should use custom prompt in the modified request passed to handler
    assert captured_request is not None
    assert captured_request.system_prompt == custom_prompt
    # Original request should be unchanged
    assert request.system_prompt is None


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
        runtime=cast("Runtime", object()),
    )

    captured_request = None

    def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal captured_request
        captured_request = req
        return ModelResponse(result=[AIMessage(content="mock response")])

    # Call wrap_model_call to trigger the middleware logic
    middleware.wrap_model_call(request, mock_handler)
    # Check that the modified request passed to handler has the expected prompt
    assert captured_request is not None
    assert captured_request.system_prompt == f"Original prompt\n\n{custom_system_prompt}"
    # Original request should be unchanged
    assert request.system_prompt == "Original prompt"


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
        runtime=cast("Runtime", object()),
        model_settings={},
    )

    captured_request = None

    def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal captured_request
        captured_request = req
        return ModelResponse(result=[AIMessage(content="mock response")])

    # Call wrap_model_call to trigger the middleware logic
    middleware.wrap_model_call(request, mock_handler)
    # Check that the modified request passed to handler has the expected prompt
    assert captured_request is not None
    assert captured_request.system_prompt == custom_system_prompt
    # Original request should be unchanged
    assert request.system_prompt is None

    # Verify tool description
    assert len(middleware.tools) == 1
    tool = middleware.tools[0]
    assert tool.description == custom_tool_description


@pytest.mark.parametrize(
    ("todos", "expected_message"),
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
            (
                "Updated todo list to ["
                "{'content': 'Task 1', 'status': 'pending'}, "
                "{'content': 'Task 2', 'status': 'in_progress'}]"
            ),
        ),
        (
            [
                {"content": "Task 1", "status": "pending"},
                {"content": "Task 2", "status": "in_progress"},
                {"content": "Task 3", "status": "completed"},
            ],
            (
                "Updated todo list to ["
                "{'content': 'Task 1', 'status': 'pending'}, "
                "{'content': 'Task 2', 'status': 'in_progress'}, "
                "{'content': 'Task 3', 'status': 'completed'}]"
            ),
        ),
    ],
)
def test_todo_middleware_write_todos_tool_execution(
    todos: list[dict[str, Any]], expected_message: str
) -> None:
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
def test_todo_middleware_write_todos_tool_validation_errors(
    invalid_todos: list[dict[str, Any]],
) -> None:
    """Test that the write_todos tool rejects invalid input."""
    tool_call = {
        "args": {"todos": invalid_todos},
        "name": "write_todos",
        "type": "tool_call",
        "id": "test_call",
    }
    with pytest.raises(ValueError, match="1 validation error for write_todos"):
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

    captured_request = None

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal captured_request
        captured_request = req
        return ModelResponse(result=[AIMessage(content="response")])

    await middleware.awrap_model_call(request, mock_handler)

    # System prompt should be set in the modified request passed to handler
    assert captured_request is not None
    assert captured_request.system_prompt is not None
    assert "write_todos" in captured_request.system_prompt
    # Original request should be unchanged
    assert request.system_prompt is None


async def test_appends_to_existing_system_prompt_async() -> None:
    """Test async version - middleware appends to existing system prompt."""
    existing_prompt = "You are a helpful assistant."
    middleware = TodoListMiddleware()
    request = _make_request(system_prompt=existing_prompt)

    captured_request = None

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal captured_request
        captured_request = req
        return ModelResponse(result=[AIMessage(content="response")])

    await middleware.awrap_model_call(request, mock_handler)

    # System prompt should contain both in the modified request passed to handler
    assert captured_request is not None
    assert captured_request.system_prompt is not None
    assert existing_prompt in captured_request.system_prompt
    assert "write_todos" in captured_request.system_prompt
    assert captured_request.system_prompt.startswith(existing_prompt)
    # Original request should be unchanged
    assert request.system_prompt == existing_prompt


async def test_custom_system_prompt_async() -> None:
    """Test async version - middleware uses custom system prompt."""
    custom_prompt = "Custom planning instructions"
    middleware = TodoListMiddleware(system_prompt=custom_prompt)
    request = _make_request(system_prompt=None)

    captured_request = None

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal captured_request
        captured_request = req
        return ModelResponse(result=[AIMessage(content="response")])

    await middleware.awrap_model_call(request, mock_handler)

    # Should use custom prompt in the modified request passed to handler
    assert captured_request is not None
    assert captured_request.system_prompt == custom_prompt


def test_parallel_write_todos_calls_rejected() -> None:
    """Test that parallel write_todos calls are rejected with error messages."""
    middleware = TodoListMiddleware()

    # Create an AI message with two write_todos tool calls
    ai_message = AIMessage(
        content="I'll update the todos",
        tool_calls=[
            {
                "name": "write_todos",
                "args": {"todos": [{"content": "Task 1", "status": "pending"}]},
                "id": "call_1",
                "type": "tool_call",
            },
            {
                "name": "write_todos",
                "args": {"todos": [{"content": "Task 2", "status": "pending"}]},
                "id": "call_2",
                "type": "tool_call",
            },
        ],
    )

    state: PlanningState = {"messages": [HumanMessage(content="Hello"), ai_message]}

    # Call after_model hook
    result = middleware.after_model(state, _fake_runtime())

    # Should return error messages
    assert result == {
        "messages": [
            ToolMessage(
                content=(
                    "Error: The `write_todos` tool should never be called multiple times "
                    "in parallel. Please call it only once per model invocation to update "
                    "the todo list."
                ),
                tool_call_id="call_1",
                status="error",
            ),
            ToolMessage(
                content=(
                    "Error: The `write_todos` tool should never be called multiple times "
                    "in parallel. Please call it only once per model invocation to update "
                    "the todo list."
                ),
                tool_call_id="call_2",
                status="error",
            ),
        ]
    }


def test_parallel_write_todos_with_other_tools() -> None:
    """Test that parallel write_todos calls are rejected but other tool calls remain."""
    middleware = TodoListMiddleware()

    # Create an AI message with two write_todos calls and one other tool call
    ai_message = AIMessage(
        content="I'll do multiple things",
        tool_calls=[
            {
                "name": "some_other_tool",
                "args": {"param": "value"},
                "id": "call_other",
                "type": "tool_call",
            },
            {
                "name": "write_todos",
                "args": {"todos": [{"content": "Task 1", "status": "pending"}]},
                "id": "call_1",
                "type": "tool_call",
            },
            {
                "name": "write_todos",
                "args": {"todos": [{"content": "Task 2", "status": "pending"}]},
                "id": "call_2",
                "type": "tool_call",
            },
        ],
    )

    state: PlanningState = {"messages": [HumanMessage(content="Hello"), ai_message]}

    # Call after_model hook
    result = middleware.after_model(state, _fake_runtime())

    # Should return error messages for write_todos calls only
    assert result == {
        "messages": [
            ToolMessage(
                content=(
                    "Error: The `write_todos` tool should never be called multiple times "
                    "in parallel. Please call it only once per model invocation to update "
                    "the todo list."
                ),
                tool_call_id="call_1",
                status="error",
            ),
            ToolMessage(
                content=(
                    "Error: The `write_todos` tool should never be called multiple times "
                    "in parallel. Please call it only once per model invocation to update "
                    "the todo list."
                ),
                tool_call_id="call_2",
                status="error",
            ),
        ]
    }


def test_single_write_todos_call_allowed() -> None:
    """Test that a single write_todos call is allowed."""
    middleware = TodoListMiddleware()

    # Create an AI message with one write_todos tool call
    ai_message = AIMessage(
        content="I'll update the todos",
        tool_calls=[
            {
                "name": "write_todos",
                "args": {"todos": [{"content": "Task 1", "status": "pending"}]},
                "id": "call_1",
                "type": "tool_call",
            },
        ],
    )

    state: PlanningState = {"messages": [HumanMessage(content="Hello"), ai_message]}

    # Call after_model hook
    result = middleware.after_model(state, _fake_runtime())

    # Should return None (no intervention needed)
    assert result is None


async def test_parallel_write_todos_calls_rejected_async() -> None:
    """Test async version - parallel write_todos calls are rejected with error messages."""
    middleware = TodoListMiddleware()

    # Create an AI message with two write_todos tool calls
    ai_message = AIMessage(
        content="I'll update the todos",
        tool_calls=[
            {
                "name": "write_todos",
                "args": {"todos": [{"content": "Task 1", "status": "pending"}]},
                "id": "call_1",
                "type": "tool_call",
            },
            {
                "name": "write_todos",
                "args": {"todos": [{"content": "Task 2", "status": "pending"}]},
                "id": "call_2",
                "type": "tool_call",
            },
        ],
    )

    state: PlanningState = {"messages": [HumanMessage(content="Hello"), ai_message]}

    # Call aafter_model hook
    result = await middleware.aafter_model(state, _fake_runtime())

    # Should return error messages
    assert result == {
        "messages": [
            ToolMessage(
                content=(
                    "Error: The `write_todos` tool should never be called multiple times "
                    "in parallel. Please call it only once per model invocation to update "
                    "the todo list."
                ),
                tool_call_id="call_1",
                status="error",
            ),
            ToolMessage(
                content=(
                    "Error: The `write_todos` tool should never be called multiple times "
                    "in parallel. Please call it only once per model invocation to update "
                    "the todo list."
                ),
                tool_call_id="call_2",
                status="error",
            ),
        ]
    }


async def test_parallel_write_todos_with_other_tools_async() -> None:
    """Test async version - parallel write_todos calls are rejected but other tool calls remain."""
    middleware = TodoListMiddleware()

    # Create an AI message with two write_todos calls and one other tool call
    ai_message = AIMessage(
        content="I'll do multiple things",
        tool_calls=[
            {
                "name": "some_other_tool",
                "args": {"param": "value"},
                "id": "call_other",
                "type": "tool_call",
            },
            {
                "name": "write_todos",
                "args": {"todos": [{"content": "Task 1", "status": "pending"}]},
                "id": "call_1",
                "type": "tool_call",
            },
            {
                "name": "write_todos",
                "args": {"todos": [{"content": "Task 2", "status": "pending"}]},
                "id": "call_2",
                "type": "tool_call",
            },
        ],
    )

    state: PlanningState = {"messages": [HumanMessage(content="Hello"), ai_message]}

    # Call aafter_model hook
    result = await middleware.aafter_model(state, _fake_runtime())

    # Should return error messages for write_todos calls only
    assert result == {
        "messages": [
            ToolMessage(
                content=(
                    "Error: The `write_todos` tool should never be called multiple times "
                    "in parallel. Please call it only once per model invocation to update "
                    "the todo list."
                ),
                tool_call_id="call_1",
                status="error",
            ),
            ToolMessage(
                content=(
                    "Error: The `write_todos` tool should never be called multiple times "
                    "in parallel. Please call it only once per model invocation to update "
                    "the todo list."
                ),
                tool_call_id="call_2",
                status="error",
            ),
        ]
    }


async def test_single_write_todos_call_allowed_async() -> None:
    """Test async version - a single write_todos call is allowed."""
    middleware = TodoListMiddleware()

    # Create an AI message with one write_todos tool call
    ai_message = AIMessage(
        content="I'll update the todos",
        tool_calls=[
            {
                "name": "write_todos",
                "args": {"todos": [{"content": "Task 1", "status": "pending"}]},
                "id": "call_1",
                "type": "tool_call",
            },
        ],
    )

    state: PlanningState = {"messages": [HumanMessage(content="Hello"), ai_message]}

    # Call aafter_model hook
    result = await middleware.aafter_model(state, _fake_runtime())

    # Should return None (no intervention needed)
    assert result is None


async def test_handler_called_with_modified_request_async() -> None:
    """Test async version - handler receives the modified request."""
    middleware = TodoListMiddleware()
    request = _make_request(system_prompt="Original")
    handler_called: dict[str, bool] = {"value": False}
    received_prompt: dict[str, str | None] = {"value": None}

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        handler_called["value"] = True
        received_prompt["value"] = req.system_prompt
        return ModelResponse(result=[AIMessage(content="response")])

    await middleware.awrap_model_call(request, mock_handler)

    assert handler_called["value"]
    assert received_prompt["value"] is not None
    assert "Original" in received_prompt["value"]
    assert "write_todos" in received_prompt["value"]


# ==============================================================================
# keep_only_last_todo_message Tests
# ==============================================================================


def test_keep_only_last_todo_message_disabled_by_default() -> None:
    """Test that keep_only_last_todo_message is disabled by default."""
    middleware = TodoListMiddleware()
    assert middleware.keep_only_last_todo_message is False


def test_keep_only_last_todo_message_enabled() -> None:
    """Test that keep_only_last_todo_message can be enabled."""
    middleware = TodoListMiddleware(keep_only_last_todo_message=True)
    assert middleware.keep_only_last_todo_message is True


def test_filter_todo_messages_no_filtering_when_disabled() -> None:
    """Test that messages are not filtered when keep_only_last_todo_message is False."""
    middleware = TodoListMiddleware(keep_only_last_todo_message=False)

    # Create messages with multiple todo updates
    messages = [
        HumanMessage(content="Task 1"),
        AIMessage(
            content="Creating todo",
            tool_calls=[
                {
                    "name": "write_todos",
                    "args": {"todos": [{"content": "Task 1", "status": "pending"}]},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="Updated todo list to [{'content': 'Task 1', 'status': 'pending'}]",
            tool_call_id="call_1",
        ),
        AIMessage(
            content="Updating todo",
            tool_calls=[
                {
                    "name": "write_todos",
                    "args": {"todos": [{"content": "Task 1", "status": "completed"}]},
                    "id": "call_2",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="Updated todo list to [{'content': 'Task 1', 'status': 'completed'}]",
            tool_call_id="call_2",
        ),
    ]

    model = GenericFakeChatModel(messages=iter([AIMessage(content="response")]))
    request = ModelRequest(
        model=model,
        system_prompt=None,
        messages=messages,
        tool_choice=None,
        tools=[],
        response_format=None,
        state=AgentState(messages=messages),
        runtime=_fake_runtime(),
        model_settings={},
    )

    captured_request = None

    def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal captured_request
        captured_request = req
        return ModelResponse(result=[AIMessage(content="response")])

    middleware.wrap_model_call(request, mock_handler)

    # All messages should be preserved
    assert captured_request is not None
    assert len(captured_request.messages) == 5


def test_filter_todo_messages_keeps_only_last_pair() -> None:
    """Test that only the last write_todos AI/ToolMessage pair is kept."""
    middleware = TodoListMiddleware(keep_only_last_todo_message=True)

    # Create messages with multiple todo updates
    messages = [
        HumanMessage(content="Task 1"),
        AIMessage(
            content="Creating todo",
            tool_calls=[
                {
                    "name": "write_todos",
                    "args": {"todos": [{"content": "Task 1", "status": "pending"}]},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="Updated todo list to [{'content': 'Task 1', 'status': 'pending'}]",
            tool_call_id="call_1",
        ),
        AIMessage(
            content="Updating todo",
            tool_calls=[
                {
                    "name": "write_todos",
                    "args": {"todos": [{"content": "Task 1", "status": "completed"}]},
                    "id": "call_2",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="Updated todo list to [{'content': 'Task 1', 'status': 'completed'}]",
            tool_call_id="call_2",
        ),
    ]

    model = GenericFakeChatModel(messages=iter([AIMessage(content="response")]))
    request = ModelRequest(
        model=model,
        system_prompt=None,
        messages=messages,
        tool_choice=None,
        tools=[],
        response_format=None,
        state=AgentState(messages=messages),
        runtime=_fake_runtime(),
        model_settings={},
    )

    captured_request = None

    def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal captured_request
        captured_request = req
        return ModelResponse(result=[AIMessage(content="response")])

    middleware.wrap_model_call(request, mock_handler)

    # Should keep: HumanMessage, last AIMessage with write_todos, last ToolMessage
    assert captured_request is not None
    assert len(captured_request.messages) == 3
    assert isinstance(captured_request.messages[0], HumanMessage)
    assert isinstance(captured_request.messages[1], AIMessage)
    assert captured_request.messages[1].tool_calls[0]["id"] == "call_2"
    assert isinstance(captured_request.messages[2], ToolMessage)
    assert captured_request.messages[2].tool_call_id == "call_2"


def test_filter_todo_messages_preserves_non_todo_messages() -> None:
    """Test that non-todo messages are preserved during filtering."""
    middleware = TodoListMiddleware(keep_only_last_todo_message=True)

    # Create messages with todo updates and other messages
    messages = [
        HumanMessage(content="Task 1"),
        AIMessage(content="Regular AI message"),
        AIMessage(
            content="Creating todo",
            tool_calls=[
                {
                    "name": "write_todos",
                    "args": {"todos": [{"content": "Task 1", "status": "pending"}]},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="Updated todo list to [{'content': 'Task 1', 'status': 'pending'}]",
            tool_call_id="call_1",
        ),
        HumanMessage(content="Task 2"),
        AIMessage(
            content="Updating todo",
            tool_calls=[
                {
                    "name": "write_todos",
                    "args": {"todos": [{"content": "Task 1", "status": "completed"}]},
                    "id": "call_2",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="Updated todo list to [{'content': 'Task 1', 'status': 'completed'}]",
            tool_call_id="call_2",
        ),
        AIMessage(content="Another regular AI message"),
    ]

    model = GenericFakeChatModel(messages=iter([AIMessage(content="response")]))
    request = ModelRequest(
        model=model,
        system_prompt=None,
        messages=messages,
        tool_choice=None,
        tools=[],
        response_format=None,
        state=AgentState(messages=messages),
        runtime=_fake_runtime(),
        model_settings={},
    )

    captured_request = None

    def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal captured_request
        captured_request = req
        return ModelResponse(result=[AIMessage(content="response")])

    middleware.wrap_model_call(request, mock_handler)

    # Should keep: all non-todo messages + last todo pair
    assert captured_request is not None
    assert len(captured_request.messages) == 6
    assert captured_request.messages[0].content == "Task 1"
    assert captured_request.messages[1].content == "Regular AI message"
    assert captured_request.messages[2].content == "Task 2"
    assert captured_request.messages[3].tool_calls[0]["id"] == "call_2"
    assert captured_request.messages[4].tool_call_id == "call_2"
    assert captured_request.messages[5].content == "Another regular AI message"


def test_filter_todo_messages_with_mixed_tool_calls() -> None:
    """Test filtering when AI messages have both write_todos and other tool calls."""
    middleware = TodoListMiddleware(keep_only_last_todo_message=True)

    # Create messages with mixed tool calls
    messages = [
        HumanMessage(content="Task 1"),
        AIMessage(
            content="Creating todo and calling other tool",
            tool_calls=[
                {
                    "name": "write_todos",
                    "args": {"todos": [{"content": "Task 1", "status": "pending"}]},
                    "id": "call_1",
                    "type": "tool_call",
                },
                {
                    "name": "other_tool",
                    "args": {"param": "value"},
                    "id": "call_other_1",
                    "type": "tool_call",
                },
            ],
        ),
        ToolMessage(
            content="Updated todo list to [{'content': 'Task 1', 'status': 'pending'}]",
            tool_call_id="call_1",
        ),
        ToolMessage(content="Other tool result", tool_call_id="call_other_1"),
        AIMessage(
            content="Updating todo",
            tool_calls=[
                {
                    "name": "write_todos",
                    "args": {"todos": [{"content": "Task 1", "status": "completed"}]},
                    "id": "call_2",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="Updated todo list to [{'content': 'Task 1', 'status': 'completed'}]",
            tool_call_id="call_2",
        ),
    ]

    model = GenericFakeChatModel(messages=iter([AIMessage(content="response")]))
    request = ModelRequest(
        model=model,
        system_prompt=None,
        messages=messages,
        tool_choice=None,
        tools=[],
        response_format=None,
        state=AgentState(messages=messages),
        runtime=_fake_runtime(),
        model_settings={},
    )

    captured_request = None

    def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal captured_request
        captured_request = req
        return ModelResponse(result=[AIMessage(content="response")])

    middleware.wrap_model_call(request, mock_handler)

    # First AI message with mixed tool calls should be removed
    # Other tool message should be preserved (not a todo message)
    # Last todo pair should be kept
    assert captured_request is not None
    assert len(captured_request.messages) == 4
    assert isinstance(captured_request.messages[0], HumanMessage)
    assert isinstance(captured_request.messages[1], ToolMessage)
    assert captured_request.messages[1].content == "Other tool result"
    assert isinstance(captured_request.messages[2], AIMessage)
    assert captured_request.messages[2].tool_calls[0]["id"] == "call_2"
    assert isinstance(captured_request.messages[3], ToolMessage)
    assert captured_request.messages[3].tool_call_id == "call_2"


def test_filter_todo_messages_no_todo_messages() -> None:
    """Test filtering when there are no todo messages."""
    middleware = TodoListMiddleware(keep_only_last_todo_message=True)

    # Create messages without any todo updates
    messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there"),
        HumanMessage(content="How are you?"),
        AIMessage(content="I'm doing well"),
    ]

    model = GenericFakeChatModel(messages=iter([AIMessage(content="response")]))
    request = ModelRequest(
        model=model,
        system_prompt=None,
        messages=messages,
        tool_choice=None,
        tools=[],
        response_format=None,
        state=AgentState(messages=messages),
        runtime=_fake_runtime(),
        model_settings={},
    )

    captured_request = None

    def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal captured_request
        captured_request = req
        return ModelResponse(result=[AIMessage(content="response")])

    middleware.wrap_model_call(request, mock_handler)

    # All messages should be preserved
    assert captured_request is not None
    assert len(captured_request.messages) == 4


def test_filter_todo_messages_single_todo_pair() -> None:
    """Test filtering when there's only one todo message pair."""
    middleware = TodoListMiddleware(keep_only_last_todo_message=True)

    # Create messages with single todo update
    messages = [
        HumanMessage(content="Task 1"),
        AIMessage(
            content="Creating todo",
            tool_calls=[
                {
                    "name": "write_todos",
                    "args": {"todos": [{"content": "Task 1", "status": "pending"}]},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="Updated todo list to [{'content': 'Task 1', 'status': 'pending'}]",
            tool_call_id="call_1",
        ),
    ]

    model = GenericFakeChatModel(messages=iter([AIMessage(content="response")]))
    request = ModelRequest(
        model=model,
        system_prompt=None,
        messages=messages,
        tool_choice=None,
        tools=[],
        response_format=None,
        state=AgentState(messages=messages),
        runtime=_fake_runtime(),
        model_settings={},
    )

    captured_request = None

    def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal captured_request
        captured_request = req
        return ModelResponse(result=[AIMessage(content="response")])

    middleware.wrap_model_call(request, mock_handler)

    # All messages should be preserved since there's only one todo pair
    assert captured_request is not None
    assert len(captured_request.messages) == 3


def test_filter_todo_messages_multiple_updates() -> None:
    """Test filtering with three todo updates (keeps only the last one)."""
    middleware = TodoListMiddleware(keep_only_last_todo_message=True)

    # Create messages with three todo updates
    messages = [
        HumanMessage(content="Start"),
        AIMessage(
            content="First todo",
            tool_calls=[
                {
                    "name": "write_todos",
                    "args": {"todos": [{"content": "Task 1", "status": "pending"}]},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="Updated todo list to [{'content': 'Task 1', 'status': 'pending'}]",
            tool_call_id="call_1",
        ),
        AIMessage(
            content="Second todo",
            tool_calls=[
                {
                    "name": "write_todos",
                    "args": {"todos": [{"content": "Task 1", "status": "in_progress"}]},
                    "id": "call_2",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="Updated todo list to [{'content': 'Task 1', 'status': 'in_progress'}]",
            tool_call_id="call_2",
        ),
        AIMessage(
            content="Third todo",
            tool_calls=[
                {
                    "name": "write_todos",
                    "args": {"todos": [{"content": "Task 1", "status": "completed"}]},
                    "id": "call_3",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="Updated todo list to [{'content': 'Task 1', 'status': 'completed'}]",
            tool_call_id="call_3",
        ),
    ]

    model = GenericFakeChatModel(messages=iter([AIMessage(content="response")]))
    request = ModelRequest(
        model=model,
        system_prompt=None,
        messages=messages,
        tool_choice=None,
        tools=[],
        response_format=None,
        state=AgentState(messages=messages),
        runtime=_fake_runtime(),
        model_settings={},
    )

    captured_request = None

    def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal captured_request
        captured_request = req
        return ModelResponse(result=[AIMessage(content="response")])

    middleware.wrap_model_call(request, mock_handler)

    # Should keep only: HumanMessage + last todo pair
    assert captured_request is not None
    assert len(captured_request.messages) == 3
    assert isinstance(captured_request.messages[0], HumanMessage)
    assert isinstance(captured_request.messages[1], AIMessage)
    assert captured_request.messages[1].tool_calls[0]["id"] == "call_3"
    assert isinstance(captured_request.messages[2], ToolMessage)
    assert captured_request.messages[2].tool_call_id == "call_3"


def test_filter_todo_messages_empty_messages_list() -> None:
    """Test filtering with an empty messages list."""
    middleware = TodoListMiddleware(keep_only_last_todo_message=True)

    messages: list[AnyMessage] = []

    model = GenericFakeChatModel(messages=iter([AIMessage(content="response")]))
    request = ModelRequest(
        model=model,
        system_prompt=None,
        messages=messages,
        tool_choice=None,
        tools=[],
        response_format=None,
        state=AgentState(messages=messages),
        runtime=_fake_runtime(),
        model_settings={},
    )

    captured_request = None

    def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal captured_request
        captured_request = req
        return ModelResponse(result=[AIMessage(content="response")])

    middleware.wrap_model_call(request, mock_handler)

    # Should return empty list
    assert captured_request is not None
    assert len(captured_request.messages) == 0


async def test_filter_todo_messages_keeps_only_last_pair_async() -> None:
    """Test async version - only the last write_todos AI/ToolMessage pair is kept."""
    middleware = TodoListMiddleware(keep_only_last_todo_message=True)

    # Create messages with multiple todo updates
    messages = [
        HumanMessage(content="Task 1"),
        AIMessage(
            content="Creating todo",
            tool_calls=[
                {
                    "name": "write_todos",
                    "args": {"todos": [{"content": "Task 1", "status": "pending"}]},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="Updated todo list to [{'content': 'Task 1', 'status': 'pending'}]",
            tool_call_id="call_1",
        ),
        AIMessage(
            content="Updating todo",
            tool_calls=[
                {
                    "name": "write_todos",
                    "args": {"todos": [{"content": "Task 1", "status": "completed"}]},
                    "id": "call_2",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="Updated todo list to [{'content': 'Task 1', 'status': 'completed'}]",
            tool_call_id="call_2",
        ),
    ]

    model = GenericFakeChatModel(messages=iter([AIMessage(content="response")]))
    request = ModelRequest(
        model=model,
        system_prompt=None,
        messages=messages,
        tool_choice=None,
        tools=[],
        response_format=None,
        state=AgentState(messages=messages),
        runtime=_fake_runtime(),
        model_settings={},
    )

    captured_request = None

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal captured_request
        captured_request = req
        return ModelResponse(result=[AIMessage(content="response")])

    await middleware.awrap_model_call(request, mock_handler)

    # Should keep: HumanMessage, last AIMessage with write_todos, last ToolMessage
    assert captured_request is not None
    assert len(captured_request.messages) == 3
    assert isinstance(captured_request.messages[0], HumanMessage)
    assert isinstance(captured_request.messages[1], AIMessage)
    assert captured_request.messages[1].tool_calls[0]["id"] == "call_2"
    assert isinstance(captured_request.messages[2], ToolMessage)
    assert captured_request.messages[2].tool_call_id == "call_2"


async def test_filter_todo_messages_preserves_non_todo_messages_async() -> None:
    """Test async version - non-todo messages are preserved during filtering."""
    middleware = TodoListMiddleware(keep_only_last_todo_message=True)

    # Create messages with todo updates and other messages
    messages = [
        HumanMessage(content="Task 1"),
        AIMessage(content="Regular AI message"),
        AIMessage(
            content="Creating todo",
            tool_calls=[
                {
                    "name": "write_todos",
                    "args": {"todos": [{"content": "Task 1", "status": "pending"}]},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="Updated todo list to [{'content': 'Task 1', 'status': 'pending'}]",
            tool_call_id="call_1",
        ),
        HumanMessage(content="Task 2"),
        AIMessage(
            content="Updating todo",
            tool_calls=[
                {
                    "name": "write_todos",
                    "args": {"todos": [{"content": "Task 1", "status": "completed"}]},
                    "id": "call_2",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="Updated todo list to [{'content': 'Task 1', 'status': 'completed'}]",
            tool_call_id="call_2",
        ),
        AIMessage(content="Another regular AI message"),
    ]

    model = GenericFakeChatModel(messages=iter([AIMessage(content="response")]))
    request = ModelRequest(
        model=model,
        system_prompt=None,
        messages=messages,
        tool_choice=None,
        tools=[],
        response_format=None,
        state=AgentState(messages=messages),
        runtime=_fake_runtime(),
        model_settings={},
    )

    captured_request = None

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal captured_request
        captured_request = req
        return ModelResponse(result=[AIMessage(content="response")])

    await middleware.awrap_model_call(request, mock_handler)

    # Should keep: all non-todo messages + last todo pair
    assert captured_request is not None
    assert len(captured_request.messages) == 6
    assert captured_request.messages[0].content == "Task 1"
    assert captured_request.messages[1].content == "Regular AI message"
    assert captured_request.messages[2].content == "Task 2"
    assert captured_request.messages[3].tool_calls[0]["id"] == "call_2"
    assert captured_request.messages[4].tool_call_id == "call_2"
    assert captured_request.messages[5].content == "Another regular AI message"

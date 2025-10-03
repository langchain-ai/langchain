"""Tests for PlanningMiddleware."""

import pytest

from langchain_core.messages import HumanMessage
from langchain.agents.middleware.planning import (
    PlanningMiddleware,
    PlanningState,
    WRITE_TODOS_SYSTEM_PROMPT,
    write_todos,
    WRITE_TODOS_TOOL_DESCRIPTION,
)
from langchain.agents.middleware.types import ModelRequest
from langchain.agents.factory import create_agent

from ..model import FakeToolCallingModel


def test_planning_middleware_initialization() -> None:
    """Test that PlanningMiddleware initializes correctly."""
    middleware = PlanningMiddleware()
    assert middleware.state_schema == PlanningState
    assert len(middleware.tools) == 1
    assert middleware.tools[0].name == "write_todos"


@pytest.mark.parametrize(
    "original_prompt,expected_prompt_prefix",
    [
        ("Original prompt", "Original prompt\n\n## `write_todos`"),
        (None, "## `write_todos`"),
    ],
)
def test_planning_middleware_modify_model_request(original_prompt, expected_prompt_prefix) -> None:
    """Test that modify_model_request handles system prompts correctly."""
    middleware = PlanningMiddleware()
    model = FakeToolCallingModel()

    request = ModelRequest(
        model=model,
        system_prompt=original_prompt,
        messages=[HumanMessage(content="Hello")],
        tool_choice=None,
        tools=[],
        response_format=None,
        model_settings={},
    )

    state: PlanningState = {"messages": [HumanMessage(content="Hello")]}
    modified_request = middleware.modify_model_request(request, state, None)
    assert modified_request.system_prompt.startswith(expected_prompt_prefix)


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
def test_planning_middleware_write_todos_tool_execution(todos, expected_message) -> None:
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
def test_planning_middleware_write_todos_tool_validation_errors(invalid_todos) -> None:
    """Test that the write_todos tool rejects invalid input."""
    tool_call = {
        "args": {"todos": invalid_todos},
        "name": "write_todos",
        "type": "tool_call",
        "id": "test_call",
    }
    with pytest.raises(Exception):
        write_todos.invoke(tool_call)


def test_planning_middleware_agent_creation_with_middleware() -> None:
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
    middleware = PlanningMiddleware()
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


def test_planning_middleware_custom_system_prompt() -> None:
    """Test that PlanningMiddleware can be initialized with custom system prompt."""
    custom_system_prompt = "Custom todo system prompt for testing"
    middleware = PlanningMiddleware(system_prompt=custom_system_prompt)
    model = FakeToolCallingModel()

    request = ModelRequest(
        model=model,
        system_prompt="Original prompt",
        messages=[HumanMessage(content="Hello")],
        tool_choice=None,
        tools=[],
        response_format=None,
        model_settings={},
    )

    state: PlanningState = {"messages": [HumanMessage(content="Hello")]}
    modified_request = middleware.modify_model_request(request, state, None)
    assert modified_request.system_prompt == f"Original prompt\n\n{custom_system_prompt}"


def test_planning_middleware_custom_tool_description() -> None:
    """Test that PlanningMiddleware can be initialized with custom tool description."""
    custom_tool_description = "Custom tool description for testing"
    middleware = PlanningMiddleware(tool_description=custom_tool_description)

    assert len(middleware.tools) == 1
    tool = middleware.tools[0]
    assert tool.description == custom_tool_description


def test_planning_middleware_custom_system_prompt_and_tool_description() -> None:
    """Test that PlanningMiddleware can be initialized with both custom prompts."""
    custom_system_prompt = "Custom system prompt"
    custom_tool_description = "Custom tool description"
    middleware = PlanningMiddleware(
        system_prompt=custom_system_prompt,
        tool_description=custom_tool_description,
    )

    # Verify system prompt
    model = FakeToolCallingModel()
    request = ModelRequest(
        model=model,
        system_prompt=None,
        messages=[HumanMessage(content="Hello")],
        tool_choice=None,
        tools=[],
        response_format=None,
        model_settings={},
    )

    state: PlanningState = {"messages": [HumanMessage(content="Hello")]}
    modified_request = middleware.modify_model_request(request, state, None)
    assert modified_request.system_prompt == custom_system_prompt

    # Verify tool description
    assert len(middleware.tools) == 1
    tool = middleware.tools[0]
    assert tool.description == custom_tool_description


def test_planning_middleware_default_prompts() -> None:
    """Test that PlanningMiddleware uses default prompts when none provided."""
    middleware = PlanningMiddleware()

    # Verify default system prompt
    assert middleware.system_prompt == WRITE_TODOS_SYSTEM_PROMPT

    # Verify default tool description
    assert middleware.tool_description == WRITE_TODOS_TOOL_DESCRIPTION
    assert len(middleware.tools) == 1
    tool = middleware.tools[0]
    assert tool.description == WRITE_TODOS_TOOL_DESCRIPTION

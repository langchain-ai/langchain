"""Tests for auto-completing ToolMessage in Command returns from tools.

Verifies that tools can return Command(update={"jump_to": "end"}) without
manually constructing ToolMessage objects. The agent should auto-create
the ToolMessage using the tool_call_id from the request context.

See: https://github.com/langchain-ai/langchain/issues/34884
"""

from typing import Any

from langchain_core.messages import HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain.agents.factory import create_agent
from tests.unit_tests.agents.model import FakeToolCallingModel


def test_tool_command_jump_to_end_without_tool_message() -> None:
    """Tool returns Command with jump_to='end' but no ToolMessage.

    The agent should auto-create the ToolMessage and stop the loop.
    """
    model_call_count = [0]

    class CountingModel(FakeToolCallingModel):
        def _generate(self, *args: Any, **kwargs: Any) -> Any:
            model_call_count[0] += 1
            return super()._generate(*args, **kwargs)

    @tool
    def validate_and_finish(data: str) -> Command[dict[str, Any]]:
        """Validate data and finish if valid."""
        if data == "valid":
            return Command(update={"jump_to": "end"})
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="validation failed, please retry",
                        tool_call_id="1",
                        name="validate_and_finish",
                    )
                ],
            }
        )

    model = CountingModel(
        tool_calls=[
            [ToolCall(name="validate_and_finish", args={"data": "valid"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[validate_and_finish],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("validate this")]},
        {"configurable": {"thread_id": "test"}},
    )

    # Model should only be called once (tool ends the loop)
    assert model_call_count[0] == 1
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1


def test_tool_command_jump_to_end_with_string_content() -> None:
    """Tool returns Command with jump_to='end' and a string message content.

    The string should be auto-wrapped into a ToolMessage.
    """
    model_call_count = [0]

    class CountingModel(FakeToolCallingModel):
        def _generate(self, *args: Any, **kwargs: Any) -> Any:
            model_call_count[0] += 1
            return super()._generate(*args, **kwargs)

    @tool
    def complete_task(task_id: str) -> Command[dict[str, Any]]:
        """Complete a task and end the loop with a message."""
        return Command(
            update={
                "messages": [f"Task {task_id} completed successfully."],
                "jump_to": "end",
            }
        )

    model = CountingModel(
        tool_calls=[
            [ToolCall(name="complete_task", args={"task_id": "123"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[complete_task],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("complete task 123")]},
        {"configurable": {"thread_id": "test"}},
    )

    assert model_call_count[0] == 1
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert "completed successfully" in tool_msgs[0].content


def test_tool_command_conditional_jump() -> None:
    """Tool dynamically decides to end or continue without manual ToolMessage.

    When valid: returns Command with jump_to='end' and no ToolMessage.
    When invalid: returns plain string (continues loop normally).
    """
    model_call_count = [0]

    class CountingModel(FakeToolCallingModel):
        def _generate(self, *args: Any, **kwargs: Any) -> Any:
            model_call_count[0] += 1
            return super()._generate(*args, **kwargs)

    call_count = [0]

    @tool
    def smart_validator(value: str) -> Any:
        """Validate and decide flow based on result."""
        call_count[0] += 1
        if value.isdigit() and int(value) > 0:
            # Dynamically end the loop
            return Command(update={"jump_to": "end"})
        # Continue normally — return a plain string
        return f"invalid: {value}, need positive number"

    model = CountingModel(
        tool_calls=[
            [ToolCall(name="smart_validator", args={"value": "abc"}, id="1")],
            [ToolCall(name="smart_validator", args={"value": "42"}, id="2")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[smart_validator],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("validate")]},
        {"configurable": {"thread_id": "test"}},
    )

    # First call: invalid → continues to model. Second call: valid → ends loop.
    assert call_count[0] == 2
    assert model_call_count[0] == 2  # called twice (initial + after first tool fail)
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 2
    assert "invalid" in tool_msgs[0].content


def test_tool_command_without_jump_to_unaffected() -> None:
    """Command without jump_to still requires explicit ToolMessage (unchanged behavior)."""

    @tool
    def normal_command_tool(data: str) -> Command[dict[str, Any]]:
        """Return a Command with explicit ToolMessage, no jump_to."""
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"processed: {data}",
                        tool_call_id="1",
                        name="normal_command_tool",
                    )
                ],
            }
        )

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="normal_command_tool", args={"data": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[normal_command_tool],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("process this")]},
        {"configurable": {"thread_id": "test"}},
    )

    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert "processed: test" in tool_msgs[0].content

"""Tests for dynamic tool jump_to feature."""

from typing import Any

from langchain_core.messages import HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain.agents.factory import create_agent
from tests.unit_tests.agents.model import FakeToolCallingModel


@tool
def verify_value(value: str) -> str:
    """Verify value and return result."""
    return f"verified: {value}"


def test_tool_jump_to_end_stops_loop() -> None:
    """Tool can set jump_to='end' to stop the agent loop without calling model again."""
    model_call_count = [0]

    class CountingModel(FakeToolCallingModel):
        def _generate(self, *args: Any, **kwargs: Any) -> Any:
            model_call_count[0] += 1
            return super()._generate(*args, **kwargs)

    @tool
    def validate_and_finish(data: str) -> Command[dict[str, Any]]:
        """Validate data and finish if valid."""
        if data == "valid":
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content="validation passed",
                            tool_call_id="1",
                            name="validate_and_finish",
                        )
                    ],
                    "jump_to": "end",
                }
            )
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

    assert model_call_count[0] == 1
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert "validation passed" in tool_msgs[0].content


def test_tool_without_jump_to_continues_to_model() -> None:
    """Tool without jump_to continues to model as usual."""
    model_call_count = [0]

    class CountingModel(FakeToolCallingModel):
        def _generate(self, *args: Any, **kwargs: Any) -> Any:
            model_call_count[0] += 1
            return super()._generate(*args, **kwargs)

    model = CountingModel(
        tool_calls=[
            [ToolCall(name="verify_value", args={"value": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[verify_value],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("verify test")]},
        {"configurable": {"thread_id": "test"}},
    )

    assert model_call_count[0] == 2
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1


def test_tool_jump_to_model_continues_loop() -> None:
    """Tool can set jump_to='model' explicitly to continue."""
    model_call_count = [0]

    class CountingModel(FakeToolCallingModel):
        def _generate(self, *args: Any, **kwargs: Any) -> Any:
            model_call_count[0] += 1
            return super()._generate(*args, **kwargs)

    @tool
    def process_and_continue(data: str) -> Command[dict[str, Any]]:
        """Process data and continue to model."""
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"processed: {data}",
                        tool_call_id="1",
                        name="process_and_continue",
                    )
                ],
                "jump_to": "model",
            }
        )

    model = CountingModel(
        tool_calls=[
            [ToolCall(name="process_and_continue", args={"data": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[process_and_continue],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("process this")]},
        {"configurable": {"thread_id": "test"}},
    )

    assert model_call_count[0] == 2
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1


def test_return_direct_takes_precedence_over_jump_to() -> None:
    """Tool with return_direct=True always ends loop, ignoring jump_to."""
    model_call_count = [0]

    class CountingModel(FakeToolCallingModel):
        def _generate(self, *args: Any, **kwargs: Any) -> Any:
            model_call_count[0] += 1
            return super()._generate(*args, **kwargs)

    @tool(return_direct=True)
    def direct_return_tool(data: str) -> str:
        """Tool with return_direct that should always end."""
        return f"direct: {data}"

    model = CountingModel(
        tool_calls=[
            [ToolCall(name="direct_return_tool", args={"data": "test"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[direct_return_tool],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("test direct")]},
        {"configurable": {"thread_id": "test"}},
    )

    assert model_call_count[0] == 1
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1


def test_conditional_jump_based_on_validation() -> None:
    """Tool dynamically decides whether to end loop based on validation result."""
    model_call_count = [0]

    class CountingModel(FakeToolCallingModel):
        def _generate(self, *args: Any, **kwargs: Any) -> Any:
            model_call_count[0] += 1
            return super()._generate(*args, **kwargs)

    call_count = [0]

    @tool
    def smart_validator(value: str) -> Command[dict[str, Any]]:
        """Validate and decide flow based on result."""
        call_count[0] += 1
        call_id = str(call_count[0])

        if value.isdigit() and int(value) > 0:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"valid number: {value}",
                            tool_call_id=call_id,
                            name="smart_validator",
                        )
                    ],
                    "jump_to": "end",
                }
            )
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"invalid: {value}, need positive number",
                        tool_call_id=call_id,
                        name="smart_validator",
                    )
                ],
            }
        )

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

    assert call_count[0] == 2
    assert model_call_count[0] == 2
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 2
    assert "invalid" in tool_msgs[0].content
    assert "valid number" in tool_msgs[1].content


def test_jump_to_end_with_final_message() -> None:
    """Tool sets jump_to='end' with informative final message."""

    @tool
    def complete_task(task_id: str) -> Command[dict[str, Any]]:
        """Complete task and end agent loop."""
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Task {task_id} completed successfully. No further action needed.",
                        tool_call_id="1",
                        name="complete_task",
                    )
                ],
                "jump_to": "end",
            }
        )

    model = FakeToolCallingModel(
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

    messages = result["messages"]
    last_msg = messages[-1]
    assert isinstance(last_msg, ToolMessage)
    assert "completed successfully" in last_msg.content

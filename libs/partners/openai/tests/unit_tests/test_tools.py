from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import Tool

from langchain_openai import ChatOpenAI, custom_tool


def test_custom_tool() -> None:
    @custom_tool
    def my_tool(x: str) -> str:
        """Do thing."""
        return "a" + x

    # Test decorator
    assert isinstance(my_tool, Tool)
    assert my_tool.metadata == {"type": "custom_tool"}
    assert my_tool.description == "Do thing."

    result = my_tool.invoke(
        {
            "type": "tool_call",
            "name": "my_tool",
            "args": {"whatever": "b"},
            "id": "abc",
            "extras": {"type": "custom_tool_call"},
        }
    )
    assert result == ToolMessage(
        [{"type": "custom_tool_call_output", "output": "ab"}],
        name="my_tool",
        tool_call_id="abc",
    )

    # Test tool schema
    llm = ChatOpenAI(model="gpt-4.1", use_responses_api=True).bind_tools([my_tool])
    assert llm.kwargs == {  # type: ignore[attr-defined]
        "tools": [{"type": "custom", "name": "my_tool", "description": "Do thing."}]
    }

    # Test passing messages back
    message_history = [
        HumanMessage("Use the tool"),
        AIMessage(
            [
                {
                    "type": "custom_tool_call",
                    "id": "ctc_abc123",
                    "call_id": "abc",
                    "name": "my_tool",
                    "input": "a",
                }
            ],
            tool_calls=[
                {
                    "type": "tool_call",
                    "name": "my_tool",
                    "args": {"__arg1": "a"},
                    "id": "abc",
                }
            ],
        ),
        result,
    ]
    payload = llm._get_request_payload(message_history)  # type: ignore[attr-defined]
    expected_input = [
        {"content": "Use the tool", "role": "user"},
        {
            "type": "custom_tool_call",
            "id": "ctc_abc123",
            "call_id": "abc",
            "name": "my_tool",
            "input": "a",
        },
        {"type": "custom_tool_call_output", "call_id": "abc", "output": "ab"},
    ]
    assert payload["input"] == expected_input


async def test_async_custom_tool() -> None:
    @custom_tool
    async def my_async_tool(x: str) -> str:
        """Do async thing."""
        return "a" + x

    # Test decorator
    assert isinstance(my_async_tool, Tool)
    assert my_async_tool.metadata == {"type": "custom_tool"}
    assert my_async_tool.description == "Do async thing."

    result = await my_async_tool.ainvoke(
        {
            "type": "tool_call",
            "name": "my_async_tool",
            "args": {"whatever": "b"},
            "id": "abc",
            "extras": {"type": "custom_tool_call"},
        }
    )
    assert result == ToolMessage(
        [{"type": "custom_tool_call_output", "output": "ab"}],
        name="my_async_tool",
        tool_call_id="abc",
    )

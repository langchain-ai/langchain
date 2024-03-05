from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.utils.tool_examples import tool_example_to_messages


class FakeCall(BaseModel):
    data: str


def test_valid_example_conversion():
    example = {"input": "This is a valid example", "tool_calls": []}
    expected_messages = [
        HumanMessage(content="This is a valid example"),
        AIMessage(content="", additional_kwargs={"tool_calls": []}),
    ]
    assert tool_example_to_messages(example) == expected_messages


def test_multiple_tool_calls():
    example = {
        "input": "This is an example",
        "tool_calls": [
            FakeCall(data="ToolCall1"),
            FakeCall(data="ToolCall2"),
            FakeCall(data="ToolCall3"),
        ],
    }
    messages = tool_example_to_messages(example)
    assert len(messages) == 5
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert isinstance(messages[2], ToolMessage)
    assert isinstance(messages[3], ToolMessage)
    assert isinstance(messages[4], ToolMessage)
    assert messages[1].additional_kwargs["tool_calls"] == [
        {
            "id": messages[2].tool_call_id,
            "type": "function",
            "function": {"name": "FakeCall", "arguments": '{"data": "ToolCall1"}'},
        },
        {
            "id": messages[3].tool_call_id,
            "type": "function",
            "function": {"name": "FakeCall", "arguments": '{"data": "ToolCall2"}'},
        },
        {
            "id": messages[4].tool_call_id,
            "type": "function",
            "function": {"name": "FakeCall", "arguments": '{"data": "ToolCall3"}'},
        },
    ]


def test_tool_outputs():
    example = {
        "input": "This is an example",
        "tool_calls": [
            FakeCall(data="ToolCall1"),
        ],
        "tool_outputs": ["Output1"],
    }
    messages = tool_example_to_messages(example)
    assert len(messages) == 3
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert isinstance(messages[2], ToolMessage)
    assert messages[1].additional_kwargs["tool_calls"] == [
        {
            "id": messages[2].tool_call_id,
            "type": "function",
            "function": {"name": "FakeCall", "arguments": '{"data": "ToolCall1"}'},
        },
    ]
    assert messages[2].content == "Output1"

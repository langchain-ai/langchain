from langchain_core.messages import AIMessage, ToolCall, ToolMessage

from langchain_classic.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain_classic.agents.output_parsers.openai_tools import (
    parse_ai_message_to_openai_tool_action,
)


def test_calls_convert_agent_action_to_messages() -> None:
    additional_kwargs1 = {
        "tool_calls": [
            {
                "id": "call_abcd12345",
                "function": {"arguments": '{"a": 3, "b": 5}', "name": "add"},
                "type": "function",
            },
        ],
    }
    message1 = AIMessage(content="", additional_kwargs=additional_kwargs1)

    actions1 = parse_ai_message_to_openai_tool_action(message1)
    additional_kwargs2 = {
        "tool_calls": [
            {
                "id": "call_abcd54321",
                "function": {"arguments": '{"a": 3, "b": 5}', "name": "subtract"},
                "type": "function",
            },
        ],
    }
    message2 = AIMessage(content="", additional_kwargs=additional_kwargs2)
    actions2 = parse_ai_message_to_openai_tool_action(message2)

    additional_kwargs3 = {
        "tool_calls": [
            {
                "id": "call_abcd67890",
                "function": {"arguments": '{"a": 3, "b": 5}', "name": "multiply"},
                "type": "function",
            },
            {
                "id": "call_abcd09876",
                "function": {"arguments": '{"a": 3, "b": 5}', "name": "divide"},
                "type": "function",
            },
        ],
    }
    message3 = AIMessage(content="", additional_kwargs=additional_kwargs3)
    actions3 = parse_ai_message_to_openai_tool_action(message3)

    message4 = AIMessage(
        content="",
        tool_calls=[
            ToolCall(
                name="exponentiate",
                args={"a": 3, "b": 5},
                id="call_abc02468",
                type="tool_call",
            ),
        ],
    )
    actions4 = parse_ai_message_to_openai_tool_action(message4)

    # for mypy
    assert isinstance(actions1, list)
    assert isinstance(actions2, list)
    assert isinstance(actions3, list)
    assert isinstance(actions4, list)

    intermediate_steps = [
        (actions1[0], "observation1"),
        (actions2[0], "observation2"),
        (actions3[0], "observation3"),
        (actions3[1], "observation4"),
        (actions4[0], "observation4"),
    ]
    expected_messages = [
        message1,
        ToolMessage(
            tool_call_id="call_abcd12345",
            content="observation1",
            additional_kwargs={"name": "add"},
        ),
        message2,
        ToolMessage(
            tool_call_id="call_abcd54321",
            content="observation2",
            additional_kwargs={"name": "subtract"},
        ),
        message3,
        ToolMessage(
            tool_call_id="call_abcd67890",
            content="observation3",
            additional_kwargs={"name": "multiply"},
        ),
        ToolMessage(
            tool_call_id="call_abcd09876",
            content="observation4",
            additional_kwargs={"name": "divide"},
        ),
        message4,
        ToolMessage(
            tool_call_id="call_abc02468",
            content="observation4",
            additional_kwargs={"name": "exponentiate"},
        ),
    ]
    output = format_to_openai_tool_messages(intermediate_steps)
    assert output == expected_messages


def test_handles_empty_input_list() -> None:
    output = format_to_openai_tool_messages([])
    assert output == []

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.output_parsers.openai_tools import (
    parse_tool_call,
)

from langchain_community.chat_models.sparkllm import (
    convert_dict_to_message,
    convert_message_to_dict,
)


def test__convert_dict_to_message_human() -> None:
    message_dict = {"role": "user", "content": "foo"}
    result = convert_dict_to_message(message_dict)
    expected_output = HumanMessage(content="foo")
    assert result == expected_output


def test__convert_dict_to_message_ai() -> None:
    message_dict = {"role": "assistant", "content": "foo"}
    result = convert_dict_to_message(message_dict)
    expected_output = AIMessage(content="foo")
    assert result == expected_output


def test__convert_dict_to_message_other_role() -> None:
    message_dict = {"role": "system", "content": "foo"}
    result = convert_dict_to_message(message_dict)
    expected_output = SystemMessage(content="foo")
    assert result == expected_output


def test__convert_dict_to_message_function_call() -> None:
    raw_function_calls = [
        {
            "function": {
                "name": "get_current_weather",
                "arguments": '{"location": "Boston", "unit": "fahrenheit"}',
            },
            "type": "function",
        }
    ]
    message_dict = {
        "role": "assistant",
        "content": "foo",
        "tool_calls": raw_function_calls,
    }
    result = convert_dict_to_message(message_dict)

    tool_calls = [
        parse_tool_call(raw_tool_call, return_id=True)
        for raw_tool_call in raw_function_calls
    ]
    expected_output = AIMessage(
        content="foo",
        additional_kwargs={"tool_calls": raw_function_calls},
        tool_calls=tool_calls,
        invalid_tool_calls=[],
    )
    assert result == expected_output


def test__convert_message_to_dict_human() -> None:
    message = HumanMessage(content="foo")
    result = convert_message_to_dict(message)
    expected_output = {"role": "user", "content": "foo"}
    assert result == expected_output


def test__convert_message_to_dict_ai() -> None:
    message = AIMessage(content="foo")
    result = convert_message_to_dict(message)
    expected_output = {"role": "assistant", "content": "foo"}
    assert result == expected_output


def test__convert_message_to_dict_system() -> None:
    message = SystemMessage(content="foo")
    result = convert_message_to_dict(message)
    expected_output = {"role": "system", "content": "foo"}
    assert result == expected_output

import json

import pytest

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.outputs import ChatGeneration


def _build_generation_with_arguments(args_str: str) -> list[ChatGeneration]:
    message = AIMessage(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "call_func",
                    "type": "function",
                    "function": {"name": "func", "arguments": args_str},
                }
            ]
        },
    )
    return [ChatGeneration(message=message)]


def test_tool_call_parses_normal_json() -> None:
    # Arrange: normal JSON string in tool call arguments
    args_str = json.dumps({"a": 1})
    result = _build_generation_with_arguments(args_str)
    parser = JsonOutputKeyToolsParser(key_name="func", first_tool_only=True)

    # Act
    output = parser.parse_result(result)  # type: ignore[arg-type]

    # Assert
    assert output == {"a": 1}


def test_tool_call_parses_fenced_json() -> None:
    # Arrange: fenced JSON should be accepted after the fix
    fenced_args = """```json\n{"a": 1}\n```"""
    result = _build_generation_with_arguments(fenced_args)
    parser = JsonOutputKeyToolsParser(key_name="func", first_tool_only=True)

    # Act
    output = parser.parse_result(result)  # type: ignore[arg-type]

    # Assert
    assert output == {"a": 1}


def test_tool_call_incomplete_json_raises() -> None:
    # Arrange: incomplete JSON should raise before and after the change
    incomplete_args = '{"a": 1'  # missing closing brace
    result = _build_generation_with_arguments(incomplete_args)
    parser = JsonOutputKeyToolsParser(key_name="func", first_tool_only=True)

    # Act + Assert
    with pytest.raises(OutputParserException):
        _ = parser.parse_result(result)  # type: ignore[arg-type]

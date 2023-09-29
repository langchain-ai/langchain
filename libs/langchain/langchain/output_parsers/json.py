from __future__ import annotations

import json
import re
from json import JSONDecodeError
from typing import Any, Callable, List, Optional

import jsonpatch

from langchain.schema import BaseOutputParser, OutputParserException
from langchain.schema.output import ChatGeneration, Generation
from langchain.schema.output_parser import BaseCumulativeTransformOutputParser


def _replace_new_line(match: re.Match[str]) -> str:
    value = match.group(2)
    value = re.sub(r"\n", r"\\n", value)
    value = re.sub(r"\r", r"\\r", value)
    value = re.sub(r"\t", r"\\t", value)
    value = re.sub(r'(?<!\\)"', r"\"", value)

    return match.group(1) + value + match.group(3)


def _custom_parser(multiline_string: str) -> str:
    """
    The LLM response for `action_input` may be a multiline
    string containing unescaped newlines, tabs or quotes. This function
    replaces those characters with their escaped counterparts.
    (newlines in JSON must be double-escaped: `\\n`)
    """
    if isinstance(multiline_string, (bytes, bytearray)):
        multiline_string = multiline_string.decode()

    multiline_string = re.sub(
        r'("action_input"\:\s*")(.*)(")',
        _replace_new_line,
        multiline_string,
        flags=re.DOTALL,
    )

    return multiline_string


# Adapted from https://github.com/KillianLucas/open-interpreter/blob/main/interpreter/utils/parse_partial_json.py
# MIT License
def parse_partial_json(s: str) -> Any:
    # Attempt to parse the string as-is.
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Initialize variables.
    new_s = ""
    stack = []
    is_inside_string = False
    escaped = False

    # Process each character in the string one at a time.
    for char in s:
        if is_inside_string:
            if char == '"' and not escaped:
                is_inside_string = False
            elif char == "\n" and not escaped:
                char = "\\n"  # Replace the newline character with the escape sequence.
            elif char == "\\":
                escaped = not escaped
            else:
                escaped = False
        else:
            if char == '"':
                is_inside_string = True
                escaped = False
            elif char == "{":
                stack.append("}")
            elif char == "[":
                stack.append("]")
            elif char == "}" or char == "]":
                if stack and stack[-1] == char:
                    stack.pop()
                else:
                    # Mismatched closing character; the input is malformed.
                    return None

        # Append the processed character to the new string.
        new_s += char

    # If we're still inside a string at the end of processing,
    # we need to close the string.
    if is_inside_string:
        new_s += '"'

    # Close any remaining open structures in the reverse order that they were opened.
    for closing_char in reversed(stack):
        new_s += closing_char

    # Attempt to parse the modified string as JSON.
    try:
        return json.loads(new_s)
    except json.JSONDecodeError:
        # If we still can't parse the string as JSON, return None to indicate failure.
        return None


def parse_json_markdown(
    json_string: str, parser: Callable[[str], Any] = json.loads
) -> dict:
    """
    Parse a JSON string from a Markdown string.

    Args:
        json_string: The Markdown string.

    Returns:
        The parsed JSON object as a Python dictionary.
    """
    # Try to find JSON string within triple backticks
    match = re.search(r"```(json)?(.*)```", json_string, re.DOTALL)

    # If no match found, assume the entire string is a JSON string
    if match is None:
        json_str = json_string
    else:
        # If match found, use the content within the backticks
        json_str = match.group(2)

    # Strip whitespace and newlines from the start and end
    json_str = json_str.strip()

    # handle newlines and other special characters inside the returned value
    json_str = _custom_parser(json_str)

    # Parse the JSON string into a Python dictionary
    parsed = parser(json_str)

    return parsed


def parse_and_check_json_markdown(text: str, expected_keys: List[str]) -> dict:
    """
    Parse a JSON string from a Markdown string and check that it
    contains the expected keys.

    Args:
        text: The Markdown string.
        expected_keys: The expected keys in the JSON string.

    Returns:
        The parsed JSON object as a Python dictionary.
    """
    try:
        json_obj = parse_json_markdown(text)
    except json.JSONDecodeError as e:
        raise OutputParserException(f"Got invalid JSON object. Error: {e}")
    for key in expected_keys:
        if key not in json_obj:
            raise OutputParserException(
                f"Got invalid return object. Expected key `{key}` "
                f"to be present, but got {json_obj}"
            )
    return json_obj


class SimpleJsonOutputParser(BaseOutputParser[Any]):
    """Parse the output of an LLM call to a JSON object."""

    def parse(self, text: str) -> Any:
        text = text.strip()
        try:
            return parse_partial_json(text)
        except JSONDecodeError as e:
            raise OutputParserException(f"Invalid json output: {text}") from e

    @property
    def _type(self) -> str:
        return "simple_json_output_parser"


class PartialFunctionsJsonOutputParser(BaseCumulativeTransformOutputParser[Any]):
    @property
    def _type(self) -> str:
        return "partial_functions_json"

    def parse_result(self, result: List[Generation]) -> Any:
        if len(result) != 1:
            raise OutputParserException(
                f"Expected exactly one result, but got {len(result)}"
            )
        generation = result[0]
        if not isinstance(generation, ChatGeneration):
            raise OutputParserException(
                "This output parser can only be used with a chat generation."
            )
        message = generation.message
        try:
            function_call = message.additional_kwargs["function_call"]
        except KeyError:
            return None
        try:
            return parse_partial_json(function_call["arguments"])
        except KeyError:
            return None

    def _diff(self, prev: Optional[Any], next: Any) -> Any:
        return jsonpatch.make_patch(prev, next).patch

    # This method would be called by the default implementation of `parse_result`
    # but we're overriding that method so it's not needed.
    def parse(self, text: str) -> Any:
        raise NotImplementedError()


class PartialJsonOutputParser(BaseCumulativeTransformOutputParser[Any]):
    @property
    def _type(self) -> str:
        return "partial_functions_json"

    def _diff(self, prev: Optional[Any], next: Any) -> Any:
        return jsonpatch.make_patch(prev, next).patch

    def parse(self, text: str) -> Any:
        return parse_json_markdown(text, parse_partial_json)

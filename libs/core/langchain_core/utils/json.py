from __future__ import annotations

import json
import re
from typing import Any, Callable, List

from langchain_core.exceptions import OutputParserException


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
        r'("action_input"\:\s*")(.*?)(")',
        _replace_new_line,
        multiline_string,
        flags=re.DOTALL,
    )

    return multiline_string


# Adapted from https://github.com/KillianLucas/open-interpreter/blob/5b6080fae1f8c68938a1e4fa8667e3744084ee21/interpreter/utils/parse_partial_json.py
# MIT License


def parse_partial_json(s: str, *, strict: bool = False) -> Any:
    """Parse a JSON string that may be missing closing braces.

    Args:
        s: The JSON string to parse.
        strict: Whether to use strict parsing. Defaults to False.

    Returns:
        The parsed JSON object as a Python dictionary.
    """
    # Attempt to parse the string as-is.
    try:
        return json.loads(s, strict=strict)
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

    # Try to parse mods of string until we succeed or run out of characters.
    while new_s:
        final_s = new_s

        # Close any remaining open structures in the reverse
        # order that they were opened.
        for closing_char in reversed(stack):
            final_s += closing_char

        # Attempt to parse the modified string as JSON.
        try:
            return json.loads(final_s, strict=strict)
        except json.JSONDecodeError:
            # If we still can't parse the string as JSON,
            # try removing the last character
            new_s = new_s[:-1]

    # If we got here, we ran out of characters to remove
    # and still couldn't parse the string as JSON, so return the parse error
    # for the original string.
    return json.loads(s, strict=strict)


def parse_json_markdown(
    json_string: str, *, parser: Callable[[str], Any] = parse_partial_json
) -> dict:
    """
    Parse a JSON string from a Markdown string.

    Args:
        json_string: The Markdown string.

    Returns:
        The parsed JSON object as a Python dictionary.
    """
    try:
        return _parse_json(json_string, parser=parser)
    except json.JSONDecodeError:
        # Try to find JSON string within triple backticks
        match = re.search(r"```(json)?(.*)", json_string, re.DOTALL)

        # If no match found, assume the entire string is a JSON string
        if match is None:
            json_str = json_string
        else:
            # If match found, use the content within the backticks
            json_str = match.group(2)
    return _parse_json(json_str, parser=parser)


def _parse_json(
    json_str: str, *, parser: Callable[[str], Any] = parse_partial_json
) -> dict:
    # Strip whitespace and newlines from the start and end
    json_str = json_str.strip().strip("`")

    # handle newlines and other special characters inside the returned value
    json_str = _custom_parser(json_str)

    # Parse the JSON string into a Python dictionary
    return parser(json_str)


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

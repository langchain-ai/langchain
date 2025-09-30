"""Utilities for JSON."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Union

from langchain_core.exceptions import OutputParserException


def _replace_new_line(match: re.Match[str]) -> str:
    value = match.group(2)
    value = re.sub(r"\n", r"\\n", value)
    value = re.sub(r"\r", r"\\r", value)
    value = re.sub(r"\t", r"\\t", value)
    value = re.sub(r'(?<!\\)"', r"\"", value)

    return match.group(1) + value + match.group(3)


def _custom_parser(multiline_string: Union[str, bytes, bytearray]) -> str:
    r"""Custom parser for multiline strings.

    The LLM response for `action_input` may be a multiline
    string containing unescaped newlines, tabs or quotes. This function
    replaces those characters with their escaped counterparts.
    (newlines in JSON must be double-escaped: `\\n`).

    Returns:
        The modified string with escaped newlines, tabs and quotes.
    """
    if isinstance(multiline_string, (bytes, bytearray)):
        multiline_string = multiline_string.decode()

    return re.sub(
        r'("action_input"\:\s*")(.*?)(")',
        _replace_new_line,
        multiline_string,
        flags=re.DOTALL,
    )


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
    new_chars = []
    stack = []
    is_inside_string = False
    escaped = False

    # Process each character in the string one at a time.
    for char in s:
        new_char = char
        if is_inside_string:
            if char == '"' and not escaped:
                is_inside_string = False
            elif char == "\n" and not escaped:
                new_char = (
                    "\\n"  # Replace the newline character with the escape sequence.
                )
            elif char == "\\":
                escaped = not escaped
            else:
                escaped = False
        elif char == '"':
            is_inside_string = True
            escaped = False
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in {"}", "]"}:
            if stack and stack[-1] == char:
                stack.pop()
            else:
                # Mismatched closing character; the input is malformed.
                return None

        # Append the processed character to the new string.
        new_chars.append(new_char)

    # If we're still inside a string at the end of processing,
    # we need to close the string.
    if is_inside_string:
        if escaped:  # Remove unterminated escape character
            new_chars.pop()
        new_chars.append('"')

    # Reverse the stack to get the closing characters.
    stack.reverse()

    # Try to parse mods of string until we succeed or run out of characters.
    while new_chars:
        # Close any remaining open structures in the reverse
        # order that they were opened.
        # Attempt to parse the modified string as JSON.
        try:
            return json.loads("".join(new_chars + stack), strict=strict)
        except json.JSONDecodeError:
            # If we still can't parse the string as JSON,
            # try removing the last character
            new_chars.pop()

    # If we got here, we ran out of characters to remove
    # and still couldn't parse the string as JSON, so return the parse error
    # for the original string.
    return json.loads(s, strict=strict)


_json_markdown_re = re.compile(r"```(json)?(.*)", re.DOTALL)


def parse_json_markdown(
    json_string: str, *, parser: Callable[[str], Any] = parse_partial_json
) -> dict:
    """Parse a JSON string from a Markdown string.

    Args:
        json_string: The Markdown string.
        parser: The parser to use. Defaults to `parse_partial_json`.

    Returns:
        The parsed JSON object as a Python dictionary.
    """
    try:
        return _parse_json(json_string, parser=parser)
    except json.JSONDecodeError:
        # Try to find JSON string within triple backticks
        match = _json_markdown_re.search(json_string)

        # If no match found, assume the entire string is a JSON string
        # Else, use the content within the backticks
        json_str = json_string if match is None else match.group(2)
    return _parse_json(json_str, parser=parser)


_json_strip_chars = " \n\r\t`"


def _parse_json(
    json_str: str, *, parser: Callable[[str], Any] = parse_partial_json
) -> dict:
    # Strip whitespace,newlines,backtick from the start and end
    json_str = json_str.strip(_json_strip_chars)

    # handle newlines and other special characters inside the returned value
    json_str = _custom_parser(json_str)

    # Parse the JSON string into a Python dictionary
    return parser(json_str)


def parse_and_check_json_markdown(text: str, expected_keys: list[str]) -> dict:
    """Parse and check a JSON string from a Markdown string.

    Checks that it contains the expected keys.

    Args:
        text: The Markdown string.
        expected_keys: The expected keys in the JSON string.

    Returns:
        The parsed JSON object as a Python dictionary.

    Raises:
        OutputParserException: If the JSON string is invalid or does not contain
            the expected keys.
    """
    try:
        json_obj = parse_json_markdown(text)
    except json.JSONDecodeError as e:
        msg = f"Got invalid JSON object. Error: {e}"
        raise OutputParserException(msg) from e
    if not isinstance(json_obj, dict):
        error_message = (
            f"Expected JSON object (dict), but got: {type(json_obj).__name__}. "
        )
        raise OutputParserException(error_message, llm_output=text)

    for key in expected_keys:
        if key not in json_obj:
            msg = (
                f"Got invalid return object. Expected key `{key}` "
                f"to be present, but got {json_obj}"
            )
            raise OutputParserException(msg)
    return json_obj

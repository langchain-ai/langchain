from __future__ import annotations

import json
import re
from json import JSONDecodeError
from typing import Any, Callable, List, Optional

import jsonpatch

from langchain.schema.output_parser import (
    BaseCumulativeTransformOutputParser,
    OutputParserException,
)

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
def parse_partial_json(s: str, *, strict: bool = False) -> Any:
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

    # Close any remaining open structures in the reverse order that they were opened.
    for closing_char in reversed(stack):
        new_s += closing_char

    # Attempt to parse the modified string as JSON.
    try:
        return json.loads(new_s, strict=strict)
    except json.JSONDecodeError:
        # If we still can't parse the string as JSON, return None to indicate failure.
        return None


def parse_json_markdown(
    json_string: str, *, parser: Callable[[str], Any] = json.loads
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

def fix_code_in_json(text: str) -> str:
    """Fixes nested code block in json markdown"""
    # Extract the code block and replace it with a placeholder
    pattern = r"```([^`]*?)```"
    match = re.search(pattern, text)
    if match:
        code_block = match.group(1)
        text = re.sub(pattern, "CODE_BLOCK_PLACEHOLDER", text, count=1)

        # Escape the special characters in the code block
        escaped_code_block = (
            code_block.replace("\n", "\\n").replace("\t", "\\t").replace('"', '\\"')
        )

        # Add backtick pairs to escaped code block
        escaped_code_block = "[BEGIN_CODE]" + escaped_code_block + "[END_CODE]"

        # Replace the placeholder in the original text with the escaped code block
        text = text.replace("CODE_BLOCK_PLACEHOLDER", escaped_code_block)

    return text


def fix_json_with_embedded_code_block(text: str, max_loop: int = 20) -> dict:
    """Try to fix json with embedded code block.

    Args:
        text: JSON string with embedded code block
        max_loop: Maximum number of loops to try fixing the JSON string
    """
    loop = 0
    while True:
        if loop > max_loop:
            raise ValueError("Max loop reached")
        try:
            text = fix_code_in_json(text)
            json.loads(text)
            break
        except json.JSONDecodeError as e:
            if text[e.pos] == "\n":
                text = text[: e.pos] + "\\n" + text[e.pos + 1 :]
                text = text.replace("[BEGIN_CODE]", "```")
            else:
                raise
        finally:
            loop += 1
    final_text = text.replace("[END_CODE]", "```")
    return json.loads(final_text)

class SimpleJsonOutputParser(BaseCumulativeTransformOutputParser[Any]):
    """Parse the output of an LLM call to a JSON object.

    When used in streaming mode, it will yield partial JSON objects containing
    all the keys that have been returned so far.

    In streaming, if `diff` is set to `True`, yields JSONPatch operations
    describing the difference between the previous and the current object.
    """

    def _diff(self, prev: Optional[Any], next: Any) -> Any:
        return jsonpatch.make_patch(prev, next).patch

    def parse(self, text: str) -> Any:
        text = text.strip()
        try:
            return parse_json_markdown(text.strip(), parser=parse_partial_json)
        except JSONDecodeError as e:
            raise OutputParserException(f"Invalid json output: {text}") from e

    @property
    def _type(self) -> str:
        return "simple_json_output_parser"

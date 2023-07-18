from __future__ import annotations

import json
import re
from json import JSONDecodeError
from typing import Any, List

from langchain.schema import BaseOutputParser, OutputParserException


def parse_json_markdown(json_string: str) -> dict:
    """
    Parse a JSON string from a Markdown string.

    Args:
        json_string: The Markdown string.

    Returns:
        The parsed JSON object as a Python dictionary.
    """
    # Try to find JSON string within triple backticks
    match = re.search(r"```(json)?(.*?)```", json_string, re.DOTALL)

    # If no match found, assume the entire string is a JSON string
    if match is None:
        json_str = json_string
    else:
        # If match found, use the content within the backticks
        json_str = match.group(2)

    # Strip whitespace and newlines from the start and end
    json_str = json_str.strip()

    # Parse the JSON string into a Python dictionary
    parsed = json.loads(json_str)

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
            return json.loads(text)
        except JSONDecodeError as e:
            raise OutputParserException(f"Invalid json output: {text}") from e

    @property
    def _type(self) -> str:
        return "simple_json_output_parser"

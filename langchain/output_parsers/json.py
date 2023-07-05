from __future__ import annotations

import json
import re
<<<<<<< HEAD
from json import JSONDecodeError
from typing import Any, List
=======
from typing import List, Union
>>>>>>> 951522c6 (Update JSON parser to work w/ multiple JSON objects in markdown)

from langchain.schema import BaseOutputParser, OutputParserException


def parse_json_markdown(json_string: str) -> Union[dict, List[dict]]:
    """
    Parse JSON objects from a Markdown string.

    Args:
        json_string: The Markdown string.

    Returns:
        The parsed JSON objects as a Python dictionary or a list of dictionaries.
    """
    # Find all JSON strings within triple backticks
    matches = re.findall(r"```json\s*({.*?})\s*```", json_string, re.DOTALL)

    # Parse each JSON string into a Python dictionary
    parsed_objects = [json.loads(match) for match in matches]

    # If there's only one JSON object, return it as a dictionary; else, return a list of dictionaries
    return parsed_objects[0] if len(parsed_objects) == 1 else parsed_objects

 
def parse_and_check_json_markdown(
    text: str, expected_keys: List[str]
) -> Union[dict, List[dict]]:
    """
    Parse JSON objects from a Markdown string and check that they
    contain the expected keys.

    Args:
        text: The Markdown string.
        expected_keys: The expected keys in the JSON objects.

    Returns:
        The parsed JSON object(s) as a Python dictionary or a list of dictionaries.
    """
    try:
        json_objects = parse_json_markdown(text)
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
    def parse(self, text: str) -> Any:
        text = text.strip()
        try:
            return json.loads(text)
        except JSONDecodeError as e:
            raise OutputParserException(f"Invalid json output: {text}") from e

    @property
    def _type(self) -> str:
        return "simple_json_output_parser"

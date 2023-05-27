from __future__ import annotations

import json
import re
from typing import List

from langchain.schema import OutputParserException


def parse_json_markdown(json_string: str) -> dict:
    # Remove all characters that are not { or [ from start of string
    json_string = re.sub("^[^\[{]*", "", json_string)

    # Remove all characters that are not } or ] from end of string
    json_string = re.sub(r"[^}\]]*$", "", json_string)

    # Parse the JSON string into a Python dictionary
    parsed = json.loads(json_string)

    return parsed


def parse_and_check_json_markdown(text: str, expected_keys: List[str]) -> dict:
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

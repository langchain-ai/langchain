from __future__ import annotations

import json
import re
from typing import List, Union

from langchain.schema import OutputParserException


def parse_json_markdown(json_string: str) -> Union[dict, List[dict]]:
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


def parse_and_check_json_markdown(
    text: str,
    expected_keys: List[str]
) -> Union[dict, List[dict]]:
    try:
        json_obj = parse_json_markdown(text)
    except json.JSONDecodeError as e:
        raise OutputParserException(f"Got invalid JSON object. Error: {e}")
    if isinstance(json_obj, list):
      expected_keys_set = set(expected_keys)
      missing_keys = []

      for record in json_obj:
            missing = expected_keys_set - set(record.keys())
            missing_keys.extend(missing)
      
      if missing_keys:
        raise OutputParserException(
            f"Got invalid return object. Expected key(s) `{missing_keys}` "
            f"to be present, but got {record}"
        )
    else:
        for key in expected_keys:
            if key not in json_obj:
                raise OutputParserException(
                    f"Got invalid return object. Expected key `{key}` "
                    f"to be present, but got {json_obj}"
                )
    return json_obj

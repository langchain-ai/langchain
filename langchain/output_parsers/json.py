import json
import re
from typing import Dict

from langchain.schema import OutputParserException


def parse_json(text: str) -> Dict:
    pattern = r"```(?:json)?\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        json_string = match.group(1)
        return json.loads(json_string)
    else:
        raise OutputParserException(f"Could not parse json: {text}")

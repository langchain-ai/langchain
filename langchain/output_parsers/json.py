import json


def parse_json_markdown(json_string: str) -> dict:
    # Remove the triple backticks if present
    json_string = json_string.replace("```json", "").replace("```", "")

    # Strip whitespace and newlines from the start and end
    json_string = json_string.strip()

    # Parse the JSON string into a Python dictionary
    parsed = json.loads(json_string)

    return parsed

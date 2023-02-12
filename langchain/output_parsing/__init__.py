import json
from typing import Any


def parse_json(text: str) -> Any:
    """Parse json string."""
    return json.loads(text)


def validate_json(text: str) -> None:
    """Validate string can be parsed as json."""
    try:
        parse_json(text)
    except Exception as e:
        raise ValueError("Text not parsable as json.") from e

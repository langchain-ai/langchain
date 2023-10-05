"""Strategies used with MultiStrategyParser parsers."""
import json

from langchain.output_parsers.json import (
    fix_json_with_embedded_code_block,
    parse_json_markdown,
)
from langchain.output_parsers.multi_strategy.base import ParseStrategy


def is_bare_json(text: str) -> dict:
    """Tries to load as bare json"""
    return json.loads(text.strip())


def json_markdown(text: str) -> dict:
    """Extract a json object from markdown markup"""
    return parse_json_markdown(text)


def fallback(text: str) -> dict:
    """Example fallback strategy."""
    return {"action": "Final Answer", "action_input": text}


# The order of the strategies is important
# They are tried in order and the first one that matches is used
json_react_strategies = (
    ParseStrategy(is_bare_json, lambda text: text.startswith("{"), name="bare_json"),
    ParseStrategy(json_markdown, lambda text: text.find("```") != -1),
    ParseStrategy(
        fix_json_with_embedded_code_block,
        lambda text: text.find("```") != -1,
        name="fix_embedded_code_block",
    ),
    # this is where a fallback would go
    # ParseStrategy(fallback, lambda _: True),
)

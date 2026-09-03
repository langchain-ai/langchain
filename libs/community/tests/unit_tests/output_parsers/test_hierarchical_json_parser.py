"""Unit tests for HierarchicalRobustJsonParser."""

import pytest
from pydantic import BaseModel

from langchain_community.output_parsers.hierarchical_json_parser import (
    HierarchicalRobustJsonParser,
)


class UserProfile(BaseModel):
    name: str
    age: int


def test_parse_clean_json() -> None:
    parser = HierarchicalRobustJsonParser()
    text = '{"name": "Alice", "role": "Engineer"}'
    result = parser.parse(text)

    assert result == {"name": "Alice", "role": "Engineer"}


def test_parse_markdown_code_block() -> None:
    parser = HierarchicalRobustJsonParser()
    text = "Here is the response:\n```json\n{\n  \"status\": \"success\",\n  \"code\": 200\n}\n```"
    result = parser.parse(text)

    assert result == {"status": "success", "code": 200}


def test_parse_single_quotes_and_trailing_comma() -> None:
    parser = HierarchicalRobustJsonParser()
    text = "{'title': 'LangChain', 'stars': 90000,}"
    result = parser.parse(text)

    assert result == {"title": "LangChain", "stars": 90000}


def test_parse_with_pydantic_validation() -> None:
    parser = HierarchicalRobustJsonParser(pydantic_object=UserProfile)
    text = '```json\n{"name": "Bob", "age": 25}\n```'
    result = parser.parse(text)

    assert isinstance(result, UserProfile)
    assert result.name == "Bob"
    assert result.age == 25


def test_parse_unclosed_brackets() -> None:
    parser = HierarchicalRobustJsonParser()
    text = '{"data": [1, 2, 3'
    result = parser.parse(text)

    assert result == {"data": [1, 2, 3]}

"""Test allowed_values feature for query constructor."""

import json
import re

from langchain_classic.chains.query_constructor.base import (
    _format_attribute_info,
    get_query_constructor_prompt,
)
from langchain_classic.chains.query_constructor.schema import AttributeInfo


class TestFormatAttributeInfoWithAllowedValues:
    """Test _format_attribute_info with allowed_values parameter."""

    def test_format_without_allowed_values(self) -> None:
        """Test formatting without allowed_values returns standard output."""
        info = [
            AttributeInfo(name="genre", type="string", description="The movie genre"),
        ]
        result = _format_attribute_info(info)
        # Unescape the curly braces for JSON parsing
        parsed = json.loads(result.replace("{{", "{").replace("}}", "}"))

        assert "genre" in parsed
        assert parsed["genre"]["type"] == "string"
        assert parsed["genre"]["description"] == "The movie genre"
        assert "allowed_values" not in parsed["genre"]

    def test_format_with_allowed_values(self) -> None:
        """Test formatting with allowed_values includes values in output."""
        info = [
            AttributeInfo(name="genre", type="string", description="The movie genre"),
            AttributeInfo(name="year", type="integer", description="Release year"),
        ]
        allowed_values = {
            "genre": ["action", "comedy", "drama", "horror"],
        }
        result = _format_attribute_info(info, allowed_values=allowed_values)
        parsed = json.loads(result.replace("{{", "{").replace("}}", "}"))

        # Genre should have allowed_values
        assert "genre" in parsed
        expected_values = ["action", "comedy", "drama", "horror"]
        assert parsed["genre"]["allowed_values"] == expected_values

        # Year should not have allowed_values (not specified)
        assert "year" in parsed
        assert "allowed_values" not in parsed["year"]

    def test_format_with_multiple_allowed_values(self) -> None:
        """Test formatting with allowed_values for multiple fields."""
        info = [
            AttributeInfo(name="genre", type="string", description="The movie genre"),
            AttributeInfo(name="rating", type="string", description="Movie rating"),
        ]
        allowed_values = {
            "genre": ["action", "comedy"],
            "rating": ["G", "PG", "PG-13", "R"],
        }
        result = _format_attribute_info(info, allowed_values=allowed_values)
        parsed = json.loads(result.replace("{{", "{").replace("}}", "}"))

        assert parsed["genre"]["allowed_values"] == ["action", "comedy"]
        assert parsed["rating"]["allowed_values"] == ["G", "PG", "PG-13", "R"]

    def test_format_with_dict_input(self) -> None:
        """Test formatting with dict-based attribute info."""
        info = [
            {"name": "genre", "type": "string", "description": "The movie genre"},
        ]
        allowed_values = {"genre": ["action", "comedy"]}
        result = _format_attribute_info(info, allowed_values=allowed_values)
        parsed = json.loads(result.replace("{{", "{").replace("}}", "}"))

        assert parsed["genre"]["allowed_values"] == ["action", "comedy"]

    def test_format_ignores_unknown_fields_in_allowed_values(self) -> None:
        """Test that allowed_values for non-existent fields are ignored."""
        info = [
            AttributeInfo(name="genre", type="string", description="The movie genre"),
        ]
        allowed_values = {
            "genre": ["action"],
            "nonexistent": ["value"],  # This field doesn't exist in info
        }
        result = _format_attribute_info(info, allowed_values=allowed_values)
        parsed = json.loads(result.replace("{{", "{").replace("}}", "}"))

        assert "genre" in parsed
        assert parsed["genre"]["allowed_values"] == ["action"]
        # nonexistent field should not appear since it's not in info
        assert "nonexistent" not in parsed


class TestGetQueryConstructorPromptWithAllowedValues:
    """Test get_query_constructor_prompt with allowed_values parameter."""

    def test_prompt_includes_allowed_values_in_attributes(self) -> None:
        """Test that the prompt includes allowed_values in attribute description."""
        attribute_info = [
            AttributeInfo(name="genre", type="string", description="The movie genre"),
        ]
        allowed_values = {"genre": ["action", "comedy", "drama"]}

        prompt = get_query_constructor_prompt(
            document_contents="Movie descriptions",
            attribute_info=attribute_info,
            allowed_values=allowed_values,
        )

        # Get the formatted prompt
        formatted = prompt.format(query="test query")

        # Check that allowed_values appear in the prompt
        assert "action" in formatted
        assert "comedy" in formatted
        assert "drama" in formatted
        assert "allowed_values" in formatted

    def test_prompt_without_allowed_values(self) -> None:
        """Test that prompt works without allowed_values in attribute info."""
        attribute_info = [
            AttributeInfo(name="genre", type="string", description="The movie genre"),
        ]

        prompt = get_query_constructor_prompt(
            document_contents="Movie descriptions",
            attribute_info=attribute_info,
        )

        formatted = prompt.format(query="test query")

        # The schema instruction about allowed_values is always present,
        # but the attribute info should not contain an "allowed_values" array
        # Check that the genre attribute doesn't have allowed_values in its JSON
        assert '"genre": {' in formatted or "'genre': {" in formatted
        # The attribute info for genre should not include allowed_values list
        # We verify by checking that there's no list pattern after genre

        # Find the genre attribute block in the formatted prompt
        genre_block_match = re.search(
            r'"genre":\s*\{[^}]+\}',
            formatted,
        )
        if genre_block_match:
            genre_block = genre_block_match.group()
            # The genre block should NOT contain allowed_values array
            assert '"allowed_values"' not in genre_block


class TestPromptInstructionsForAllowedValues:
    """Test that prompt schema includes instructions about allowed_values."""

    def test_schema_mentions_allowed_values_instruction(self) -> None:
        """Test that schema prompt includes instruction about using allowed_values."""
        from langchain_classic.chains.query_constructor.prompt import DEFAULT_SCHEMA

        assert "allowed_values" in DEFAULT_SCHEMA
        assert "only use values from that list" in DEFAULT_SCHEMA

    def test_schema_with_limit_mentions_allowed_values_instruction(self) -> None:
        """Test that schema with limit prompt includes allowed_values instruction."""
        from langchain_classic.chains.query_constructor.prompt import SCHEMA_WITH_LIMIT

        assert "allowed_values" in SCHEMA_WITH_LIMIT
        assert "only use values from that list" in SCHEMA_WITH_LIMIT

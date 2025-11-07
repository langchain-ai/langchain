"""Unit tests for output parsers."""

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.outputs import Generation
from pydantic import BaseModel, Field

from langchain_perplexity.output_parsers import (
    ReasoningJsonOutputParser,
    ReasoningStructuredOutputParser,
    strip_think_tags,
)


class TestStripThinkTags:
    """Tests for the strip_think_tags function."""

    def test_strip_simple_think_tags(self) -> None:
        """Test stripping simple think tags."""
        text = "Hello <think>some reasoning</think> world"
        result = strip_think_tags(text)
        assert result == "Hello  world"

    def test_strip_multiple_think_tags(self) -> None:
        """Test stripping multiple think tags."""
        text = "<think>first</think> Hello <think>second</think> world\
            <think>third</think>"
        result = strip_think_tags(text)
        assert result == "Hello  world"

    def test_strip_nested_like_think_tags(self) -> None:
        """Test stripping think tags that might appear nested."""
        text = "<think>outer <think>inner</think> still outer</think> result"
        result = strip_think_tags(text)
        # The function removes from first <think> to first </think>
        # then continues from after that </think>
        assert result == "still outer</think> result"

    def test_strip_think_tags_no_closing_tag(self) -> None:
        """Test handling of think tags without closing tag."""
        text = "Hello <think>unclosed reasoning world"
        result = strip_think_tags(text)
        # Treats unclosed tag as literal text
        assert result == "Hello <think>unclosed reasoning world"

    def test_strip_think_tags_empty_content(self) -> None:
        """Test stripping empty think tags."""
        text = "Hello <think></think> world"
        result = strip_think_tags(text)
        assert result == "Hello  world"

    def test_strip_think_tags_no_tags(self) -> None:
        """Test text without any think tags."""
        text = "Hello world"
        result = strip_think_tags(text)
        assert result == "Hello world"

    def test_strip_think_tags_only_tags(self) -> None:
        """Test text containing only think tags."""
        text = "<think>reasoning</think>"
        result = strip_think_tags(text)
        assert result == ""

    def test_strip_think_tags_multiline(self) -> None:
        """Test stripping think tags across multiple lines."""
        text = """Hello
<think>
reasoning line 1
reasoning line 2
</think>
world"""
        result = strip_think_tags(text)
        assert result == "Hello\n\nworld"

    def test_strip_think_tags_with_special_chars(self) -> None:
        """Test think tags containing special characters."""
        text = 'Before <think>{"key": "value"}</think> After'
        result = strip_think_tags(text)
        assert result == "Before  After"


class TestReasoningJsonOutputParser:
    """Tests for ReasoningJsonOutputParser."""

    def test_parse_json_without_think_tags(self) -> None:
        """Test parsing JSON without think tags."""
        parser = ReasoningJsonOutputParser()
        text = '{"name": "John", "age": 30}'
        generation = Generation(text=text)
        result = parser.parse_result([generation])
        assert result == {"name": "John", "age": 30}

    def test_parse_json_with_think_tags(self) -> None:
        """Test parsing JSON with think tags."""
        parser = ReasoningJsonOutputParser()
        text = '<think>Let me construct the JSON</think>{"name": "John", "age": 30}'
        generation = Generation(text=text)
        result = parser.parse_result([generation])
        assert result == {"name": "John", "age": 30}

    def test_parse_json_with_multiple_think_tags(self) -> None:
        """Test parsing JSON with multiple think tags."""
        parser = ReasoningJsonOutputParser()
        text = '<think>Step 1</think>{"name": <think>thinking</think>"John", "age": 30}'
        generation = Generation(text=text)
        result = parser.parse_result([generation])
        assert result == {"name": "John", "age": 30}

    def test_parse_markdown_json_with_think_tags(self) -> None:
        """Test parsing markdown-wrapped JSON with think tags."""
        parser = ReasoningJsonOutputParser()
        text = """<think>Building response</think>
```json
{"name": "John", "age": 30}
```"""
        generation = Generation(text=text)
        result = parser.parse_result([generation])
        assert result == {"name": "John", "age": 30}

    def test_parse_complex_json_with_think_tags(self) -> None:
        """Test parsing complex nested JSON with think tags."""
        parser = ReasoningJsonOutputParser()
        text = """<think>Creating nested structure</think>
{
    "user": {
        "name": "John",
        "address": {
            "city": "NYC",
            "zip": "10001"
        }
    },
    "items": [1, 2, 3]
}"""
        generation = Generation(text=text)
        result = parser.parse_result([generation])
        assert result == {
            "user": {"name": "John", "address": {"city": "NYC", "zip": "10001"}},
            "items": [1, 2, 3],
        }

    def test_parse_invalid_json_with_think_tags(self) -> None:
        """Test that invalid JSON raises an exception even with think tags."""
        parser = ReasoningJsonOutputParser()
        text = "<think>This will fail</think>{invalid json}"
        generation = Generation(text=text)
        with pytest.raises(OutputParserException):
            parser.parse_result([generation])

    def test_parse_empty_string_after_stripping(self) -> None:
        """Test parsing when only think tags remain."""
        parser = ReasoningJsonOutputParser()
        text = "<think>Only reasoning, no output</think>"
        generation = Generation(text=text)
        with pytest.raises(OutputParserException):
            parser.parse_result([generation])

    def test_parse_json_array_with_think_tags(self) -> None:
        """Test parsing JSON array with think tags."""
        parser = ReasoningJsonOutputParser()
        text = '<think>Creating array</think>[{"id": 1}, {"id": 2}]'
        generation = Generation(text=text)
        result = parser.parse_result([generation])
        assert result == [{"id": 1}, {"id": 2}]

    def test_partial_json_parsing_with_think_tags(self) -> None:
        """Test partial JSON parsing with think tags."""
        parser = ReasoningJsonOutputParser()
        text = '<think>Starting</think>{"name": "John", "age":'
        generation = Generation(text=text)
        # Partial parsing should handle incomplete JSON
        result = parser.parse_result([generation], partial=True)
        assert result == {"name": "John"}


class MockPerson(BaseModel):
    """Mock Pydantic model for testing."""

    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")
    email: str | None = Field(default=None, description="The person's email")


class MockCompany(BaseModel):
    """Mock nested Pydantic model for testing."""

    company_name: str = Field(description="Company name")
    employees: list[MockPerson] = Field(description="List of employees")
    founded_year: int = Field(description="Year founded")


class TestReasoningStructuredOutputParser:
    """Tests for ReasoningStructuredOutputParser."""

    def test_parse_structured_output_without_think_tags(self) -> None:
        """Test parsing structured output without think tags."""
        parser: ReasoningStructuredOutputParser[MockPerson] = (
            ReasoningStructuredOutputParser(pydantic_object=MockPerson)
        )
        text = '{"name": "John Doe", "age": 30, "email": "john@example.com"}'
        generation = Generation(text=text)
        result = parser.parse_result([generation])
        assert isinstance(result, MockPerson)
        assert result.name == "John Doe"
        assert result.age == 30
        assert result.email == "john@example.com"

    def test_parse_structured_output_with_think_tags(self) -> None:
        """Test parsing structured output with think tags."""
        parser: ReasoningStructuredOutputParser[MockPerson] = (
            ReasoningStructuredOutputParser(pydantic_object=MockPerson)
        )
        text = '<think>Let me create a person\
            object</think>{"name": "John Doe", "age": 30}'
        generation = Generation(text=text)
        result = parser.parse_result([generation])
        assert isinstance(result, MockPerson)
        assert result.name == "John Doe"
        assert result.age == 30
        assert result.email is None

    def test_parse_structured_output_with_multiple_think_tags(self) -> None:
        """Test parsing with multiple think tags."""
        parser: ReasoningStructuredOutputParser[MockPerson] = (
            ReasoningStructuredOutputParser(pydantic_object=MockPerson)
        )
        text = """<think>Step 1: Determine name</think>
<think>Step 2: Determine age</think>
{"name": "Jane Smith", "age": 25}"""
        generation = Generation(text=text)
        result = parser.parse_result([generation])
        assert isinstance(result, MockPerson)
        assert result.name == "Jane Smith"
        assert result.age == 25

    def test_parse_structured_output_markdown_with_think_tags(self) -> None:
        """Test parsing markdown-wrapped structured output with think tags."""
        parser: ReasoningStructuredOutputParser[MockPerson] = (
            ReasoningStructuredOutputParser(pydantic_object=MockPerson)
        )
        text = """<think>Building person object</think>
```json
{"name": "Alice Brown", "age": 35, "email": "alice@example.com"}
```"""
        generation = Generation(text=text)
        result = parser.parse_result([generation])
        assert isinstance(result, MockPerson)
        assert result.name == "Alice Brown"
        assert result.age == 35
        assert result.email == "alice@example.com"

    def test_parse_nested_structured_output_with_think_tags(self) -> None:
        """Test parsing nested Pydantic models with think tags."""
        parser: ReasoningStructuredOutputParser[MockCompany] = (
            ReasoningStructuredOutputParser(pydantic_object=MockCompany)
        )
        text = """<think>Creating company with employees</think>
{
    "company_name": "Tech Corp",
    "founded_year": 2020,
    "employees": [
        {"name": "John", "age": 30},
        {"name": "Jane", "age": 28}
    ]
}"""
        generation = Generation(text=text)
        result = parser.parse_result([generation])
        assert isinstance(result, MockCompany)
        assert result.company_name == "Tech Corp"
        assert result.founded_year == 2020
        assert len(result.employees) == 2
        assert result.employees[0].name == "John"
        assert result.employees[1].name == "Jane"

    def test_parse_invalid_structured_output_with_think_tags(self) -> None:
        """Test that invalid structured output raises exception."""
        parser: ReasoningStructuredOutputParser[MockPerson] = (
            ReasoningStructuredOutputParser(pydantic_object=MockPerson)
        )
        # Missing required field 'age'
        text = '<think>Creating person</think>{"name": "John"}'
        generation = Generation(text=text)
        with pytest.raises(OutputParserException):
            parser.parse_result([generation])

    def test_parse_structured_wrong_type_with_think_tags(self) -> None:
        """Test that wrong types raise validation errors."""
        parser: ReasoningStructuredOutputParser[MockPerson] = (
            ReasoningStructuredOutputParser(pydantic_object=MockPerson)
        )
        # Age should be int, not string
        text = '<think>Creating person</think>{"name": "John", "age": "thirty"}'
        generation = Generation(text=text)
        with pytest.raises(OutputParserException):
            parser.parse_result([generation])

    def test_parse_empty_after_stripping_think_tags(self) -> None:
        """Test handling when only think tags remain."""
        parser: ReasoningStructuredOutputParser[MockPerson] = (
            ReasoningStructuredOutputParser(pydantic_object=MockPerson)
        )
        text = "<think>Only reasoning here</think>"
        generation = Generation(text=text)
        with pytest.raises(OutputParserException):
            parser.parse_result([generation])

    def test_get_format_instructions(self) -> None:
        """Test that format instructions work correctly."""
        parser: ReasoningStructuredOutputParser[MockPerson] = (
            ReasoningStructuredOutputParser(pydantic_object=MockPerson)
        )
        instructions = parser.get_format_instructions()
        assert "MockPerson" in instructions or "name" in instructions
        assert isinstance(instructions, str)

    def test_partial_structured_parsing_with_think_tags(self) -> None:
        """Test partial parsing of structured output with think tags."""
        parser: ReasoningStructuredOutputParser[MockPerson] = (
            ReasoningStructuredOutputParser(pydantic_object=MockPerson)
        )
        text = '<think>Starting</think>{"name": "John", "age": 30'
        generation = Generation(text=text)
        # Partial parsing should handle incomplete JSON
        result = parser.parse_result([generation], partial=True)
        # With partial=True, it should return what it can parse
        assert "name" in result or isinstance(result, MockPerson)

    def test_parser_with_think_tags_in_json_values(self) -> None:
        """Test that think tags in JSON string values don't cause issues."""
        parser: ReasoningStructuredOutputParser[MockPerson] = (
            ReasoningStructuredOutputParser(pydantic_object=MockPerson)
        )
        # Think tags should be stripped before JSON parsing, so they won't be in values
        text = '<think>reasoning</think>{"name": "John <Doe>", "age": 30}'
        generation = Generation(text=text)
        result = parser.parse_result([generation])
        assert isinstance(result, MockPerson)
        assert result.name == "John <Doe>"
        assert result.age == 30

    def test_multiline_think_tags_with_structured_output(self) -> None:
        """Test parsing structured output with multiline think tags."""
        parser: ReasoningStructuredOutputParser[MockPerson] = (
            ReasoningStructuredOutputParser(pydantic_object=MockPerson)
        )
        text = """<think>
Step 1: Consider the requirements
Step 2: Structure the data
Step 3: Format as JSON
</think>
{"name": "Bob Wilson", "age": 40, "email": "bob@example.com"}"""
        generation = Generation(text=text)
        result = parser.parse_result([generation])
        assert isinstance(result, MockPerson)
        assert result.name == "Bob Wilson"
        assert result.age == 40
        assert result.email == "bob@example.com"

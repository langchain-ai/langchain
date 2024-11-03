from typing import Any, Dict

from langchain_core.exceptions import OutputParserException

from langchain.output_parsers import ResponseSchema, StructuredOutputParser


def test_parse() -> None:
    """Test parsing structured output."""
    response_schemas = [
        ResponseSchema(name="name", description="desc"),
        ResponseSchema(name="age", description="desc"),
    ]
    parser = StructuredOutputParser.from_response_schemas(response_schemas)

    # Test valid JSON input
    text = '```json\n{"name": "John", "age": 30}\n```'
    expected_result = {"name": "John", "age": 30}
    result = parser.parse(text)
    assert result == expected_result, f"Expected {expected_result}, but got {result}"

    # Test invalid JSON input
    text = '```json\n{"name": "John"}\n```'
    try:
        parser.parse(text)
    except OutputParserException:
        pass  # Test passes if OutputParserException is raised
    else:
        assert False, f"Expected OutputParserException, but got {parser.parse(text)}"


def test_output_type() -> None:
    """Test the output type of the structured output parser is Dict[str, Any]."""
    response_schemas = [
        ResponseSchema(name="name", description="desc"),
        ResponseSchema(name="age", description="desc"),
    ]
    parser = StructuredOutputParser.from_response_schemas(response_schemas)
    assert parser.OutputType == Dict[str, Any]

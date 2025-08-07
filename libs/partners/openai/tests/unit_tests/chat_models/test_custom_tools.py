"""Test custom tools functionality."""

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages.tool import tool_call
from langchain_core.tools import tool

from langchain_openai.chat_models.base import (
    _convert_delta_to_message_chunk,
    _convert_dict_to_message,
)
from langchain_openai.chat_models.cfg_grammar import (
    CFGValidator,
    validate_cfg_format,
    validate_custom_tool_output,
)


def test_custom_tool_decorator():
    """Test that custom tools can be created with the `@tool` decorator."""

    @custom_tool
    def execute_code(code: str) -> str:
        """Execute arbitrary Python code."""
        return f"Executed: {code}"

    assert execute_code.custom_tool is True

    result = execute_code.invoke({"text_input": "print('hello')"})
    assert result == "Executed: print('hello')"


def test_regular_tool_not_custom():
    """Test that regular tools are not marked as custom."""

    @tool
    def get_weather(location: str) -> str:
        """Get weather for a location."""
        return f"Weather in {location}: sunny"

    assert get_weather.custom_tool is False


def test_tool_call_with_text_input():
    """Test creating tool calls with `text_input`."""

    custom_call = tool_call(
        name="execute_code", text_input="print('hello world')", id="call_123"
    )

    assert custom_call["name"] == "execute_code"
    assert custom_call.get("text_input") == "print('hello world')"
    assert "args" not in custom_call
    assert custom_call["id"] == "call_123"


def test_tool_call_validation():
    """Test that `tool_call()` allows flexible creation."""

    # Should allow both args and text_input (validation happens at execution time)
    call_with_both = tool_call(
        name="test", args={"x": 1}, text_input="some text", id="call_123"
    )
    assert call_with_both["name"] == "test"
    assert call_with_both.get("args") == {"x": 1}
    assert call_with_both.get("text_input") == "some text"

    # Should allow empty args/text_input (backward compatibility)
    call_empty = tool_call(name="test", id="call_123")
    assert call_empty["name"] == "test"
    assert call_empty.get("args", {}) == {}


def test_custom_tool_call_parsing():
    """Test parsing custom tool calls from OpenAI response format."""

    # Simulate OpenAI custom tool call response
    openai_response = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "custom",
                "name": "execute_code",
                "input": "print('hello world')",
                "id": "call_abc123",
            }
        ],
    }

    # Parse the message
    message = _convert_dict_to_message(openai_response)

    assert isinstance(message, AIMessage)
    assert len(message.tool_calls) == 1

    tool_call = message.tool_calls[0]
    assert tool_call["name"] == "execute_code"
    assert tool_call.get("text_input") == "print('hello world')"
    assert "args" not in tool_call  # Custom tools don't have an args field
    assert tool_call["id"] == "call_abc123"
    assert tool_call.get("type") == "tool_call"


def test_regular_tool_call_parsing_unchanged():
    """Test that regular tool call parsing still works."""

    # Simulate regular OpenAI function tool call response
    openai_response = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Paris", "unit": "celsius"}',
                },
                "id": "call_def456",
            }
        ],
    }

    # Parse the message
    message = _convert_dict_to_message(openai_response)

    assert isinstance(message, AIMessage)
    assert len(message.tool_calls) == 1

    tool_call = message.tool_calls[0]
    assert tool_call["name"] == "get_weather"
    assert tool_call.get("args") == {"location": "Paris", "unit": "celsius"}
    assert "text_input" not in tool_call
    assert tool_call["id"] == "call_def456"


def test_custom_tool_streaming_text_input():
    """Test streaming custom tool calls use `text_input` field."""
    chunk1 = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "custom",
                "name": "execute_code",
                "input": "print('hello",
                "id": "call_abc123",
                "index": 0,
            }
        ],
    }

    chunk2 = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "custom",
                "name": None,
                "input": " world')",
                "id": None,
                "index": 0,
            }
        ],
    }

    message_chunk1 = _convert_delta_to_message_chunk(chunk1, AIMessageChunk)
    message_chunk2 = _convert_delta_to_message_chunk(chunk2, AIMessageChunk)

    # Verify first chunk
    assert isinstance(message_chunk1, AIMessageChunk)
    assert len(message_chunk1.tool_call_chunks) == 1
    tool_call_chunk1 = message_chunk1.tool_call_chunks[0]
    assert tool_call_chunk1["name"] == "execute_code"
    assert tool_call_chunk1.get("text_input") == "print('hello"
    assert tool_call_chunk1.get("args") == ""  # Empty for custom tools
    assert tool_call_chunk1["id"] == "call_abc123"
    assert tool_call_chunk1["index"] == 0

    # Verify second chunk
    assert isinstance(message_chunk2, AIMessageChunk)
    assert len(message_chunk2.tool_call_chunks) == 1
    tool_call_chunk2 = message_chunk2.tool_call_chunks[0]
    assert tool_call_chunk2["name"] is None
    assert tool_call_chunk2.get("text_input") == " world')"
    assert tool_call_chunk2.get("args") == ""  # Empty for custom tools
    assert tool_call_chunk2["id"] is None
    assert tool_call_chunk2["index"] == 0

    # Test chunk aggregation
    combined = message_chunk1 + message_chunk2
    assert isinstance(combined, AIMessageChunk)
    assert len(combined.tool_call_chunks) == 1
    combined_chunk = combined.tool_call_chunks[0]
    assert combined_chunk["name"] == "execute_code"
    assert combined_chunk.get("text_input") == "print('hello world')"
    assert combined_chunk.get("args") == ""  # Empty for custom tools
    assert combined_chunk["id"] == "call_abc123"


def test_function_tool_streaming_args():
    """Test streaming function tool calls still use args field with JSON."""
    chunk1 = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location": "Par'},
                "id": "call_def456",
                "index": 0,
            }
        ],
    }

    chunk2 = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": None, "arguments": 'is", "unit": "celsius"}'},
                "id": None,
                "index": 0,
            }
        ],
    }

    # Parse the chunks
    message_chunk1 = _convert_delta_to_message_chunk(chunk1, AIMessageChunk)
    message_chunk2 = _convert_delta_to_message_chunk(chunk2, AIMessageChunk)

    # Verify first chunk
    assert isinstance(message_chunk1, AIMessageChunk)
    assert len(message_chunk1.tool_call_chunks) == 1
    tool_call_chunk1 = message_chunk1.tool_call_chunks[0]
    assert tool_call_chunk1["name"] == "get_weather"
    assert tool_call_chunk1.get("args") == '{"location": "Par'
    assert "text_input" not in tool_call_chunk1
    assert tool_call_chunk1["id"] == "call_def456"
    assert tool_call_chunk1["index"] == 0

    # Verify second chunk
    assert isinstance(message_chunk2, AIMessageChunk)
    assert len(message_chunk2.tool_call_chunks) == 1
    tool_call_chunk2 = message_chunk2.tool_call_chunks[0]
    assert tool_call_chunk2["name"] is None
    assert tool_call_chunk2.get("args") == 'is", "unit": "celsius"}'
    assert "text_input" not in tool_call_chunk2
    assert tool_call_chunk2["id"] is None
    assert tool_call_chunk2["index"] == 0

    # Test chunk aggregation
    combined = message_chunk1 + message_chunk2
    assert isinstance(combined, AIMessageChunk)
    assert len(combined.tool_call_chunks) == 1
    combined_chunk = combined.tool_call_chunks[0]
    assert combined_chunk["name"] == "get_weather"
    assert combined_chunk.get("args") == '{"location": "Paris", "unit": "celsius"}'
    assert "text_input" not in combined_chunk
    assert combined_chunk["id"] == "call_def456"


def test_mixed_tool_streaming():
    """Test streaming with both custom and function tools in same response."""
    # Simulate mixed tool streaming chunk from OpenAI
    chunk = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "custom",
                "name": "execute_code",
                "input": "x = 5",
                "id": "call_custom_123",
                "index": 0,
            },
            {
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
                "id": "call_func_456",
                "index": 1,
            },
        ],
    }

    # Parse the chunk
    message_chunk = _convert_delta_to_message_chunk(chunk, AIMessageChunk)

    assert isinstance(message_chunk, AIMessageChunk)
    assert len(message_chunk.tool_call_chunks) == 2

    # Verify custom tool chunk
    custom_chunk = message_chunk.tool_call_chunks[0]
    assert custom_chunk["name"] == "execute_code"
    assert custom_chunk.get("text_input") == "x = 5"
    assert custom_chunk.get("args") == ""  # Empty for custom tools
    assert custom_chunk["id"] == "call_custom_123"
    assert custom_chunk["index"] == 0

    # Verify function tool chunk
    function_chunk = message_chunk.tool_call_chunks[1]
    assert function_chunk["name"] == "get_weather"
    assert function_chunk.get("args") == '{"location": "NYC"}'
    assert "text_input" not in function_chunk
    assert function_chunk["id"] == "call_func_456"
    assert function_chunk["index"] == 1


# CFG Grammar Tests


class TestCFGValidator:
    """Test CFG validator functionality."""

    def test_cfg_validator_initialization(self):
        """Test CFG validator can be initialized with valid grammar."""
        grammar = """
        start: expr
        expr: NUMBER ("+" | "-" | "*") NUMBER
        NUMBER: /[0-9]+/
        """
        validator = CFGValidator(grammar)
        assert validator.grammar == grammar
        assert validator.parser is not None

    def test_cfg_validator_initialization_invalid_grammar(self):
        """Test CFG validator raises error with invalid grammar."""
        invalid_grammar = "invalid grammar string [["
        with pytest.raises(Exception):
            CFGValidator(invalid_grammar)

    def test_cfg_validator_validate_valid_input(self):
        """Test CFG validator accepts valid input."""
        grammar = """
        start: expr
        expr: NUMBER ("+" | "-" | "*") NUMBER
        NUMBER: /[0-9]+/
        %import common.WS
        %ignore WS
        """
        validator = CFGValidator(grammar)

        assert validator.validate("5 + 3") is True
        assert validator.validate("10 * 2") is True
        assert validator.validate("100 - 50") is True

    def test_cfg_validator_validate_invalid_input(self):
        """Test CFG validator rejects invalid input."""
        grammar = """
        start: expr
        expr: NUMBER ("+" | "-" | "*") NUMBER
        NUMBER: /[0-9]+/
        %import common.WS
        %ignore WS
        """
        validator = CFGValidator(grammar)

        assert validator.validate("hello") is False
        assert validator.validate("5 + + 3") is False
        assert validator.validate("5 + ") is False
        assert validator.validate("+ 5") is False

    def test_cfg_validator_parse_valid_input(self):
        """Test CFG validator can parse valid input."""
        grammar = """
        start: expr
        expr: NUMBER ("+" | "-" | "*") NUMBER
        NUMBER: /[0-9]+/
        %import common.WS
        %ignore WS
        """
        validator = CFGValidator(grammar)

        tree = validator.parse("5 + 3")
        assert tree is not None
        assert str(tree.data) == "start"

    def test_cfg_validator_parse_invalid_input(self):
        """Test CFG validator raises error when parsing invalid input."""
        grammar = """
        start: expr
        expr: NUMBER ("+" | "-" | "*") NUMBER
        NUMBER: /[0-9]+/
        %import common.WS
        %ignore WS
        """
        validator = CFGValidator(grammar)

        with pytest.raises(Exception):
            validator.parse("invalid input")

    def test_cfg_validator_complex_grammar(self):
        """Test CFG validator with more complex grammar."""
        # SQL-like grammar
        grammar = """
        start: query
        query: "SELECT" field_list "FROM" table_name where_clause?
        field_list: field ("," field)*
        field: IDENTIFIER
        table_name: IDENTIFIER
        where_clause: "WHERE" condition
        condition: IDENTIFIER "=" STRING
        IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/
        STRING: /"[^"]*"/
        %import common.WS
        %ignore WS
        """
        validator = CFGValidator(grammar)

        # Valid SQL queries
        assert validator.validate("SELECT id FROM users") is True
        assert (
            validator.validate('SELECT id, name FROM users WHERE status = "active"')
            is True
        )

        # Invalid SQL queries
        assert validator.validate("SELECT FROM users") is False
        assert validator.validate("INVALID QUERY") is False

    def test_cfg_validator_python_code_grammar(self):
        """Test CFG validator with Python code grammar."""
        # Simple Python expression grammar
        grammar = """
        start: statement
        statement: assignment | expression
        assignment: IDENTIFIER "=" expression
        expression: term (("+" | "-") term)*
        term: factor (("*" | "/") factor)*
        factor: NUMBER | IDENTIFIER | "(" expression ")"
        IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/
        NUMBER: /[0-9]+/
        %import common.WS
        %ignore WS
        """
        validator = CFGValidator(grammar)

        # Valid Python expressions
        assert validator.validate("x = 5") is True
        assert validator.validate("result = a + b * c") is True
        assert validator.validate("(x + y) / 2") is True

        # Invalid Python expressions
        assert validator.validate("x =") is False
        assert validator.validate("+ + +") is False


class TestValidateCFGFormat:
    """Test validate_cfg_format function."""

    def test_validate_cfg_format_valid_grammar_format(self):
        """Test validate_cfg_format with valid grammar format."""
        tool_format = {
            "type": "grammar",
            "grammar": "start: expr\nexpr: NUMBER\nNUMBER: /[0-9]+/",
        }

        validator = validate_cfg_format(tool_format)
        assert validator is not None
        assert isinstance(validator, CFGValidator)

    def test_validate_cfg_format_non_grammar_format(self):
        """Test validate_cfg_format with non-grammar format."""
        tool_format = {"type": "json_schema", "schema": {}}

        validator = validate_cfg_format(tool_format)
        assert validator is None

    def test_validate_cfg_format_missing_grammar(self):
        """Test validate_cfg_format with missing grammar field."""
        tool_format = {"type": "grammar"}

        with pytest.raises(ValueError, match="Grammar format requires 'grammar' field"):
            validate_cfg_format(tool_format)

    def test_validate_cfg_format_invalid_grammar_type(self):
        """Test validate_cfg_format with non-string grammar."""
        tool_format = {"type": "grammar", "grammar": ["not", "a", "string"]}

        with pytest.raises(ValueError, match="Grammar must be a string"):
            validate_cfg_format(tool_format)

    def test_validate_cfg_format_invalid_grammar_syntax(self):
        """Test validate_cfg_format with invalid grammar syntax."""
        tool_format = {"type": "grammar", "grammar": "invalid grammar [[ syntax"}

        with pytest.raises(ValueError, match="Invalid grammar specification"):
            validate_cfg_format(tool_format)

    def test_validate_cfg_format_non_dict_input(self):
        """Test validate_cfg_format with non-dict input."""
        assert validate_cfg_format("not a dict") is None
        assert validate_cfg_format(None) is None
        assert validate_cfg_format([]) is None


class TestValidateCustomToolOutput:
    """Test validate_custom_tool_output function."""

    def test_validate_custom_tool_output_with_validator(self):
        """Test validate_custom_tool_output with CFG validator."""
        grammar = """
        start: expr
        expr: NUMBER ("+" | "-") NUMBER
        NUMBER: /[0-9]+/
        %import common.WS
        %ignore WS
        """
        validator = CFGValidator(grammar)

        # Valid outputs
        assert validate_custom_tool_output("5 + 3", validator) is True
        assert validate_custom_tool_output("10 - 7", validator) is True

        # Invalid outputs
        assert validate_custom_tool_output("hello", validator) is False
        assert validate_custom_tool_output("5 + + 3", validator) is False

    def test_validate_custom_tool_output_without_validator(self):
        """Test validate_custom_tool_output without CFG validator."""
        # Should return True when no validator is provided
        assert validate_custom_tool_output("any string", None) is True
        assert validate_custom_tool_output("", None) is True
        assert validate_custom_tool_output("invalid grammar", None) is True


class TestCFGIntegration:
    """Test CFG integration with custom tools."""

    def test_custom_tool_call_with_cfg_validation(self):
        """Test that CFG validation can be integrated with custom tool calls."""
        # Arithmetic expressions
        grammar = """
        start: expr
        expr: NUMBER ("+" | "-" | "*" | "/") NUMBER
        NUMBER: /[0-9]+/
        %import common.WS
        %ignore WS
        """

        # Simulate custom tool definition with CFG format
        tool_format = {"type": "grammar", "grammar": grammar}

        validator = validate_cfg_format(tool_format)
        assert validator is not None

        # Test valid tool outputs
        valid_outputs = ["5 + 3", "10 * 2", "100 / 5", "50 - 25"]
        for output in valid_outputs:
            assert validate_custom_tool_output(output, validator) is True

        # Test invalid tool outputs
        invalid_outputs = ["hello", "5 + + 3", "invalid", "5 +"]
        for output in invalid_outputs:
            assert validate_custom_tool_output(output, validator) is False

    def test_sql_query_cfg_validation(self):
        """Test CFG validation for SQL-like queries."""
        sql_grammar = """
        start: query
        query: "SELECT" field "FROM" table
        field: IDENTIFIER
        table: IDENTIFIER
        IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/
        %import common.WS
        %ignore WS
        """

        tool_format = {"type": "grammar", "grammar": sql_grammar}

        validator = validate_cfg_format(tool_format)
        assert validator is not None

        # Valid SQL queries
        assert validate_custom_tool_output("SELECT id FROM users", validator) is True
        assert (
            validate_custom_tool_output("SELECT name FROM products", validator) is True
        )

        # Invalid SQL queries
        assert validate_custom_tool_output("SELECT FROM users", validator) is False
        assert validate_custom_tool_output("INVALID QUERY", validator) is False

    def test_python_expression_cfg_validation(self):
        """Test CFG validation for Python expressions."""
        python_grammar = """
        start: assignment
        assignment: IDENTIFIER "=" expression
        expression: term (("+" | "-") term)*
        term: factor (("*" | "/") factor)*
        factor: NUMBER | IDENTIFIER
        IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/
        NUMBER: /[0-9]+/
        %import common.WS
        %ignore WS
        """

        tool_format = {"type": "grammar", "grammar": python_grammar}

        validator = validate_cfg_format(tool_format)
        assert validator is not None

        # Valid Python assignments
        assert validate_custom_tool_output("x = 5", validator) is True
        assert validate_custom_tool_output("result = a + b", validator) is True
        assert (
            validate_custom_tool_output("total = price * quantity", validator) is True
        )

        # Invalid Python assignments
        assert validate_custom_tool_output("x =", validator) is False
        assert validate_custom_tool_output("= 5", validator) is False
        assert validate_custom_tool_output("hello world", validator) is False


@pytest.mark.skipif(CFGValidator is None, reason="lark package not available")
class TestCFGErrorHandling:
    """Test CFG error handling when lark is not available."""

    def test_cfg_validator_import_error(self, monkeypatch):
        """Test CFG validator handles missing lark import gracefully."""
        # Mock the import to fail
        monkeypatch.setattr("langchain_openai.chat_models.cfg_grammar.Lark", None)

        with pytest.raises(ImportError, match="The 'lark' package is required"):
            CFGValidator("start: NUMBER\nNUMBER: /[0-9]+/")

    def test_cfg_edge_cases(self):
        """Test CFG validator edge cases."""
        grammar = """
        start: item*
        item: WORD
        WORD: /\\w+/
        %import common.WS
        %ignore WS
        """
        validator = CFGValidator(grammar)

        # Empty string should be valid (zero items)
        assert validator.validate("") is True

        # Single word should be valid
        assert validator.validate("hello") is True

        # Multiple words should be valid
        assert validator.validate("hello world test") is True

"""Tests for default_tool_parser and default_tool_chunk_parser."""

from langchain_core.messages.tool import (
    default_tool_chunk_parser,
    default_tool_parser,
)


class TestDefaultToolParser:
    """Tests for default_tool_parser."""

    def test_valid_tool_calls(self) -> None:
        """Valid tool calls are parsed correctly."""
        raw = [
            {
                "function": {"name": "my_tool", "arguments": '{"a": 1}'},
                "id": "call_1",
            },
            {
                "function": {"name": "other_tool", "arguments": '{"b": "x"}'},
                "id": "call_2",
            },
        ]
        tool_calls, invalid_tool_calls = default_tool_parser(raw)
        assert len(tool_calls) == 2
        assert len(invalid_tool_calls) == 0
        assert tool_calls[0]["name"] == "my_tool"
        assert tool_calls[0]["args"] == {"a": 1}
        assert tool_calls[0]["id"] == "call_1"
        assert tool_calls[1]["name"] == "other_tool"
        assert tool_calls[1]["args"] == {"b": "x"}

    def test_missing_function_key_skipped(self) -> None:
        """Entries without a 'function' key are silently skipped."""
        raw = [{"id": "call_1"}]
        tool_calls, invalid_tool_calls = default_tool_parser(raw)
        assert len(tool_calls) == 0
        assert len(invalid_tool_calls) == 0

    def test_empty_list(self) -> None:
        """Empty input produces empty output."""
        tool_calls, invalid_tool_calls = default_tool_parser([])
        assert tool_calls == []
        assert invalid_tool_calls == []

    def test_invalid_json_arguments(self) -> None:
        """Malformed JSON in arguments routes to invalid_tool_calls."""
        raw = [
            {
                "function": {"name": "my_tool", "arguments": "not json"},
                "id": "call_1",
            }
        ]
        tool_calls, invalid_tool_calls = default_tool_parser(raw)
        assert len(tool_calls) == 0
        assert len(invalid_tool_calls) == 1
        assert invalid_tool_calls[0]["name"] == "my_tool"
        assert invalid_tool_calls[0]["args"] == "not json"
        assert invalid_tool_calls[0]["id"] == "call_1"
        assert invalid_tool_calls[0]["error"] is not None

    def test_function_value_is_none(self) -> None:
        """When function value is None, entry is routed to invalid_tool_calls."""
        raw = [{"function": None, "id": "call_1"}]
        tool_calls, invalid_tool_calls = default_tool_parser(raw)
        assert len(tool_calls) == 0
        assert len(invalid_tool_calls) == 1
        assert invalid_tool_calls[0]["name"] is None
        assert invalid_tool_calls[0]["id"] == "call_1"
        assert invalid_tool_calls[0]["error"] is not None
        assert "dict" in invalid_tool_calls[0]["error"]

    def test_function_dict_missing_keys(self) -> None:
        """When function dict is empty or missing keys, entry is invalid."""
        raw = [{"function": {}, "id": "call_1"}]
        tool_calls, invalid_tool_calls = default_tool_parser(raw)
        assert len(tool_calls) == 0
        assert len(invalid_tool_calls) == 1
        assert invalid_tool_calls[0]["id"] == "call_1"
        assert invalid_tool_calls[0]["error"] is not None

    def test_malformed_does_not_drop_valid_calls(self) -> None:
        """A malformed entry does not prevent valid entries from being parsed."""
        raw = [
            {"function": None, "id": "bad"},
            {
                "function": {"name": "good_tool", "arguments": '{"a": 1}'},
                "id": "good",
            },
        ]
        tool_calls, invalid_tool_calls = default_tool_parser(raw)
        assert len(tool_calls) == 1
        assert len(invalid_tool_calls) == 1
        assert tool_calls[0]["name"] == "good_tool"
        assert tool_calls[0]["id"] == "good"
        assert invalid_tool_calls[0]["id"] == "bad"

    def test_function_name_is_none(self) -> None:
        """When function name is None, it defaults to empty string."""
        raw = [
            {
                "function": {"name": None, "arguments": '{"a": 1}'},
                "id": "call_1",
            }
        ]
        tool_calls, invalid_tool_calls = default_tool_parser(raw)
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == ""

    def test_missing_id(self) -> None:
        """Missing id field defaults to None."""
        raw = [
            {
                "function": {"name": "my_tool", "arguments": '{"a": 1}'},
            }
        ]
        tool_calls, invalid_tool_calls = default_tool_parser(raw)
        assert len(tool_calls) == 1
        assert tool_calls[0]["id"] is None


class TestDefaultToolChunkParser:
    """Tests for default_tool_chunk_parser."""

    def test_valid_chunks(self) -> None:
        """Valid tool call chunks are parsed correctly."""
        raw = [
            {
                "function": {"name": "my_tool", "arguments": '{"a":'},
                "id": "call_1",
                "index": 0,
            }
        ]
        chunks = default_tool_chunk_parser(raw)
        assert len(chunks) == 1
        assert chunks[0]["name"] == "my_tool"
        assert chunks[0]["args"] == '{"a":'
        assert chunks[0]["id"] == "call_1"
        assert chunks[0]["index"] == 0

    def test_missing_function_key(self) -> None:
        """Entries without 'function' key get None for name and args."""
        raw = [{"id": "call_1", "index": 0}]
        chunks = default_tool_chunk_parser(raw)
        assert len(chunks) == 1
        assert chunks[0]["name"] is None
        assert chunks[0]["args"] is None

    def test_function_value_is_none(self) -> None:
        """When function value is None, name and args default to None."""
        raw = [{"function": None, "id": "call_1", "index": 0}]
        chunks = default_tool_chunk_parser(raw)
        assert len(chunks) == 1
        assert chunks[0]["name"] is None
        assert chunks[0]["args"] is None

    def test_function_dict_missing_keys(self) -> None:
        """When function dict is missing keys, values default to None."""
        raw = [{"function": {}, "id": "call_1", "index": 0}]
        chunks = default_tool_chunk_parser(raw)
        assert len(chunks) == 1
        assert chunks[0]["name"] is None
        assert chunks[0]["args"] is None

    def test_empty_list(self) -> None:
        """Empty input produces empty output."""
        chunks = default_tool_chunk_parser([])
        assert chunks == []

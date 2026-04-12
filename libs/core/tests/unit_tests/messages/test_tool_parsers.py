"""Tests for default_tool_parser and default_tool_chunk_parser."""

from langchain_core.messages.tool import (
    default_tool_chunk_parser,
    default_tool_parser,
)


class TestDefaultToolParser:
    def test_valid_tool_call(self) -> None:
        raw = [
            {
                "function": {"name": "get_weather", "arguments": '{"city": "Cairo"}'},
                "id": "call_1",
            }
        ]
        tool_calls, invalid_tool_calls = default_tool_parser(raw)
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "get_weather"
        assert tool_calls[0]["args"] == {"city": "Cairo"}
        assert tool_calls[0]["id"] == "call_1"
        assert len(invalid_tool_calls) == 0

    def test_invalid_json_arguments(self) -> None:
        raw = [
            {
                "function": {"name": "get_weather", "arguments": "not json"},
                "id": "call_1",
            }
        ]
        tool_calls, invalid_tool_calls = default_tool_parser(raw)
        assert len(tool_calls) == 0
        assert len(invalid_tool_calls) == 1
        assert invalid_tool_calls[0]["name"] == "get_weather"
        assert invalid_tool_calls[0]["args"] == "not json"

    def test_missing_function_key_skipped(self) -> None:
        raw = [{"id": "call_1"}]
        tool_calls, invalid_tool_calls = default_tool_parser(raw)
        assert len(tool_calls) == 0
        assert len(invalid_tool_calls) == 0

    def test_function_is_none(self) -> None:
        """function key present but value is None should not crash."""
        raw = [{"function": None, "id": "call_1"}]
        tool_calls, invalid_tool_calls = default_tool_parser(raw)
        assert len(tool_calls) == 0
        assert len(invalid_tool_calls) == 1
        assert invalid_tool_calls[0]["id"] == "call_1"

    def test_function_missing_name_key(self) -> None:
        """function dict without 'name' should not crash."""
        raw = [{"function": {"arguments": '{"x": 1}'}, "id": "call_1"}]
        tool_calls, invalid_tool_calls = default_tool_parser(raw)
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == ""
        assert tool_calls[0]["args"] == {"x": 1}

    def test_function_missing_arguments_key(self) -> None:
        """function dict without 'arguments' should not crash."""
        raw = [{"function": {"name": "my_tool"}, "id": "call_1"}]
        tool_calls, invalid_tool_calls = default_tool_parser(raw)
        # Missing arguments → empty string → JSONDecodeError → invalid
        assert len(tool_calls) == 0
        assert len(invalid_tool_calls) == 1
        assert invalid_tool_calls[0]["name"] == "my_tool"

    def test_function_empty_dict(self) -> None:
        """Empty function dict should not crash."""
        raw = [{"function": {}, "id": "call_1"}]
        tool_calls, invalid_tool_calls = default_tool_parser(raw)
        assert len(tool_calls) == 0
        assert len(invalid_tool_calls) == 1

    def test_function_is_string(self) -> None:
        """function key with string value should not crash."""
        raw = [{"function": "some_string", "id": "call_1"}]
        tool_calls, invalid_tool_calls = default_tool_parser(raw)
        assert len(tool_calls) == 0
        assert len(invalid_tool_calls) == 1

    def test_mixed_valid_and_malformed(self) -> None:
        """Valid tool calls should still be parsed even if others are malformed."""
        raw = [
            {"function": None, "id": "bad_1"},
            {
                "function": {"name": "good_tool", "arguments": '{"a": 1}'},
                "id": "good_1",
            },
            {"function": {"name": "bad_json", "arguments": "{bad"}, "id": "bad_2"},
        ]
        tool_calls, invalid_tool_calls = default_tool_parser(raw)
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "good_tool"
        assert len(invalid_tool_calls) == 2

    def test_function_arguments_is_none(self) -> None:
        """arguments key present but None should not crash."""
        raw = [
            {"function": {"name": "my_tool", "arguments": None}, "id": "call_1"}
        ]
        tool_calls, invalid_tool_calls = default_tool_parser(raw)
        # None arguments → or "" → JSONDecodeError → invalid
        assert len(tool_calls) == 0
        assert len(invalid_tool_calls) == 1


class TestDefaultToolChunkParser:
    def test_valid_chunk(self) -> None:
        raw = [
            {
                "function": {"name": "get_weather", "arguments": '{"city":'},
                "id": "call_1",
                "index": 0,
            }
        ]
        chunks = default_tool_chunk_parser(raw)
        assert len(chunks) == 1
        assert chunks[0]["name"] == "get_weather"
        assert chunks[0]["args"] == '{"city":'
        assert chunks[0]["id"] == "call_1"
        assert chunks[0]["index"] == 0

    def test_missing_function_key(self) -> None:
        raw = [{"id": "call_1", "index": 0}]
        chunks = default_tool_chunk_parser(raw)
        assert len(chunks) == 1
        assert chunks[0]["name"] is None
        assert chunks[0]["args"] is None

    def test_function_is_none(self) -> None:
        """function key present but None should not crash."""
        raw = [{"function": None, "id": "call_1", "index": 0}]
        chunks = default_tool_chunk_parser(raw)
        assert len(chunks) == 1
        assert chunks[0]["name"] is None
        assert chunks[0]["args"] is None

    def test_function_missing_keys(self) -> None:
        """function dict with missing name/arguments should not crash."""
        raw = [{"function": {}, "id": "call_1", "index": 0}]
        chunks = default_tool_chunk_parser(raw)
        assert len(chunks) == 1
        assert chunks[0]["name"] is None
        assert chunks[0]["args"] is None

    def test_function_partial_keys(self) -> None:
        """function dict with only arguments should not crash."""
        raw = [{"function": {"arguments": '{"x":'}, "id": "call_1", "index": 0}]
        chunks = default_tool_chunk_parser(raw)
        assert len(chunks) == 1
        assert chunks[0]["name"] is None
        assert chunks[0]["args"] == '{"x":'

    def test_function_is_string(self) -> None:
        """function key with non-dict value should not crash."""
        raw = [{"function": "bad", "id": "call_1", "index": 0}]
        chunks = default_tool_chunk_parser(raw)
        assert len(chunks) == 1
        assert chunks[0]["name"] is None
        assert chunks[0]["args"] is None

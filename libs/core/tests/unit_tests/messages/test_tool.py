"""Tests for default_tool_parser and default_tool_chunk_parser."""

from langchain_core.messages.tool import (
    default_tool_chunk_parser,
    default_tool_parser,
)


class TestDefaultToolParser:
    def test_valid_tool_call(self) -> None:
        raw = [
            {
                "function": {"name": "my_tool", "arguments": '{"a": 1}'},
                "id": "call_1",
            }
        ]
        tool_calls, invalid_tool_calls = default_tool_parser(raw)
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "my_tool"
        assert tool_calls[0]["args"] == {"a": 1}
        assert tool_calls[0]["id"] == "call_1"
        assert len(invalid_tool_calls) == 0

    def test_invalid_json_arguments(self) -> None:
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

    def test_missing_function_key_skipped(self) -> None:
        raw = [{"id": "call_1"}]
        tool_calls, invalid_tool_calls = default_tool_parser(raw)
        assert len(tool_calls) == 0
        assert len(invalid_tool_calls) == 0

    def test_function_value_none(self) -> None:
        """When function value is None, the call should be marked invalid."""
        raw = [{"function": None, "id": "call_1"}]
        tool_calls, invalid_tool_calls = default_tool_parser(raw)
        assert len(tool_calls) == 0
        assert len(invalid_tool_calls) == 1
        assert invalid_tool_calls[0]["id"] == "call_1"

    def test_function_dict_missing_keys(self) -> None:
        """When function dict is empty, the call should be marked invalid."""
        raw = [{"function": {}, "id": "call_1"}]
        tool_calls, invalid_tool_calls = default_tool_parser(raw)
        assert len(tool_calls) == 0
        assert len(invalid_tool_calls) == 1

    def test_malformed_does_not_drop_valid_calls(self) -> None:
        """A malformed call should not prevent valid calls from being parsed."""
        raw = [
            {"function": None, "id": "bad"},
            {
                "function": {"name": "good_tool", "arguments": '{"a": 1}'},
                "id": "good",
            },
        ]
        tool_calls, invalid_tool_calls = default_tool_parser(raw)
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "good_tool"
        assert tool_calls[0]["id"] == "good"
        assert len(invalid_tool_calls) == 1
        assert invalid_tool_calls[0]["id"] == "bad"


class TestDefaultToolChunkParser:
    def test_valid_chunk(self) -> None:
        raw = [
            {
                "function": {"name": "my_tool", "arguments": '{"a": 1}'},
                "id": "call_1",
                "index": 0,
            }
        ]
        chunks = default_tool_chunk_parser(raw)
        assert len(chunks) == 1
        assert chunks[0]["name"] == "my_tool"
        assert chunks[0]["args"] == '{"a": 1}'
        assert chunks[0]["id"] == "call_1"
        assert chunks[0]["index"] == 0

    def test_missing_function_key(self) -> None:
        raw = [{"id": "call_1", "index": 0}]
        chunks = default_tool_chunk_parser(raw)
        assert len(chunks) == 1
        assert chunks[0]["name"] is None
        assert chunks[0]["args"] is None

    def test_function_value_none(self) -> None:
        """When function value is None, should not crash."""
        raw = [{"function": None, "id": "call_1", "index": 0}]
        chunks = default_tool_chunk_parser(raw)
        assert len(chunks) == 1
        assert chunks[0]["name"] is None
        assert chunks[0]["args"] is None

    def test_function_dict_missing_keys(self) -> None:
        """When function dict has missing keys, should use None defaults."""
        raw = [{"function": {}, "id": "call_1", "index": 0}]
        chunks = default_tool_chunk_parser(raw)
        assert len(chunks) == 1
        assert chunks[0]["name"] is None
        assert chunks[0]["args"] is None

    def test_malformed_does_not_drop_valid_chunks(self) -> None:
        """A malformed chunk should not prevent valid chunks from being parsed."""
        raw = [
            {"function": None, "id": "bad", "index": 0},
            {
                "function": {"name": "good_tool", "arguments": '{"a": 1}'},
                "id": "good",
                "index": 1,
            },
        ]
        chunks = default_tool_chunk_parser(raw)
        assert len(chunks) == 2
        assert chunks[1]["name"] == "good_tool"

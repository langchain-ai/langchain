"""Tests for merge utilities."""

from langchain_core.utils._merge import merge_dicts


def test_merge_dicts_tool_calls_not_concatenated():
    """Test that tool call fields are not concatenated during merge."""
    # Simulate streaming chunks for parallel tool calls
    left = {
        "tool_calls": [
            {
                "index": 0,
                "id": "call_abc",
                "name": "read_file",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path"'},
            }
        ]
    }

    right = {
        "tool_calls": [
            {
                "index": 0,
                "id": "",
                "name": "",
                "type": "",
                "function": {"name": "", "arguments": ':"config.yaml"}'},
            }
        ]
    }

    # Merge the dicts
    merged = merge_dicts(left, right)

    # Tool call fields should NOT be concatenated
    # The id, name, and type should remain as they were in left dict
    # Only arguments should be concatenated
    assert merged["tool_calls"][0]["id"] == "call_abc", "id should not be concatenated"
    assert (
        merged["tool_calls"][0]["name"] == "read_file"
    ), "name should not be concatenated"
    assert (
        merged["tool_calls"][0]["type"] == "function"
    ), "type should not be concatenated"
    assert (
        merged["tool_calls"][0]["function"]["name"] == "read_file"
    ), "function name should not be concatenated"
    assert (
        merged["tool_calls"][0]["function"]["arguments"]
        == '{"path":"config.yaml"}'
    ), "arguments should be concatenated"


def test_merge_dicts_multiple_tool_calls():
    """Test merging with multiple tool calls in streaming."""
    left = {
        "tool_calls": [
            {"index": 0, "id": "call_1", "name": "tool_a", "type": "function"},
            {"index": 1, "id": "call_2", "name": "tool_b", "type": "function"},
        ]
    }

    right = {
        "tool_calls": [
            {"index": 0, "id": "", "name": "", "type": ""},
            {"index": 1, "id": "", "name": "", "type": ""},
        ]
    }

    merged = merge_dicts(left, right)

    assert merged["tool_calls"][0]["id"] == "call_1"
    assert merged["tool_calls"][0]["name"] == "tool_a"
    assert merged["tool_calls"][0]["type"] == "function"
    assert merged["tool_calls"][1]["id"] == "call_2"
    assert merged["tool_calls"][1]["name"] == "tool_b"
    assert merged["tool_calls"][1]["type"] == "function"


def test_merge_dicts_tool_call_id_only_once():
    """Test that tool_call_id is not concatenated when same in both dicts."""
    left = {"tool_call_id": "call_abc"}
    right = {"tool_call_id": "call_abc"}

    merged = merge_dicts(left, right)

    assert merged["tool_call_id"] == "call_abc", "tool_call_id should not be duplicated"


def test_merge_dicts_tool_call_fields_not_concatenated():
    """Test that tool call fields with empty right values are not concatenated."""
    # This is the actual bug reported in issue #34807
    # When streaming, providers send empty strings for tool call fields
    # These should not be concatenated with existing values
    left = {"id": "call_abc", "name": "read_file", "type": "function"}

    # Streaming providers often send empty strings for these fields
    right = {"id": "", "name": "", "type": ""}

    merged = merge_dicts(left, right)

    # Empty strings should not be concatenated
    assert merged["id"] == "call_abc", "id should not have empty string concatenated"
    assert (
        merged["name"] == "read_file"
    ), "name should not have empty string concatenated"
    assert (
        merged["type"] == "function"
    ), "type should not have empty string concatenated"

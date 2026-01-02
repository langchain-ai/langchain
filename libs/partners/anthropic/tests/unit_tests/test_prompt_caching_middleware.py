"""Unit tests for cache_control handling with code_execution blocks."""

from langchain_anthropic.chat_models import (
    _collect_code_execution_tool_ids,
    _is_code_execution_related_block,
)


class TestCollectCodeExecutionToolIds:
    """Tests for _collect_code_execution_tool_ids function."""

    def test_empty_messages(self) -> None:
        """Test with empty messages list."""
        result = _collect_code_execution_tool_ids([])
        assert result == set()

    def test_no_code_execution_calls(self) -> None:
        """Test messages without any code_execution calls."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_regular",
                        "name": "get_weather",
                        "input": {"location": "NYC"},
                    }
                ],
            },
        ]
        result = _collect_code_execution_tool_ids(messages)
        assert result == set()

    def test_single_code_execution_call(self) -> None:
        """Test with a single code_execution tool call."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_code_exec_1",
                        "name": "get_weather",
                        "input": {"location": "NYC"},
                        "caller": {
                            "type": "code_execution_20250825",
                            "tool_id": "srvtoolu_abc123",
                        },
                    }
                ],
            },
        ]
        result = _collect_code_execution_tool_ids(messages)
        assert result == {"toolu_code_exec_1"}

    def test_multiple_code_execution_calls(self) -> None:
        """Test with multiple code_execution tool calls."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_regular",
                        "name": "search",
                        "input": {"query": "test"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_code_exec_1",
                        "name": "get_weather",
                        "input": {"location": "NYC"},
                        "caller": {
                            "type": "code_execution_20250825",
                            "tool_id": "srvtoolu_abc",
                        },
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_code_exec_2",
                        "name": "get_weather",
                        "input": {"location": "SF"},
                        "caller": {
                            "type": "code_execution_20250825",
                            "tool_id": "srvtoolu_def",
                        },
                    },
                ],
            },
        ]
        result = _collect_code_execution_tool_ids(messages)
        assert result == {"toolu_code_exec_1", "toolu_code_exec_2"}
        assert "toolu_regular" not in result

    def test_future_code_execution_version(self) -> None:
        """Test with a hypothetical future code_execution version."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_future",
                        "name": "get_weather",
                        "input": {},
                        "caller": {
                            "type": "code_execution_20260101",
                            "tool_id": "srvtoolu_future",
                        },
                    }
                ],
            },
        ]
        result = _collect_code_execution_tool_ids(messages)
        assert result == {"toolu_future"}

    def test_ignores_user_messages(self) -> None:
        """Test that user messages are ignored."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": "result",
                    }
                ],
            },
        ]
        result = _collect_code_execution_tool_ids(messages)
        assert result == set()

    def test_handles_string_content(self) -> None:
        """Test that string content is handled gracefully."""
        messages = [
            {
                "role": "assistant",
                "content": "Just a text response",
            },
        ]
        result = _collect_code_execution_tool_ids(messages)
        assert result == set()


class TestIsCodeExecutionRelatedBlock:
    """Tests for _is_code_execution_related_block function."""

    def test_regular_tool_use_block(self) -> None:
        """Test regular tool_use block without caller."""
        block = {
            "type": "tool_use",
            "id": "toolu_regular",
            "name": "get_weather",
            "input": {"location": "NYC"},
        }
        assert not _is_code_execution_related_block(block, set())

    def test_code_execution_tool_use_block(self) -> None:
        """Test tool_use block called by code_execution."""
        block = {
            "type": "tool_use",
            "id": "toolu_code_exec",
            "name": "get_weather",
            "input": {"location": "NYC"},
            "caller": {
                "type": "code_execution_20250825",
                "tool_id": "srvtoolu_abc",
            },
        }
        assert _is_code_execution_related_block(block, set())

    def test_regular_tool_result_block(self) -> None:
        """Test tool_result block for regular tool."""
        block = {
            "type": "tool_result",
            "tool_use_id": "toolu_regular",
            "content": "Sunny, 72°F",
        }
        code_exec_ids = {"toolu_code_exec"}
        assert not _is_code_execution_related_block(block, code_exec_ids)

    def test_code_execution_tool_result_block(self) -> None:
        """Test tool_result block for code_execution called tool."""
        block = {
            "type": "tool_result",
            "tool_use_id": "toolu_code_exec",
            "content": "Sunny, 72°F",
        }
        code_exec_ids = {"toolu_code_exec"}
        assert _is_code_execution_related_block(block, code_exec_ids)

    def test_text_block(self) -> None:
        """Test that text blocks are not flagged."""
        block = {"type": "text", "text": "Hello world"}
        assert not _is_code_execution_related_block(block, set())

    def test_non_dict_block(self) -> None:
        """Test that non-dict values return False."""
        assert not _is_code_execution_related_block("string", set())  # type: ignore[arg-type]
        assert not _is_code_execution_related_block(None, set())  # type: ignore[arg-type]
        assert not _is_code_execution_related_block(123, set())  # type: ignore[arg-type]

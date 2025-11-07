"""Unit tests for Anthropic text editor and memory tool middleware."""

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from langchain_anthropic.middleware.anthropic_tools import (
    AnthropicToolsState,
    StateClaudeMemoryMiddleware,
    StateClaudeTextEditorMiddleware,
    _validate_path,
)


class TestPathValidation:
    """Test path validation and security."""

    def test_basic_path_normalization(self) -> None:
        """Test basic path normalization."""
        assert _validate_path("/foo/bar") == "/foo/bar"
        assert _validate_path("foo/bar") == "/foo/bar"
        assert _validate_path("/foo//bar") == "/foo/bar"
        assert _validate_path("/foo/./bar") == "/foo/bar"

    def test_path_traversal_blocked(self) -> None:
        """Test that path traversal attempts are blocked."""
        with pytest.raises(ValueError, match="Path traversal not allowed"):
            _validate_path("/foo/../etc/passwd")

        with pytest.raises(ValueError, match="Path traversal not allowed"):
            _validate_path("../etc/passwd")

        with pytest.raises(ValueError, match="Path traversal not allowed"):
            _validate_path("~/.ssh/id_rsa")

    def test_allowed_prefixes(self) -> None:
        """Test path prefix validation."""
        # Should pass
        assert (
            _validate_path("/workspace/file.txt", allowed_prefixes=["/workspace"])
            == "/workspace/file.txt"
        )

        # Should fail
        with pytest.raises(ValueError, match="Path must start with"):
            _validate_path("/etc/passwd", allowed_prefixes=["/workspace"])

        with pytest.raises(ValueError, match="Path must start with"):
            _validate_path(
                "/workspacemalicious/file.txt", allowed_prefixes=["/workspace/"]
            )

    def test_memories_prefix(self) -> None:
        """Test /memories prefix validation for memory tools."""
        assert (
            _validate_path("/memories/notes.txt", allowed_prefixes=["/memories"])
            == "/memories/notes.txt"
        )

        with pytest.raises(ValueError, match="Path must start with"):
            _validate_path("/other/notes.txt", allowed_prefixes=["/memories"])


class TestTextEditorMiddleware:
    """Test text editor middleware functionality."""

    def test_middleware_initialization(self) -> None:
        """Test middleware initializes correctly."""
        middleware = StateClaudeTextEditorMiddleware()
        assert middleware.state_schema == AnthropicToolsState
        assert middleware.tool_type == "text_editor_20250728"
        assert middleware.tool_name == "str_replace_based_edit_tool"
        assert middleware.state_key == "text_editor_files"

        # With path restrictions
        middleware = StateClaudeTextEditorMiddleware(
            allowed_path_prefixes=["/workspace"]
        )
        assert middleware.allowed_prefixes == ["/workspace"]


class TestMemoryMiddleware:
    """Test memory middleware functionality."""

    def test_middleware_initialization(self) -> None:
        """Test middleware initializes correctly."""
        middleware = StateClaudeMemoryMiddleware()
        assert middleware.state_schema == AnthropicToolsState
        assert middleware.tool_type == "memory_20250818"
        assert middleware.tool_name == "memory"
        assert middleware.state_key == "memory_files"
        assert middleware.system_prompt  # Should have default prompt

    def test_custom_system_prompt(self) -> None:
        """Test custom system prompt can be set."""
        custom_prompt = "Custom memory instructions"
        middleware = StateClaudeMemoryMiddleware(system_prompt=custom_prompt)
        assert middleware.system_prompt == custom_prompt


class TestFileOperations:
    """Test file operation implementations via wrap_tool_call."""

    def test_view_operation(self) -> None:
        """Test view command execution."""
        middleware = StateClaudeTextEditorMiddleware()

        state: AnthropicToolsState = {
            "messages": [],
            "text_editor_files": {
                "/test.txt": {
                    "content": ["line1", "line2", "line3"],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                }
            },
        }

        args = {"command": "view", "path": "/test.txt"}
        result = middleware._handle_view(args, state, "test_id")

        assert isinstance(result, Command)
        assert result.update is not None
        messages = result.update.get("messages", [])
        assert len(messages) == 1
        assert isinstance(messages[0], ToolMessage)
        assert messages[0].content == "1|line1\n2|line2\n3|line3"
        assert messages[0].tool_call_id == "test_id"

    def test_create_operation(self) -> None:
        """Test create command execution."""
        middleware = StateClaudeTextEditorMiddleware()

        state: AnthropicToolsState = {"messages": []}

        args = {"command": "create", "path": "/test.txt", "file_text": "line1\nline2"}
        result = middleware._handle_create(args, state, "test_id")

        assert isinstance(result, Command)
        assert result.update is not None
        files = result.update.get("text_editor_files", {})
        assert "/test.txt" in files
        assert files["/test.txt"]["content"] == ["line1", "line2"]

    def test_path_prefix_enforcement(self) -> None:
        """Test that path prefixes are enforced."""
        middleware = StateClaudeTextEditorMiddleware(
            allowed_path_prefixes=["/workspace"]
        )

        state: AnthropicToolsState = {"messages": []}

        # Should fail with /etc/passwd
        args = {"command": "create", "path": "/etc/passwd", "file_text": "test"}

        with pytest.raises(ValueError, match="Path must start with"):
            middleware._handle_create(args, state, "test_id")

    def test_memories_prefix_enforcement(self) -> None:
        """Test that /memories prefix is enforced for memory middleware."""
        middleware = StateClaudeMemoryMiddleware()

        state: AnthropicToolsState = {"messages": []}

        # Should fail with /other/path
        args = {"command": "create", "path": "/other/path.txt", "file_text": "test"}

        with pytest.raises(ValueError, match="/memories"):
            middleware._handle_create(args, state, "test_id")

    def test_str_replace_operation(self) -> None:
        """Test str_replace command execution."""
        middleware = StateClaudeTextEditorMiddleware()

        state: AnthropicToolsState = {
            "messages": [],
            "text_editor_files": {
                "/test.txt": {
                    "content": ["Hello world", "Goodbye world"],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                }
            },
        }

        args = {
            "command": "str_replace",
            "path": "/test.txt",
            "old_str": "world",
            "new_str": "universe",
        }
        result = middleware._handle_str_replace(args, state, "test_id")

        assert isinstance(result, Command)
        assert result.update is not None
        files = result.update.get("text_editor_files", {})
        # Should only replace first occurrence
        assert files["/test.txt"]["content"] == ["Hello universe", "Goodbye world"]

    def test_insert_operation(self) -> None:
        """Test insert command execution."""
        middleware = StateClaudeTextEditorMiddleware()

        state: AnthropicToolsState = {
            "messages": [],
            "text_editor_files": {
                "/test.txt": {
                    "content": ["line1", "line2"],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                }
            },
        }

        args = {
            "command": "insert",
            "path": "/test.txt",
            "insert_line": 0,
            "new_str": "inserted",
        }
        result = middleware._handle_insert(args, state, "test_id")

        assert isinstance(result, Command)
        assert result.update is not None
        files = result.update.get("text_editor_files", {})
        assert files["/test.txt"]["content"] == ["inserted", "line1", "line2"]

    def test_delete_operation(self) -> None:
        """Test delete command execution (memory only)."""
        middleware = StateClaudeMemoryMiddleware()

        state: AnthropicToolsState = {
            "messages": [],
            "memory_files": {
                "/memories/test.txt": {
                    "content": ["line1"],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                }
            },
        }

        args = {"command": "delete", "path": "/memories/test.txt"}
        result = middleware._handle_delete(args, state, "test_id")

        assert isinstance(result, Command)
        assert result.update is not None
        files = result.update.get("memory_files", {})
        # Deleted files are marked as None in state
        assert files.get("/memories/test.txt") is None

    def test_rename_operation(self) -> None:
        """Test rename command execution (memory only)."""
        middleware = StateClaudeMemoryMiddleware()

        state: AnthropicToolsState = {
            "messages": [],
            "memory_files": {
                "/memories/old.txt": {
                    "content": ["line1"],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                }
            },
        }

        args = {
            "command": "rename",
            "old_path": "/memories/old.txt",
            "new_path": "/memories/new.txt",
        }
        result = middleware._handle_rename(args, state, "test_id")

        assert isinstance(result, Command)
        assert result.update is not None
        files = result.update.get("memory_files", {})
        # Old path is marked as None (deleted)
        assert files.get("/memories/old.txt") is None
        # New path has the file data
        assert files.get("/memories/new.txt") is not None
        assert files["/memories/new.txt"]["content"] == ["line1"]

"""Unit tests for file search middleware."""

from pathlib import Path
from typing import Any

import pytest

from langchain_anthropic.middleware.anthropic_tools import AnthropicToolsState
from langchain_anthropic.middleware.file_search import (
    FilesystemFileSearchMiddleware,
    StateFileSearchMiddleware,
)


class TestSearchMiddlewareInitialization:
    """Test search middleware initialization."""

    def test_middleware_initialization(self) -> None:
        """Test middleware initializes correctly."""
        middleware = StateFileSearchMiddleware()
        assert middleware.state_schema == AnthropicToolsState
        assert middleware.state_key == "text_editor_files"

    def test_custom_state_key(self) -> None:
        """Test middleware with custom state key."""
        middleware = StateFileSearchMiddleware(state_key="memory_files")
        assert middleware.state_key == "memory_files"


class TestGlobSearch:
    """Test Glob file pattern matching."""

    def test_glob_basic_pattern(self) -> None:
        """Test basic glob pattern matching."""
        middleware = StateFileSearchMiddleware()

        test_state: AnthropicToolsState = {
            "messages": [],
            "text_editor_files": {
                "/src/main.py": {
                    "content": ["print('hello')"],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                },
                "/src/utils.py": {
                    "content": ["def helper(): pass"],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                },
                "/README.md": {
                    "content": ["# Project"],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                },
            },
        }

        # Call tool function directly (state is injected in real usage)
        result = middleware.glob_search.func(pattern="*.py", state=test_state)

        assert isinstance(result, str)
        assert "/src/main.py" in result
        assert "/src/utils.py" in result
        assert "/README.md" not in result

    def test_glob_recursive_pattern(self) -> None:
        """Test recursive glob pattern matching."""
        middleware = StateFileSearchMiddleware()

        state: AnthropicToolsState = {
            "messages": [],
            "text_editor_files": {
                "/src/main.py": {
                    "content": [],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                },
                "/src/utils/helper.py": {
                    "content": [],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                },
                "/tests/test_main.py": {
                    "content": [],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                },
            },
        }

        result = middleware.glob_search.func(pattern="**/*.py", state=state)

        assert isinstance(result, str)
        lines = result.split("\n")
        assert len(lines) == 3
        assert all(".py" in line for line in lines)

    def test_glob_with_base_path(self) -> None:
        """Test glob with base path restriction."""
        middleware = StateFileSearchMiddleware()

        state: AnthropicToolsState = {
            "messages": [],
            "text_editor_files": {
                "/src/main.py": {
                    "content": [],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                },
                "/tests/test.py": {
                    "content": [],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                },
            },
        }

        result = middleware.glob_search.func(
            pattern="**/*.py", path="/src", state=state
        )

        assert isinstance(result, str)
        assert "/src/main.py" in result
        assert "/tests/test.py" not in result

    def test_glob_no_matches(self) -> None:
        """Test glob with no matching files."""
        middleware = StateFileSearchMiddleware()

        state: AnthropicToolsState = {
            "messages": [],
            "text_editor_files": {
                "/src/main.py": {
                    "content": [],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                },
            },
        }

        result = middleware.glob_search.func(pattern="*.ts", state=state)

        assert isinstance(result, str)
        assert result == "No files found"

    def test_glob_sorts_by_modified_time(self) -> None:
        """Test that glob results are sorted by modification time."""
        middleware = StateFileSearchMiddleware()

        state: AnthropicToolsState = {
            "messages": [],
            "text_editor_files": {
                "/old.py": {
                    "content": [],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                },
                "/new.py": {
                    "content": [],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-02T00:00:00",
                },
            },
        }

        result = middleware.glob_search.func(pattern="*.py", state=state)

        lines = result.split("\n")
        # Most recent first
        assert lines[0] == "/new.py"
        assert lines[1] == "/old.py"


class TestGrepSearch:
    """Test Grep content search."""

    def test_grep_files_with_matches_mode(self) -> None:
        """Test grep with files_with_matches output mode."""
        middleware = StateFileSearchMiddleware()

        state: AnthropicToolsState = {
            "messages": [],
            "text_editor_files": {
                "/src/main.py": {
                    "content": ["def foo():", "    pass"],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                },
                "/src/utils.py": {
                    "content": ["def bar():", "    return None"],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                },
                "/README.md": {
                    "content": ["# Documentation", "No code here"],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                },
            },
        }

        result = middleware.grep_search.func(pattern=r"def \w+\(\):", state=state)

        assert isinstance(result, str)
        assert "/src/main.py" in result
        assert "/src/utils.py" in result
        assert "/README.md" not in result
        # Should only have file paths, not line content

    def test_grep_invalid_include_pattern(self) -> None:
        """Return error when include glob is invalid."""
        middleware = StateFileSearchMiddleware()

        state: AnthropicToolsState = {
            "messages": [],
            "text_editor_files": {
                "/src/main.py": {
                    "content": ["def foo():"],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                }
            },
        }

        result = middleware.grep_search.func(
            pattern=r"def", include="*.{py", state=state
        )

        assert result == "Invalid include pattern"


class TestFilesystemGrepSearch:
    """Tests for filesystem-backed grep search."""

    def test_grep_invalid_include_pattern(self, tmp_path: Path) -> None:
        """Return error when include glob cannot be parsed."""

        (tmp_path / "example.py").write_text("print('hello')\n", encoding="utf-8")

        middleware = FilesystemFileSearchMiddleware(
            root_path=str(tmp_path), use_ripgrep=False
        )

        result = middleware.grep_search.func(pattern="print", include="*.{py")

        assert result == "Invalid include pattern"

    def test_ripgrep_command_uses_literal_pattern(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ensure ripgrep receives pattern after ``--`` to avoid option parsing."""

        (tmp_path / "example.py").write_text("print('hello')\n", encoding="utf-8")

        middleware = FilesystemFileSearchMiddleware(
            root_path=str(tmp_path), use_ripgrep=True
        )

        captured: dict[str, list[str]] = {}

        class DummyResult:
            stdout = ""

        def fake_run(*args: Any, **kwargs: Any) -> DummyResult:
            cmd = args[0]
            captured["cmd"] = cmd
            return DummyResult()

        monkeypatch.setattr(
            "langchain_anthropic.middleware.file_search.subprocess.run", fake_run
        )

        middleware._ripgrep_search("--pattern", "/", None)

        assert "cmd" in captured
        cmd = captured["cmd"]
        assert cmd[:2] == ["rg", "--json"]
        assert "--" in cmd
        separator_index = cmd.index("--")
        assert cmd[separator_index + 1] == "--pattern"

    def test_grep_content_mode(self) -> None:
        """Test grep with content output mode."""
        middleware = StateFileSearchMiddleware()

        state: AnthropicToolsState = {
            "messages": [],
            "text_editor_files": {
                "/src/main.py": {
                    "content": ["def foo():", "    pass", "def bar():"],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                },
            },
        }

        result = middleware.grep_search.func(
            pattern=r"def \w+\(\):", output_mode="content", state=state
        )

        assert isinstance(result, str)
        lines = result.split("\n")
        assert len(lines) == 2
        assert lines[0] == "/src/main.py:1:def foo():"
        assert lines[1] == "/src/main.py:3:def bar():"

    def test_grep_count_mode(self) -> None:
        """Test grep with count output mode."""
        middleware = StateFileSearchMiddleware()

        state: AnthropicToolsState = {
            "messages": [],
            "text_editor_files": {
                "/src/main.py": {
                    "content": ["TODO: fix this", "print('hello')", "TODO: add tests"],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                },
                "/src/utils.py": {
                    "content": ["TODO: implement"],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                },
            },
        }

        result = middleware.grep_search.func(
            pattern=r"TODO", output_mode="count", state=state
        )

        assert isinstance(result, str)
        lines = result.split("\n")
        assert "/src/main.py:2" in lines
        assert "/src/utils.py:1" in lines

    def test_grep_with_include_filter(self) -> None:
        """Test grep with include file pattern filter."""
        middleware = StateFileSearchMiddleware()

        state: AnthropicToolsState = {
            "messages": [],
            "text_editor_files": {
                "/src/main.py": {
                    "content": ["import os"],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                },
                "/src/main.ts": {
                    "content": ["import os from 'os'"],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                },
            },
        }

        result = middleware.grep_search.func(
            pattern="import", include="*.py", state=state
        )

        assert isinstance(result, str)
        assert "/src/main.py" in result
        assert "/src/main.ts" not in result

    def test_grep_with_brace_expansion_filter(self) -> None:
        """Test grep with brace expansion in include filter."""
        middleware = StateFileSearchMiddleware()

        state: AnthropicToolsState = {
            "messages": [],
            "text_editor_files": {
                "/src/main.ts": {
                    "content": ["const x = 1"],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                },
                "/src/App.tsx": {
                    "content": ["const y = 2"],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                },
                "/src/main.py": {
                    "content": ["z = 3"],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                },
            },
        }

        result = middleware.grep_search.func(
            pattern="const", include="*.{ts,tsx}", state=state
        )

        assert isinstance(result, str)
        assert "/src/main.ts" in result
        assert "/src/App.tsx" in result
        assert "/src/main.py" not in result

    def test_grep_with_base_path(self) -> None:
        """Test grep with base path restriction."""
        middleware = StateFileSearchMiddleware()

        state: AnthropicToolsState = {
            "messages": [],
            "text_editor_files": {
                "/src/main.py": {
                    "content": ["import foo"],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                },
                "/tests/test.py": {
                    "content": ["import foo"],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                },
            },
        }

        result = middleware.grep_search.func(pattern="import", path="/src", state=state)

        assert isinstance(result, str)
        assert "/src/main.py" in result
        assert "/tests/test.py" not in result

    def test_grep_no_matches(self) -> None:
        """Test grep with no matching content."""
        middleware = StateFileSearchMiddleware()

        state: AnthropicToolsState = {
            "messages": [],
            "text_editor_files": {
                "/src/main.py": {
                    "content": ["print('hello')"],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                },
            },
        }

        result = middleware.grep_search.func(pattern=r"TODO", state=state)

        assert isinstance(result, str)
        assert result == "No matches found"

    def test_grep_invalid_regex(self) -> None:
        """Test grep with invalid regex pattern."""
        middleware = StateFileSearchMiddleware()

        state: AnthropicToolsState = {
            "messages": [],
            "text_editor_files": {},
        }

        result = middleware.grep_search.func(pattern=r"[unclosed", state=state)

        assert isinstance(result, str)
        assert "Invalid regex pattern" in result


class TestSearchWithDifferentBackends:
    """Test searching with different backend configurations."""

    def test_glob_default_backend(self) -> None:
        """Test that glob searches the default backend (text_editor_files)."""
        middleware = StateFileSearchMiddleware()

        state: AnthropicToolsState = {
            "messages": [],
            "text_editor_files": {
                "/src/main.py": {
                    "content": [],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                },
            },
            "memory_files": {
                "/memories/notes.txt": {
                    "content": [],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                },
            },
        }

        result = middleware.glob_search.func(pattern="**/*", state=state)

        assert isinstance(result, str)
        assert "/src/main.py" in result
        # Should NOT find memory_files since default backend is text_editor_files
        assert "/memories/notes.txt" not in result

    def test_grep_default_backend(self) -> None:
        """Test that grep searches the default backend (text_editor_files)."""
        middleware = StateFileSearchMiddleware()

        state: AnthropicToolsState = {
            "messages": [],
            "text_editor_files": {
                "/src/main.py": {
                    "content": ["TODO: implement"],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                },
            },
            "memory_files": {
                "/memories/tasks.txt": {
                    "content": ["TODO: review"],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                },
            },
        }

        result = middleware.grep_search.func(pattern=r"TODO", state=state)

        assert isinstance(result, str)
        assert "/src/main.py" in result
        # Should NOT find memory_files since default backend is text_editor_files
        assert "/memories/tasks.txt" not in result

    def test_search_with_single_store(self) -> None:
        """Test searching with a specific state key."""
        middleware = StateFileSearchMiddleware(state_key="text_editor_files")

        state: AnthropicToolsState = {
            "messages": [],
            "text_editor_files": {
                "/src/main.py": {
                    "content": ["code"],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                },
            },
            "memory_files": {
                "/memories/notes.txt": {
                    "content": ["notes"],
                    "created_at": "2025-01-01T00:00:00",
                    "modified_at": "2025-01-01T00:00:00",
                },
            },
        }

        result = middleware.grep_search.func(pattern=r".*", state=state)

        assert isinstance(result, str)
        assert "/src/main.py" in result
        assert "/memories/notes.txt" not in result

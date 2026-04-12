"""Unit tests for file search middleware."""

from pathlib import Path
from typing import Any

import pytest
from langchain_core.tools import StructuredTool

from langchain.agents.middleware.file_search import (
    FilesystemFileSearchMiddleware,
    _expand_include_patterns,
    _is_valid_include_pattern,
    _match_include_pattern,
)


class TestFilesystemGrepSearch:
    """Tests for filesystem-backed grep search."""

    def test_grep_invalid_include_pattern(self, tmp_path: Path) -> None:
        """Return error when include glob cannot be parsed."""
        (tmp_path / "example.py").write_text("print('hello')\n", encoding="utf-8")

        middleware = FilesystemFileSearchMiddleware(root_path=str(tmp_path), use_ripgrep=False)

        assert isinstance(middleware.grep_search, StructuredTool)
        assert middleware.grep_search.func is not None
        result = middleware.grep_search.func(pattern="print", include="*.{py")

        assert result == "Invalid include pattern"

    def test_ripgrep_command_uses_literal_pattern(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ensure ripgrep receives pattern after ``--`` to avoid option parsing."""
        (tmp_path / "example.py").write_text("print('hello')\n", encoding="utf-8")

        middleware = FilesystemFileSearchMiddleware(root_path=str(tmp_path), use_ripgrep=True)

        captured: dict[str, list[str]] = {}

        class DummyResult:
            stdout = ""

        def fake_run(*args: Any, **_kwargs: Any) -> DummyResult:
            cmd = args[0]
            captured["cmd"] = cmd
            return DummyResult()

        monkeypatch.setattr("langchain.agents.middleware.file_search.subprocess.run", fake_run)

        middleware._ripgrep_search("--pattern", "/", None)

        assert "cmd" in captured
        cmd = captured["cmd"]
        assert cmd[:2] == ["rg", "--json"]
        assert "--" in cmd
        separator_index = cmd.index("--")
        assert cmd[separator_index + 1] == "--pattern"

    def test_grep_basic_search_python_fallback(self, tmp_path: Path) -> None:
        """Test basic grep search using Python fallback."""
        (tmp_path / "file1.py").write_text("def hello():\n    pass\n", encoding="utf-8")
        (tmp_path / "file2.py").write_text("def world():\n    pass\n", encoding="utf-8")
        (tmp_path / "file3.txt").write_text("hello world\n", encoding="utf-8")

        middleware = FilesystemFileSearchMiddleware(root_path=str(tmp_path), use_ripgrep=False)

        assert isinstance(middleware.grep_search, StructuredTool)
        assert middleware.grep_search.func is not None
        result = middleware.grep_search.func(pattern="hello")

        assert "/file1.py" in result
        assert "/file3.txt" in result
        assert "/file2.py" not in result

    def test_grep_with_include_filter(self, tmp_path: Path) -> None:
        """Test grep search with include pattern filter."""
        (tmp_path / "file1.py").write_text("def hello():\n    pass\n", encoding="utf-8")
        (tmp_path / "file2.txt").write_text("hello world\n", encoding="utf-8")

        middleware = FilesystemFileSearchMiddleware(root_path=str(tmp_path), use_ripgrep=False)

        assert isinstance(middleware.grep_search, StructuredTool)
        assert middleware.grep_search.func is not None
        result = middleware.grep_search.func(pattern="hello", include="*.py")

        assert "/file1.py" in result
        assert "/file2.txt" not in result

    def test_grep_output_mode_content(self, tmp_path: Path) -> None:
        """Test grep search with content output mode."""
        (tmp_path / "test.py").write_text("line1\nhello\nline3\n", encoding="utf-8")

        middleware = FilesystemFileSearchMiddleware(root_path=str(tmp_path), use_ripgrep=False)

        assert isinstance(middleware.grep_search, StructuredTool)
        assert middleware.grep_search.func is not None
        result = middleware.grep_search.func(pattern="hello", output_mode="content")

        assert "/test.py:2:hello" in result

    def test_grep_output_mode_count(self, tmp_path: Path) -> None:
        """Test grep search with count output mode."""
        (tmp_path / "test.py").write_text("hello\nhello\nworld\n", encoding="utf-8")

        middleware = FilesystemFileSearchMiddleware(root_path=str(tmp_path), use_ripgrep=False)

        assert isinstance(middleware.grep_search, StructuredTool)
        assert middleware.grep_search.func is not None
        result = middleware.grep_search.func(pattern="hello", output_mode="count")

        assert "/test.py:2" in result

    def test_grep_invalid_regex_pattern(self, tmp_path: Path) -> None:
        """Test grep search with invalid regex pattern."""
        (tmp_path / "test.py").write_text("hello\n", encoding="utf-8")

        middleware = FilesystemFileSearchMiddleware(root_path=str(tmp_path), use_ripgrep=False)

        assert isinstance(middleware.grep_search, StructuredTool)
        assert middleware.grep_search.func is not None
        result = middleware.grep_search.func(pattern="[invalid")

        assert "Invalid regex pattern" in result

    def test_grep_no_matches(self, tmp_path: Path) -> None:
        """Test grep search with no matches."""
        (tmp_path / "test.py").write_text("hello\n", encoding="utf-8")

        middleware = FilesystemFileSearchMiddleware(root_path=str(tmp_path), use_ripgrep=False)

        assert isinstance(middleware.grep_search, StructuredTool)
        assert middleware.grep_search.func is not None
        result = middleware.grep_search.func(pattern="notfound")

        assert result == "No matches found"


class TestFilesystemGlobSearch:
    """Tests for filesystem-backed glob search."""

    def test_glob_basic_pattern(self, tmp_path: Path) -> None:
        """Test basic glob pattern matching."""
        (tmp_path / "file1.py").write_text("content", encoding="utf-8")
        (tmp_path / "file2.py").write_text("content", encoding="utf-8")
        (tmp_path / "file3.txt").write_text("content", encoding="utf-8")

        middleware = FilesystemFileSearchMiddleware(root_path=str(tmp_path))

        assert isinstance(middleware.glob_search, StructuredTool)
        assert middleware.glob_search.func is not None
        result = middleware.glob_search.func(pattern="*.py")

        assert "/file1.py" in result
        assert "/file2.py" in result
        assert "/file3.txt" not in result

    def test_glob_recursive_pattern(self, tmp_path: Path) -> None:
        """Test recursive glob pattern matching."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "test.py").write_text("content", encoding="utf-8")
        (tmp_path / "src" / "nested").mkdir()
        (tmp_path / "src" / "nested" / "deep.py").write_text("content", encoding="utf-8")

        middleware = FilesystemFileSearchMiddleware(root_path=str(tmp_path))

        assert isinstance(middleware.glob_search, StructuredTool)
        assert middleware.glob_search.func is not None
        result = middleware.glob_search.func(pattern="**/*.py")

        assert "/src/test.py" in result
        assert "/src/nested/deep.py" in result

    def test_glob_with_subdirectory_path(self, tmp_path: Path) -> None:
        """Test glob search starting from subdirectory."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "file1.py").write_text("content", encoding="utf-8")
        (tmp_path / "other").mkdir()
        (tmp_path / "other" / "file2.py").write_text("content", encoding="utf-8")

        middleware = FilesystemFileSearchMiddleware(root_path=str(tmp_path))

        assert isinstance(middleware.glob_search, StructuredTool)
        assert middleware.glob_search.func is not None
        result = middleware.glob_search.func(pattern="*.py", path="/src")

        assert "/src/file1.py" in result
        assert "/other/file2.py" not in result

    def test_glob_no_matches(self, tmp_path: Path) -> None:
        """Test glob search with no matches."""
        (tmp_path / "file.txt").write_text("content", encoding="utf-8")

        middleware = FilesystemFileSearchMiddleware(root_path=str(tmp_path))

        assert isinstance(middleware.glob_search, StructuredTool)
        assert middleware.glob_search.func is not None
        result = middleware.glob_search.func(pattern="*.py")

        assert result == "No files found"

    def test_glob_invalid_path(self, tmp_path: Path) -> None:
        """Test glob search with non-existent path."""
        middleware = FilesystemFileSearchMiddleware(root_path=str(tmp_path))

        assert isinstance(middleware.glob_search, StructuredTool)
        assert middleware.glob_search.func is not None
        result = middleware.glob_search.func(pattern="*.py", path="/nonexistent")

        assert result == "No files found"


class TestPathTraversalSecurity:
    """Security tests for path traversal protection."""

    def test_path_traversal_with_double_dots(self, tmp_path: Path) -> None:
        """Test that path traversal with .. is blocked."""
        (tmp_path / "allowed").mkdir()
        (tmp_path / "allowed" / "file.txt").write_text("content", encoding="utf-8")

        # Create file outside root
        parent = tmp_path.parent
        (parent / "secret.txt").write_text("secret", encoding="utf-8")

        middleware = FilesystemFileSearchMiddleware(root_path=str(tmp_path / "allowed"))

        # Try to escape with ..
        assert isinstance(middleware.glob_search, StructuredTool)
        assert middleware.glob_search.func is not None
        result = middleware.glob_search.func(pattern="*.txt", path="/../")

        assert result == "No files found"
        assert "secret" not in result

    def test_path_traversal_with_absolute_path(self, tmp_path: Path) -> None:
        """Test that absolute paths outside root are blocked."""
        (tmp_path / "allowed").mkdir()

        # Create file outside root
        (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")

        middleware = FilesystemFileSearchMiddleware(root_path=str(tmp_path / "allowed"))

        # Try to access with absolute path
        assert isinstance(middleware.glob_search, StructuredTool)
        assert middleware.glob_search.func is not None
        result = middleware.glob_search.func(pattern="*.txt", path=str(tmp_path))

        assert result == "No files found"

    def test_path_traversal_with_symlink(self, tmp_path: Path) -> None:
        """Test that symlinks outside root are blocked."""
        (tmp_path / "allowed").mkdir()
        (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")

        # Create symlink from allowed dir to parent
        try:
            (tmp_path / "allowed" / "link").symlink_to(tmp_path)
        except OSError:
            pytest.skip("Symlink creation not supported")

        middleware = FilesystemFileSearchMiddleware(root_path=str(tmp_path / "allowed"))

        # Try to access via symlink
        assert isinstance(middleware.glob_search, StructuredTool)
        assert middleware.glob_search.func is not None
        result = middleware.glob_search.func(pattern="*.txt", path="/link")

        assert result == "No files found"

    def test_validate_path_blocks_tilde(self, tmp_path: Path) -> None:
        """Test that tilde paths are handled safely."""
        middleware = FilesystemFileSearchMiddleware(root_path=str(tmp_path))

        assert isinstance(middleware.glob_search, StructuredTool)
        assert middleware.glob_search.func is not None
        result = middleware.glob_search.func(pattern="*.txt", path="~/")

        assert result == "No files found"

    def test_grep_path_traversal_protection(self, tmp_path: Path) -> None:
        """Test that grep also protects against path traversal."""
        (tmp_path / "allowed").mkdir()
        (tmp_path / "secret.txt").write_text("secret content", encoding="utf-8")

        middleware = FilesystemFileSearchMiddleware(
            root_path=str(tmp_path / "allowed"), use_ripgrep=False
        )

        # Try to search outside root
        assert isinstance(middleware.grep_search, StructuredTool)
        assert middleware.grep_search.func is not None
        result = middleware.grep_search.func(pattern="secret", path="/../")

        assert result == "No matches found"
        assert "secret" not in result


class TestExpandIncludePatterns:
    """Tests for _expand_include_patterns helper function."""

    def test_expand_patterns_basic_brace_expansion(self) -> None:
        """Test basic brace expansion with multiple options."""
        result = _expand_include_patterns("*.{py,txt}")
        assert result == ["*.py", "*.txt"]

    def test_expand_patterns_nested_braces(self) -> None:
        """Test nested brace expansion."""
        result = _expand_include_patterns("test.{a,b}.{c,d}")
        assert result is not None
        assert len(result) == 4
        assert "test.a.c" in result
        assert "test.b.d" in result

    @pytest.mark.parametrize(
        "pattern",
        [
            "*.py}",  # closing brace without opening
            "*.{}",  # empty braces
            "*.{py",  # unclosed brace
        ],
    )
    def test_expand_patterns_invalid_braces(self, pattern: str) -> None:
        """Test patterns with invalid brace syntax return None."""
        result = _expand_include_patterns(pattern)
        assert result is None


class TestValidateIncludePattern:
    """Tests for _is_valid_include_pattern helper function."""

    @pytest.mark.parametrize(
        "pattern",
        [
            "",  # empty pattern
            "*.py\x00",  # null byte
            "*.py\n",  # newline
        ],
    )
    def test_validate_invalid_patterns(self, pattern: str) -> None:
        """Test that invalid patterns are rejected."""
        assert not _is_valid_include_pattern(pattern)


class TestMatchIncludePattern:
    """Tests for _match_include_pattern helper function."""

    def test_match_pattern_with_braces(self) -> None:
        """Test matching with brace expansion."""
        assert _match_include_pattern("test.py", "*.{py,txt}")
        assert _match_include_pattern("test.txt", "*.{py,txt}")
        assert not _match_include_pattern("test.md", "*.{py,txt}")

    def test_match_pattern_invalid_expansion(self) -> None:
        """Test matching with pattern that cannot be expanded returns False."""
        assert not _match_include_pattern("test.py", "*.{}")


class TestGrepEdgeCases:
    """Tests for edge cases in grep search."""

    def test_grep_with_special_chars_in_pattern(self, tmp_path: Path) -> None:
        """Test grep with special characters in pattern."""
        (tmp_path / "test.py").write_text("def test():\n    pass\n", encoding="utf-8")

        middleware = FilesystemFileSearchMiddleware(root_path=str(tmp_path), use_ripgrep=False)

        assert isinstance(middleware.grep_search, StructuredTool)
        assert middleware.grep_search.func is not None
        result = middleware.grep_search.func(pattern="def.*:")

        assert "/test.py" in result

    def test_grep_case_insensitive(self, tmp_path: Path) -> None:
        """Test grep with case-insensitive search."""
        (tmp_path / "test.py").write_text("HELLO world\n", encoding="utf-8")

        middleware = FilesystemFileSearchMiddleware(root_path=str(tmp_path), use_ripgrep=False)

        assert isinstance(middleware.grep_search, StructuredTool)
        assert middleware.grep_search.func is not None
        result = middleware.grep_search.func(pattern="(?i)hello")

        assert "/test.py" in result

    def test_grep_with_large_file_skipping(self, tmp_path: Path) -> None:
        """Test that grep skips files larger than max_file_size_mb."""
        # Create a file larger than 1MB
        large_content = "x" * (2 * 1024 * 1024)  # 2MB
        (tmp_path / "large.txt").write_text(large_content, encoding="utf-8")
        (tmp_path / "small.txt").write_text("x", encoding="utf-8")

        middleware = FilesystemFileSearchMiddleware(
            root_path=str(tmp_path),
            use_ripgrep=False,
            max_file_size_mb=1,  # 1MB limit
        )

        assert isinstance(middleware.grep_search, StructuredTool)
        assert middleware.grep_search.func is not None
        result = middleware.grep_search.func(pattern="x")

        # Large file should be skipped
        assert "/small.txt" in result

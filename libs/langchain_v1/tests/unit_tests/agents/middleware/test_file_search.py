"""Unit tests for filesystem file search middleware."""

from pathlib import Path
from typing import Any

import pytest

from langchain.agents.middleware.file_search import FilesystemFileSearchMiddleware


class TestFilesystemGrepSearch:
    """Tests for filesystem-backed grep search."""

    def test_ripgrep_command_uses_literal_pattern(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ensure ripgrep receives pattern after ``--`` to avoid option parsing."""
        (tmp_path / "example.py").write_text("print('hello')\n", encoding="utf-8")

        middleware = FilesystemFileSearchMiddleware(root_path=str(tmp_path), use_ripgrep=True)

        captured: dict[str, list[str]] = {}

        class DummyResult:
            stdout = ""

        def fake_run(*args: Any, **kwargs: Any) -> DummyResult:
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

    def test_filesystem_glob_basic(self, tmp_path: Path) -> None:
        """Test basic filesystem glob search."""
        (tmp_path / "main.py").write_text("print('hello')\n", encoding="utf-8")
        (tmp_path / "utils.py").write_text("def helper(): pass\n", encoding="utf-8")
        (tmp_path / "README.md").write_text("# Project\n", encoding="utf-8")

        middleware = FilesystemFileSearchMiddleware(root_path=str(tmp_path))
        result = middleware.glob_search.func(pattern="*.py")

        assert isinstance(result, str)
        assert "/main.py" in result
        assert "/utils.py" in result
        assert "/README.md" not in result

    def test_filesystem_grep_python_fallback(self, tmp_path: Path) -> None:
        """Test filesystem grep with Python fallback (no ripgrep)."""
        (tmp_path / "main.py").write_text("def foo():\n    pass\n", encoding="utf-8")
        (tmp_path / "utils.py").write_text("def bar():\n    return None\n", encoding="utf-8")

        middleware = FilesystemFileSearchMiddleware(root_path=str(tmp_path), use_ripgrep=False)
        result = middleware.grep_search.func(pattern=r"def \w+\(\):")

        assert isinstance(result, str)
        assert "/main.py" in result
        assert "/utils.py" in result

    def test_filesystem_grep_with_include(self, tmp_path: Path) -> None:
        """Test filesystem grep with include filter."""
        (tmp_path / "main.py").write_text("import os\n", encoding="utf-8")
        (tmp_path / "main.ts").write_text("import os from 'os'\n", encoding="utf-8")

        middleware = FilesystemFileSearchMiddleware(root_path=str(tmp_path), use_ripgrep=False)
        result = middleware.grep_search.func(pattern="import", include="*.py")

        assert isinstance(result, str)
        assert "/main.py" in result
        assert "/main.ts" not in result

    def test_filesystem_grep_content_mode(self, tmp_path: Path) -> None:
        """Test filesystem grep with content output mode."""
        (tmp_path / "main.py").write_text(
            "def foo():\n    pass\ndef bar():\n    pass\n", encoding="utf-8"
        )

        middleware = FilesystemFileSearchMiddleware(root_path=str(tmp_path), use_ripgrep=False)
        result = middleware.grep_search.func(pattern=r"def \w+\(\):", output_mode="content")

        assert isinstance(result, str)
        lines = result.split("\n")
        assert len(lines) == 2
        assert "/main.py:1:def foo():" in lines[0]
        assert "/main.py:3:def bar():" in lines[1]

    def test_filesystem_grep_count_mode(self, tmp_path: Path) -> None:
        """Test filesystem grep with count output mode."""
        (tmp_path / "main.py").write_text(
            "TODO: fix this\nprint('hello')\nTODO: add tests\n", encoding="utf-8"
        )
        (tmp_path / "utils.py").write_text("TODO: implement\n", encoding="utf-8")

        middleware = FilesystemFileSearchMiddleware(root_path=str(tmp_path), use_ripgrep=False)
        result = middleware.grep_search.func(pattern=r"TODO", output_mode="count")

        assert isinstance(result, str)
        lines = result.split("\n")
        assert "/main.py:2" in lines
        assert "/utils.py:1" in lines

    def test_filesystem_glob_no_matches(self, tmp_path: Path) -> None:
        """Test filesystem glob with no matches."""
        (tmp_path / "main.py").write_text("print('hello')\n", encoding="utf-8")

        middleware = FilesystemFileSearchMiddleware(root_path=str(tmp_path))
        result = middleware.glob_search.func(pattern="*.ts")

        assert result == "No files found"

    def test_filesystem_grep_no_matches(self, tmp_path: Path) -> None:
        """Test filesystem grep with no matches."""
        (tmp_path / "main.py").write_text("print('hello')\n", encoding="utf-8")

        middleware = FilesystemFileSearchMiddleware(root_path=str(tmp_path), use_ripgrep=False)
        result = middleware.grep_search.func(pattern=r"TODO")

        assert result == "No matches found"

    def test_filesystem_grep_invalid_regex(self, tmp_path: Path) -> None:
        """Test filesystem grep with invalid regex."""
        middleware = FilesystemFileSearchMiddleware(root_path=str(tmp_path), use_ripgrep=False)
        result = middleware.grep_search.func(pattern=r"[unclosed")

        assert "Invalid regex pattern" in result

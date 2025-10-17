"""Unit tests for file search middleware."""

from pathlib import Path
from typing import Any

import pytest

from langchain.agents.middleware.file_search import (
    FilesystemFileSearchMiddleware,
)


class TestFilesystemGrepSearch:
    """Tests for filesystem-backed grep search."""

    def test_grep_invalid_include_pattern(self, tmp_path: Path) -> None:
        """Return error when include glob cannot be parsed."""

        (tmp_path / "example.py").write_text("print('hello')\n", encoding="utf-8")

        middleware = FilesystemFileSearchMiddleware(root_path=str(tmp_path), use_ripgrep=False)

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

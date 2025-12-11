"""File search middleware for Anthropic text editor and memory tools.

This module provides Glob and Grep search tools that operate on files stored
in state or filesystem.
"""

from __future__ import annotations

import fnmatch
import json
import re
import subprocess
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from langchain_core.tools import tool

from langchain.agents.middleware.types import AgentMiddleware


def _expand_include_patterns(pattern: str) -> list[str] | None:
    """Expand brace patterns like `*.{py,pyi}` into a list of globs."""
    if "}" in pattern and "{" not in pattern:
        return None

    expanded: list[str] = []

    def _expand(current: str) -> None:
        start = current.find("{")
        if start == -1:
            expanded.append(current)
            return

        end = current.find("}", start)
        if end == -1:
            raise ValueError

        prefix = current[:start]
        suffix = current[end + 1 :]
        inner = current[start + 1 : end]
        if not inner:
            raise ValueError

        for option in inner.split(","):
            _expand(prefix + option + suffix)

    try:
        _expand(pattern)
    except ValueError:
        return None

    return expanded


def _is_valid_include_pattern(pattern: str) -> bool:
    """Validate glob pattern used for include filters."""
    if not pattern:
        return False

    if any(char in pattern for char in ("\x00", "\n", "\r")):
        return False

    expanded = _expand_include_patterns(pattern)
    if expanded is None:
        return False

    try:
        for candidate in expanded:
            re.compile(fnmatch.translate(candidate))
    except re.error:
        return False

    return True


def _match_include_pattern(basename: str, pattern: str) -> bool:
    """Return True if the basename matches the include pattern."""
    expanded = _expand_include_patterns(pattern)
    if not expanded:
        return False

    return any(fnmatch.fnmatch(basename, candidate) for candidate in expanded)


class FilesystemFileSearchMiddleware(AgentMiddleware):
    """Provides Glob and Grep search over filesystem files.

    This middleware adds two tools that search through local filesystem:

    - Glob: Fast file pattern matching by file path
    - Grep: Fast content search using ripgrep or Python fallback

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain.agents.middleware import (
            FilesystemFileSearchMiddleware,
        )

        agent = create_agent(
            model=model,
            tools=[],  # Add tools as needed
            middleware=[
                FilesystemFileSearchMiddleware(root_path="/workspace"),
            ],
        )
        ```
    """

    def __init__(
        self,
        *,
        root_path: str,
        use_ripgrep: bool = True,
        max_file_size_mb: int = 10,
    ) -> None:
        """Initialize the search middleware.

        Args:
            root_path: Root directory to search.
            use_ripgrep: Whether to use `ripgrep` for search.

                Falls back to Python if `ripgrep` unavailable.
            max_file_size_mb: Maximum file size to search in MB.
        """
        self.root_path = Path(root_path).resolve()
        self.use_ripgrep = use_ripgrep
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

        # Create tool instances as closures that capture self
        @tool
        def glob_search(pattern: str, path: str = "/") -> str:
            """Fast file pattern matching tool that works with any codebase size.

            Supports glob patterns like `**/*.js` or `src/**/*.ts`.

            Returns matching file paths sorted by modification time.

            Use this tool when you need to find files by name patterns.

            Args:
                pattern: The glob pattern to match files against.
                path: The directory to search in. If not specified, searches from root.

            Returns:
                Newline-separated list of matching file paths, sorted by modification
                time (most recently modified first). Returns `'No files found'` if no
                matches.
            """
            try:
                base_full = self._validate_and_resolve_path(path)
            except ValueError:
                return "No files found"

            if not base_full.exists() or not base_full.is_dir():
                return "No files found"

            # Use pathlib glob
            matching: list[tuple[str, str]] = []
            for match in base_full.glob(pattern):
                if match.is_file():
                    # Convert to virtual path
                    virtual_path = "/" + str(match.relative_to(self.root_path))
                    stat = match.stat()
                    modified_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
                    matching.append((virtual_path, modified_at))

            if not matching:
                return "No files found"

            file_paths = [p for p, _ in matching]
            return "\n".join(file_paths)

        @tool
        def grep_search(
            pattern: str,
            path: str = "/",
            include: str | None = None,
            output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches",
        ) -> str:
            """Fast content search tool that works with any codebase size.

            Searches file contents using regular expressions. Supports full regex
            syntax and filters files by pattern with the include parameter.

            Args:
                pattern: The regular expression pattern to search for in file contents.
                path: The directory to search in. If not specified, searches from root.
                include: File pattern to filter (e.g., `'*.js'`, `'*.{ts,tsx}'`).
                output_mode: Output format:

                    - `'files_with_matches'`: Only file paths containing matches
                    - `'content'`: Matching lines with `file:line:content` format
                    - `'count'`: Count of matches per file

            Returns:
                Search results formatted according to `output_mode`.
                    Returns `'No matches found'` if no results.
            """
            # Compile regex pattern (for validation)
            try:
                re.compile(pattern)
            except re.error as e:
                return f"Invalid regex pattern: {e}"

            if include and not _is_valid_include_pattern(include):
                return "Invalid include pattern"

            # Try ripgrep first if enabled
            results = None
            if self.use_ripgrep:
                with suppress(
                    FileNotFoundError,
                    subprocess.CalledProcessError,
                    subprocess.TimeoutExpired,
                ):
                    results = self._ripgrep_search(pattern, path, include)

            # Python fallback if ripgrep failed or is disabled
            if results is None:
                results = self._python_search(pattern, path, include)

            if not results:
                return "No matches found"

            # Format output based on mode
            return self._format_grep_results(results, output_mode)

        self.glob_search = glob_search
        self.grep_search = grep_search
        self.tools = [glob_search, grep_search]

    def _validate_and_resolve_path(self, path: str) -> Path:
        """Validate and resolve a virtual path to filesystem path."""
        # Normalize path
        if not path.startswith("/"):
            path = "/" + path

        # Check for path traversal
        if ".." in path or "~" in path:
            msg = "Path traversal not allowed"
            raise ValueError(msg)

        # Convert virtual path to filesystem path
        relative = path.lstrip("/")
        full_path = (self.root_path / relative).resolve()

        # Ensure path is within root
        try:
            full_path.relative_to(self.root_path)
        except ValueError:
            msg = f"Path outside root directory: {path}"
            raise ValueError(msg) from None

        return full_path

    def _ripgrep_search(
        self, pattern: str, base_path: str, include: str | None
    ) -> dict[str, list[tuple[int, str]]]:
        """Search using ripgrep subprocess."""
        try:
            base_full = self._validate_and_resolve_path(base_path)
        except ValueError:
            return {}

        if not base_full.exists():
            return {}

        # Build ripgrep command
        cmd = ["rg", "--json"]

        if include:
            # Convert glob pattern to ripgrep glob
            cmd.extend(["--glob", include])

        cmd.extend(["--", pattern, str(base_full)])

        try:
            result = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Fallback to Python search if ripgrep unavailable or times out
            return self._python_search(pattern, base_path, include)

        # Parse ripgrep JSON output
        results: dict[str, list[tuple[int, str]]] = {}
        for line in result.stdout.splitlines():
            try:
                data = json.loads(line)
                if data["type"] == "match":
                    path = data["data"]["path"]["text"]
                    # Convert to virtual path
                    virtual_path = "/" + str(Path(path).relative_to(self.root_path))
                    line_num = data["data"]["line_number"]
                    line_text = data["data"]["lines"]["text"].rstrip("\n")

                    if virtual_path not in results:
                        results[virtual_path] = []
                    results[virtual_path].append((line_num, line_text))
            except (json.JSONDecodeError, KeyError):
                continue

        return results

    def _python_search(
        self, pattern: str, base_path: str, include: str | None
    ) -> dict[str, list[tuple[int, str]]]:
        """Search using Python regex (fallback)."""
        try:
            base_full = self._validate_and_resolve_path(base_path)
        except ValueError:
            return {}

        if not base_full.exists():
            return {}

        regex = re.compile(pattern)
        results: dict[str, list[tuple[int, str]]] = {}

        # Walk directory tree
        for file_path in base_full.rglob("*"):
            if not file_path.is_file():
                continue

            # Check include filter
            if include and not _match_include_pattern(file_path.name, include):
                continue

            # Skip files that are too large
            if file_path.stat().st_size > self.max_file_size_bytes:
                continue

            try:
                content = file_path.read_text()
            except (UnicodeDecodeError, PermissionError):
                continue

            # Search content
            for line_num, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    virtual_path = "/" + str(file_path.relative_to(self.root_path))
                    if virtual_path not in results:
                        results[virtual_path] = []
                    results[virtual_path].append((line_num, line))

        return results

    def _format_grep_results(
        self,
        results: dict[str, list[tuple[int, str]]],
        output_mode: str,
    ) -> str:
        """Format grep results based on output mode."""
        if output_mode == "files_with_matches":
            # Just return file paths
            return "\n".join(sorted(results.keys()))

        if output_mode == "content":
            # Return file:line:content format
            lines = []
            for file_path in sorted(results.keys()):
                for line_num, line in results[file_path]:
                    lines.append(f"{file_path}:{line_num}:{line}")
            return "\n".join(lines)

        if output_mode == "count":
            # Return file:count format
            lines = []
            for file_path in sorted(results.keys()):
                count = len(results[file_path])
                lines.append(f"{file_path}:{count}")
            return "\n".join(lines)

        # Default to files_with_matches
        return "\n".join(sorted(results.keys()))


__all__ = [
    "FilesystemFileSearchMiddleware",
]

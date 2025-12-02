"""File search middleware for Anthropic text editor and memory tools.

This module provides Glob and Grep search tools that operate on files stored
in state or filesystem.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain.tools import ToolRuntime, tool

from langchain_anthropic.middleware.anthropic_tools import AnthropicToolsState


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
    """Return `True` if the basename matches the include pattern."""
    expanded = _expand_include_patterns(pattern)
    if not expanded:
        return False

    return any(fnmatch.fnmatch(basename, candidate) for candidate in expanded)


class StateFileSearchMiddleware(AgentMiddleware):
    """Provides Glob and Grep search over state-based files.

    This middleware adds two tools that search through virtual files in state:

    - Glob: Fast file pattern matching by file path
    - Grep: Fast content search using regular expressions

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain.agents.middleware import (
            StateTextEditorToolMiddleware,
            StateFileSearchMiddleware,
        )

        agent = create_agent(
            model=model,
            tools=[],
            middleware=[
                StateTextEditorToolMiddleware(),
                StateFileSearchMiddleware(),
            ],
        )
        ```
    """

    state_schema = AnthropicToolsState

    def __init__(
        self,
        *,
        state_key: str = "text_editor_files",
    ) -> None:
        """Initialize the search middleware.

        Args:
            state_key: State key to search

                Use `'memory_files'` to search memory tool files.
        """
        self.state_key = state_key

        # Create tool instances
        @tool
        def glob_search(  # noqa: D417
            runtime: ToolRuntime[None, AnthropicToolsState],
            pattern: str,
            path: str = "/",
        ) -> str:
            """Fast file pattern matching tool that works with any codebase size.

            Supports glob patterns like `**/*.js` or `src/**/*.ts`.

            Returns matching file paths sorted by modification time.

            Use this tool when you need to find files by name patterns.

            Args:
                pattern: The glob pattern to match files against.
                path: The directory to search in.

                    If not specified, searches from root.

            Returns:
                Newline-separated list of matching file paths, sorted by modification
                    time (most recently modified first).

                    Returns `'No files found'` if no matches.
            """
            return self._handle_glob_search(pattern, path, runtime.state)

        @tool
        def grep_search(  # noqa: D417
            runtime: ToolRuntime[None, AnthropicToolsState],
            pattern: str,
            path: str = "/",
            include: str | None = None,
            output_mode: Literal[
                "files_with_matches", "content", "count"
            ] = "files_with_matches",
        ) -> str:
            """Fast content search tool that works with any codebase size.

            Searches file contents using regular expressions.

            Supports full regex syntax and filters files by pattern with the include
            parameter.

            Args:
                pattern: The regular expression pattern to search for in file contents.
                path: The directory to search in. If not specified, searches from root.
                include: File pattern to filter (e.g., `'*.js'`, `'*.{ts,tsx}'`).
                output_mode: Output format.

                    Options:

                    - `'files_with_matches'`: Only file paths containing matches
                    - `'content'`: Matching lines with file:line:content format
                    - `'count'`: Count of matches per file

            Returns:
                Search results formatted according to `output_mode`.

                    Returns `'No matches found'` if no results.
            """
            return self._handle_grep_search(
                pattern, path, include, output_mode, runtime.state
            )

        self.glob_search = glob_search
        self.grep_search = grep_search
        self.tools = [glob_search, grep_search]

    def _handle_glob_search(
        self,
        pattern: str,
        path: str,
        state: AnthropicToolsState,
    ) -> str:
        """Handle glob search operation.

        Args:
            pattern: The glob pattern to match files against.
            path: The directory to search in.
            state: The current agent state.

        Returns:
            Newline-separated list of matching file paths, sorted by modification
                time (most recently modified first).

                Returns `'No files found'` if no matches.
        """
        # Normalize base path
        base_path = path if path.startswith("/") else "/" + path

        # Get files from state
        files = cast("dict[str, Any]", state.get(self.state_key, {}))

        # Match files
        matches = []
        for file_path, file_data in files.items():
            if file_path.startswith(base_path):
                # Get relative path from base
                if base_path == "/":
                    relative = file_path[1:]  # Remove leading /
                elif file_path == base_path:
                    relative = Path(file_path).name
                elif file_path.startswith(base_path + "/"):
                    relative = file_path[len(base_path) + 1 :]
                else:
                    continue

                # Match against pattern
                # Handle ** pattern which requires special care
                # PurePosixPath.match doesn't match single-level paths
                # against **/pattern
                is_match = PurePosixPath(relative).match(pattern)
                if not is_match and pattern.startswith("**/"):
                    # Also try matching without the **/ prefix for files in base dir
                    is_match = PurePosixPath(relative).match(pattern[3:])

                if is_match:
                    matches.append((file_path, file_data["modified_at"]))

        if not matches:
            return "No files found"

        # Sort by modification time
        matches.sort(key=lambda x: x[1], reverse=True)
        file_paths = [path for path, _ in matches]

        return "\n".join(file_paths)

    def _handle_grep_search(
        self,
        pattern: str,
        path: str,
        include: str | None,
        output_mode: str,
        state: AnthropicToolsState,
    ) -> str:
        """Handle grep search operation.

        Args:
            pattern: The regular expression pattern to search for in file contents.
            path: The directory to search in.
            include: File pattern to filter (e.g., `'*.js'`, `'*.{ts,tsx}'`).
            output_mode: Output format.
            state: The current agent state.

        Returns:
            Search results formatted according to `output_mode`.

                Returns `'No matches found'` if no results.
        """
        # Normalize base path
        base_path = path if path.startswith("/") else "/" + path

        # Compile regex pattern (for validation)
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Invalid regex pattern: {e}"

        if include and not _is_valid_include_pattern(include):
            return "Invalid include pattern"

        # Search files
        files = cast("dict[str, Any]", state.get(self.state_key, {}))
        results: dict[str, list[tuple[int, str]]] = {}

        for file_path, file_data in files.items():
            if not file_path.startswith(base_path):
                continue

            # Check include filter
            if include:
                basename = Path(file_path).name
                if not _match_include_pattern(basename, include):
                    continue

            # Search file content
            for line_num, line in enumerate(file_data["content"], 1):
                if regex.search(line):
                    if file_path not in results:
                        results[file_path] = []
                    results[file_path].append((line_num, line))

        if not results:
            return "No matches found"

        # Format output based on mode
        return self._format_grep_results(results, output_mode)

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
    "StateFileSearchMiddleware",
]

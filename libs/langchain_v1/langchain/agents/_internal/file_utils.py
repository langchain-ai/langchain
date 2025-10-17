"""Shared utility functions for file operations in middleware."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from collections.abc import Sequence

# Constants
MEMORIES_PREFIX = "/memories/"
EMPTY_CONTENT_WARNING = "System reminder: File exists but has empty contents"
MAX_LINE_LENGTH = 2000
LINE_NUMBER_WIDTH = 6


class FileData(TypedDict):
    """Data structure for storing file contents with metadata."""

    content: list[str]
    """Lines of the file."""

    created_at: str
    """ISO 8601 timestamp of file creation."""

    modified_at: str
    """ISO 8601 timestamp of last modification."""


def file_data_reducer(
    left: dict[str, FileData] | None, right: dict[str, FileData | None]
) -> dict[str, FileData]:
    """Merge file updates with support for deletions.

    This reducer enables file deletion by treating `None` values in the right
    dictionary as deletion markers. It's designed to work with LangGraph's
    state management where annotated reducers control how state updates merge.

    Args:
        left: Existing files dictionary. May be `None` during initialization.
        right: New files dictionary to merge. Files with `None` values are
            treated as deletion markers and removed from the result.

    Returns:
        Merged dictionary where right overwrites left for matching keys,
        and `None` values in right trigger deletions.

    Example:
        ```python
        existing = {"/file1.txt": FileData(...), "/file2.txt": FileData(...)}
        updates = {"/file2.txt": None, "/file3.txt": FileData(...)}
        result = file_data_reducer(existing, updates)
        # Result: {"/file1.txt": FileData(...), "/file3.txt": FileData(...)}
        ```
    """
    if left is None:
        # Filter out None values when initializing
        return {k: v for k, v in right.items() if v is not None}

    # Merge, filtering out None values (deletions)
    result = {**left}
    for key, value in right.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = value
    return result


def validate_path(path: str, *, allowed_prefixes: Sequence[str] | None = None) -> str:
    """Validate and normalize file path for security.

    Ensures paths are safe to use by preventing directory traversal attacks
    and enforcing consistent formatting. All paths are normalized to use
    forward slashes and start with a leading slash.

    Args:
        path: The path to validate and normalize.
        allowed_prefixes: Optional list of allowed path prefixes. If provided,
            the normalized path must start with one of these prefixes.

    Returns:
        Normalized canonical path starting with `/` and using forward slashes.

    Raises:
        ValueError: If path contains traversal sequences (`..` or `~`) or does
            not start with an allowed prefix when `allowed_prefixes` is specified.

    Example:
        ```python
        validate_path("foo/bar")  # Returns: "/foo/bar"
        validate_path("/./foo//bar")  # Returns: "/foo/bar"
        validate_path("../etc/passwd")  # Raises ValueError
        validate_path("/data/file.txt", allowed_prefixes=["/data/"])  # OK
        validate_path("/etc/file.txt", allowed_prefixes=["/data/"])  # Raises ValueError
        ```
    """
    # Reject paths with traversal attempts
    if ".." in path or path.startswith("~"):
        msg = f"Path traversal not allowed: {path}"
        raise ValueError(msg)

    # Normalize path (resolve ., //, etc.)
    normalized = os.path.normpath(path)

    # Convert to forward slashes for consistency
    normalized = normalized.replace("\\", "/")

    # Ensure path starts with /
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"

    # Check allowed prefixes if specified
    if allowed_prefixes is not None and not any(
        normalized.startswith(prefix) for prefix in allowed_prefixes
    ):
        msg = f"Path must start with one of {allowed_prefixes}: {path}"
        raise ValueError(msg)

    return normalized


def format_content_with_line_numbers(
    content: str | list[str],
    *,
    format_style: Literal["pipe", "tab"] = "pipe",
    start_line: int = 1,
) -> str:
    r"""Format file content with line numbers for display.

    Converts file content to a numbered format similar to `cat -n` output,
    with support for two different formatting styles.

    Args:
        content: File content as a string or list of lines.
        format_style: Format style for line numbers:
            - `"pipe"`: Compact format like `"1|content"`
            - `"tab"`: Right-aligned format like `"     1\tcontent"` (lines truncated at 2000 chars)
        start_line: Starting line number (default: 1).

    Returns:
        Formatted content with line numbers prepended to each line.

    Example:
        ```python
        content = "Hello\nWorld"
        format_content_with_line_numbers(content, format_style="pipe")
        # Returns: "1|Hello\n2|World"

        format_content_with_line_numbers(content, format_style="tab", start_line=10)
        # Returns: "    10\tHello\n    11\tWorld"
        ```
    """
    if isinstance(content, str):
        lines = content.split("\n")
        # Remove trailing empty line from split
        if lines and lines[-1] == "":
            lines = lines[:-1]
    else:
        lines = content

    if format_style == "pipe":
        return "\n".join(f"{i + start_line}|{line}" for i, line in enumerate(lines))

    # Tab format with defined width and line truncation
    return "\n".join(
        f"{i + start_line:{LINE_NUMBER_WIDTH}d}\t{line[:MAX_LINE_LENGTH]}"
        for i, line in enumerate(lines)
    )


def apply_string_replacement(
    content: str,
    old_string: str,
    new_string: str,
    *,
    replace_all: bool = False,
) -> tuple[str, int]:
    """Apply exact string replacement to content.

    Replaces occurrences of a string within content and returns both the
    modified content and the number of replacements made.

    Args:
        content: Original content to modify.
        old_string: String to find and replace.
        new_string: Replacement string.
        replace_all: If `True`, replace all occurrences. If `False`, replace
            only the first occurrence (default).

    Returns:
        Tuple of `(modified_content, replacement_count)`.

    Example:
        ```python
        content = "foo bar foo"
        apply_string_replacement(content, "foo", "baz", replace_all=False)
        # Returns: ("baz bar foo", 1)

        apply_string_replacement(content, "foo", "baz", replace_all=True)
        # Returns: ("baz bar baz", 2)
        ```
    """
    if replace_all:
        count = content.count(old_string)
        new_content = content.replace(old_string, new_string)
    else:
        count = 1 if old_string in content else 0
        new_content = content.replace(old_string, new_string, 1)

    return new_content, count


def create_file_data(
    content: str | list[str],
    *,
    created_at: str | None = None,
) -> FileData:
    r"""Create a FileData object with automatic timestamp generation.

    Args:
        content: File content as a string or list of lines.
        created_at: Optional creation timestamp in ISO 8601 format.
            If `None`, uses the current UTC time.

    Returns:
        FileData object with content and timestamps.

    Example:
        ```python
        file_data = create_file_data("Hello\nWorld")
        # Returns: {"content": ["Hello", "World"], "created_at": "2024-...",
        #           "modified_at": "2024-..."}
        ```
    """
    lines = content.split("\n") if isinstance(content, str) else content
    now = datetime.now(timezone.utc).isoformat()

    return {
        "content": lines,
        "created_at": created_at or now,
        "modified_at": now,
    }


def update_file_data(
    file_data: FileData,
    content: str | list[str],
) -> FileData:
    """Update FileData with new content while preserving creation timestamp.

    Args:
        file_data: Existing FileData object to update.
        content: New file content as a string or list of lines.

    Returns:
        Updated FileData object with new content and updated `modified_at`
        timestamp. The `created_at` timestamp is preserved from the original.

    Example:
        ```python
        original = create_file_data("Hello")
        updated = update_file_data(original, "Hello World")
        # updated["created_at"] == original["created_at"]
        # updated["modified_at"] > original["modified_at"]
        ```
    """
    lines = content.split("\n") if isinstance(content, str) else content
    now = datetime.now(timezone.utc).isoformat()

    return {
        "content": lines,
        "created_at": file_data["created_at"],
        "modified_at": now,
    }


def file_data_to_string(file_data: FileData) -> str:
    r"""Convert FileData to plain string content.

    Joins the lines stored in FileData with newline characters to produce
    a single string representation of the file content.

    Args:
        file_data: FileData object containing lines of content.

    Returns:
        File content as a single string with lines joined by newlines.

    Example:
        ```python
        file_data = {
            "content": ["Hello", "World"],
            "created_at": "...",
            "modified_at": "...",
        }
        file_data_to_string(file_data)  # Returns: "Hello\nWorld"
        ```
    """
    return "\n".join(file_data["content"])


def list_directory(files: dict[str, FileData], path: str) -> list[str]:
    """List files in a directory (direct children only).

    Returns only the direct children of the specified directory path,
    excluding files in subdirectories.

    Args:
        files: Dictionary mapping file paths to FileData objects.
        path: Normalized directory path to list files from.

    Returns:
        Sorted list of file paths that are direct children of the directory.

    Example:
        ```python
        files = {
            "/dir/file1.txt": FileData(...),
            "/dir/file2.txt": FileData(...),
            "/dir/subdir/file3.txt": FileData(...),
        }
        list_directory(files, "/dir")
        # Returns: ["/dir/file1.txt", "/dir/file2.txt"]
        # Note: /dir/subdir/file3.txt is excluded (not a direct child)
        ```
    """
    # Ensure path ends with / for directory matching
    dir_path = path if path.endswith("/") else f"{path}/"

    matching_files = []
    for file_path in files:
        if file_path.startswith(dir_path):
            # Get relative path from directory
            relative = file_path[len(dir_path) :]
            # Only include direct children (no subdirectories)
            if "/" not in relative:
                matching_files.append(file_path)

    return sorted(matching_files)


def check_empty_content(content: str) -> str | None:
    """Check if file content is empty and return a warning message.

    Args:
        content: File content to check.

    Returns:
        Warning message string if content is empty or contains only whitespace,
        `None` otherwise.

    Example:
        ```python
        check_empty_content("")  # Returns: "System reminder: File exists but has empty contents"
        check_empty_content("   ")  # Returns: "System reminder: File exists but has empty contents"
        check_empty_content("Hello")  # Returns: None
        ```
    """
    if not content or content.strip() == "":
        return EMPTY_CONTENT_WARNING
    return None


def has_memories_prefix(file_path: str) -> bool:
    """Check if a file path is in the longterm memory filesystem.

    Longterm memory files are distinguished by the `/memories/` path prefix.

    Args:
        file_path: File path to check.

    Returns:
        `True` if the file path starts with `/memories/`, `False` otherwise.

    Example:
        ```python
        has_memories_prefix("/memories/notes.txt")  # Returns: True
        has_memories_prefix("/temp/file.txt")  # Returns: False
        ```
    """
    return file_path.startswith(MEMORIES_PREFIX)


def append_memories_prefix(file_path: str) -> str:
    """Add the longterm memory prefix to a file path.

    Args:
        file_path: File path to prefix.

    Returns:
        File path with `/memories` prepended.

    Example:
        ```python
        append_memories_prefix("/notes.txt")  # Returns: "/memories/notes.txt"
        ```
    """
    return f"/memories{file_path}"


def strip_memories_prefix(file_path: str) -> str:
    """Remove the longterm memory prefix from a file path.

    Args:
        file_path: File path potentially containing the memories prefix.

    Returns:
        File path with `/memories` removed if present at the start.

    Example:
        ```python
        strip_memories_prefix("/memories/notes.txt")  # Returns: "/notes.txt"
        strip_memories_prefix("/notes.txt")  # Returns: "/notes.txt"
        ```
    """
    if file_path.startswith(MEMORIES_PREFIX):
        return file_path[len(MEMORIES_PREFIX) - 1 :]  # Keep the leading slash
    return file_path

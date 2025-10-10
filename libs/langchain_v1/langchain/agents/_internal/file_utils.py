"""Shared utility functions for file operations in middleware."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from collections.abc import Sequence


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
    """Custom reducer that merges file updates.

    Args:
        left: Existing files dict.
        right: New files dict to merge (None values delete files).

    Returns:
        Merged dict where right overwrites left for matching keys.
    """
    if left is None:
        # Filter out None values when initializing
        return {k: v for k, v in right.items() if v is not None}

    # Merge, filtering out None values (deletions)
    result = {**left}
    for k, v in right.items():
        if v is None:
            result.pop(k, None)
        else:
            result[k] = v
    return result


def validate_path(path: str, *, allowed_prefixes: Sequence[str] | None = None) -> str:
    """Validate and normalize file path for security.

    Args:
        path: The path to validate.
        allowed_prefixes: Optional list of allowed path prefixes.

    Returns:
        Normalized canonical path.

    Raises:
        ValueError: If path contains traversal sequences or violates prefix rules.
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
    r"""Format file content with line numbers.

    Args:
        content: File content as string or list of lines.
        format_style: "pipe" for "1|content" or "tab" for "     1\tcontent".
        start_line: Starting line number.

    Returns:
        Formatted content with line numbers.
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

    return "\n".join(f"{i + start_line:6d}\t{line[:2000]}" for i, line in enumerate(lines))


def apply_string_replacement(
    content: str,
    old_string: str,
    new_string: str,
    *,
    replace_all: bool = False,
) -> tuple[str, int]:
    """Apply string replacement to content.

    Args:
        content: Original content.
        old_string: String to replace.
        new_string: Replacement string.
        replace_all: If True, replace all occurrences. Otherwise, replace first.

    Returns:
        Tuple of (new_content, replacement_count).
    """
    if replace_all:
        count = content.count(old_string)
        new_content = content.replace(old_string, new_string)
    else:
        count = 1
        new_content = content.replace(old_string, new_string, 1)

    return new_content, count


def create_file_data(
    content: str | list[str],
    *,
    created_at: str | None = None,
) -> FileData:
    """Create a FileData object from content.

    Args:
        content: File content as string or list of lines.
        created_at: Optional creation timestamp. If None, uses current time.

    Returns:
        FileData object.
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
    """Update a FileData object with new content.

    Args:
        file_data: Existing FileData object.
        content: New file content as string or list of lines.

    Returns:
        Updated FileData object with new modified_at timestamp.
    """
    lines = content.split("\n") if isinstance(content, str) else content

    now = datetime.now(timezone.utc).isoformat()

    return {
        "content": lines,
        "created_at": file_data["created_at"],
        "modified_at": now,
    }


def file_data_to_string(file_data: FileData) -> str:
    """Convert FileData to plain string content.

    Args:
        file_data: FileData object.

    Returns:
        File content as string.
    """
    return "\n".join(file_data["content"])


def list_directory(files: dict[str, FileData], path: str) -> list[str]:
    """List files in a directory.

    Args:
        files: Files dict mapping paths to FileData.
        path: Normalized directory path.

    Returns:
        Sorted list of file paths in the directory.
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
    """Check if file content is empty and return warning message.

    Args:
        content: File content.

    Returns:
        Warning message if empty, None otherwise.
    """
    if not content or content.strip() == "":
        return "System reminder: File exists but has empty contents"
    return None


def has_memories_prefix(file_path: str) -> bool:
    """Check if file path has the memories prefix.

    Args:
        file_path: File path.

    Returns:
        True if file path has the memories prefix, False otherwise.
    """
    return file_path.startswith("/memories/")


def append_memories_prefix(file_path: str) -> str:
    """Append the memories prefix to a file path.

    Args:
        file_path: File path.

    Returns:
        File path with the memories prefix.
    """
    return f"/memories{file_path}"


def strip_memories_prefix(file_path: str) -> str:
    """Strip the memories prefix from a file path.

    Args:
        file_path: File path.

    Returns:
        File path without the memories prefix.
    """
    return file_path.replace("/memories", "")

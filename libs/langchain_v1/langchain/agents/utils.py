"""Shared utility functions for memory backend implementations.

This module contains both user-facing string formatters and structured
helpers used by backends and the composite router. Structured helpers
enable composition without fragile string parsing.
"""

import functools
import os
import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any, Final, Literal, overload

import wcmatch.glob as wcglob

from langchain.agents.protocol import FileData, GrepResult, ReadResult
from langchain.agents.protocol import FileInfo as _FileInfo
from langchain.agents.protocol import GrepMatch as _GrepMatch

EMPTY_CONTENT_WARNING = "System reminder: File exists but has empty contents"
MAX_VIDEO_INPUT_BYTES: Final = 1024 * 1024 * 1024
"""Maximum raw video payload size accepted by `read_file` frame extraction."""

FileType = Literal["text", "image", "audio", "video", "file"]
"""Classification of a file by extension."""

_EXTENSION_TO_FILE_TYPE: dict[str, FileType] = {
    # Images (https://ai.google.dev/gemini-api/docs/image-understanding)
    ".png": "image",
    ".jpeg": "image",
    ".jpg": "image",
    ".webp": "image",
    ".gif": "image",
    ".heic": "image",
    ".heif": "image",
    # Video (https://ai.google.dev/gemini-api/docs/video-understanding)
    ".mp4": "video",
    ".mpeg": "video",
    ".mov": "video",
    ".avi": "video",
    ".flv": "video",
    ".mpg": "video",
    ".webm": "video",
    ".wmv": "video",
    ".3gpp": "video",
    # Audio (https://ai.google.dev/gemini-api/docs/audio)
    ".wav": "audio",
    ".mp3": "audio",
    ".aiff": "audio",
    ".aac": "audio",
    ".ogg": "audio",
    ".flac": "audio",
    # Files
    ".pdf": "file",
    ".ppt": "file",
    ".pptx": "file",
}
"""Extension-to-type mapping for non-text files.

Optional features may layer on additional classifications at the use site. For
example, `read_file` treats `.mkv` as video only when the optional video
dependencies are installed.

Derived from Google's multimodal API supported formats:

- Images: https://ai.google.dev/gemini-api/docs/image-understanding
- Video: https://ai.google.dev/gemini-api/docs/video-understanding
- Audio: https://ai.google.dev/gemini-api/docs/audio
"""

MAX_LINE_LENGTH = 5000
TOOL_RESULT_TOKEN_LIMIT = 20000  # Same threshold as eviction
TRUNCATION_GUIDANCE = "... [results truncated, try being more specific with your parameters]"

# Re-export protocol types for backwards compatibility
FileInfo = _FileInfo
GrepMatch = _GrepMatch


@functools.lru_cache(maxsize=256)
def compile_grep_include_glob(pattern: str) -> Callable[[str], bool]:
    """Compile a grep include-glob into a matcher with ripgrep-like semantics.

    Provides one shared include-glob behavior for every backend so the same
    `grep(..., glob=...)` call closely mirrors ripgrep for common include
    patterns, whether or not ripgrep is installed:

    - Patterns without a `/` match the basename at any depth.

        Example: `*.py` matches `src/app/main.py`.
    - Patterns containing a `/` match the path relative to the grep search
        root, with `**` support.

        Example: `src/**/*.py` matches `src/app/main.py`.
    - A leading `/` anchors the pattern to the search root; it narrows the match
        rather than widening it.

        Example: `/*.py` matches `top.py` but not `src/app/main.py`.

    Exclusion/negation patterns (a leading `!`) are not supported: the `!` is
    treated literally rather than inverting the match, so results for such
    patterns can diverge from `rg --glob '!...'`.

    Args:
        pattern: Glob include pattern.

    Returns:
        Predicate accepting a search-root-relative POSIX path; returns True when
        the path is included by `pattern`.
    """
    flags = wcglob.BRACE | wcglob.GLOBSTAR
    # A leading `/` anchors to the search root: strip it so it matches against
    # the (slash-less) relative path, but decide anchoring from the original
    # pattern so `/*.py` stays root-anchored instead of collapsing to a
    # basename-at-any-depth match.
    anchored = "/" in pattern
    compiled = wcglob.compile(pattern.lstrip("/"), flags=flags)

    if anchored:

        def matcher(rel_path: str) -> bool:
            return bool(compiled.match(rel_path))
    else:

        def matcher(rel_path: str) -> bool:
            return bool(compiled.match(PurePosixPath(rel_path).name))

    return matcher


def compile_recursive_glob(pattern: str) -> Callable[[str], bool]:
    """Compile a `glob` pattern into a per-entry matcher for a recursive walk.

    `Path.rglob(pattern)` is equivalent to `Path.glob("**/" + pattern)`, so the
    pattern matches at any depth (e.g. `*.py` matches `src/app/main.py`). Prefix
    the pattern with `**/` and compile it with globstar support so a matcher can
    be applied to each visited entry while walking the tree, letting the caller
    enforce a deadline on every entry instead of only on matched paths.

    Depth (`GLOBSTAR`) and dotfile matching (`DOTMATCH`) mirror `Path.rglob`:
    `DOTMATCH` is required because `wcmatch` excludes dotfiles by default whereas
    stdlib `rglob` includes them. Brace expansion (`BRACE`) is an intentional
    *divergence* from `rglob` — `{a,b}.py` expands here but `Path.rglob` treats
    the braces literally — chosen so `glob` matches the include-glob semantics of
    `compile_grep_include_glob`.

    Args:
        pattern: Glob pattern (a leading `/` is stripped).

    Returns:
        Predicate accepting a search-root-relative POSIX path; returns True when
        the path matches `pattern` under recursive-glob semantics.
    """
    flags = wcglob.BRACE | wcglob.GLOBSTAR | wcglob.DOTMATCH
    compiled = wcglob.compile("**/" + pattern.lstrip("/"), flags=flags)

    def matcher(rel_path: str) -> bool:
        return bool(compiled.match(rel_path))

    return matcher


def _normalize_content(file_data: FileData) -> str:
    """Normalize current and legacy file data content to a plain string.

    Args:
        file_data: `FileData` dict with `content` key.

    Returns:
        Content as a single string.

    Raises:
        TypeError: If content is neither a string nor a legacy list of strings.
    """
    content: object = file_data["content"]
    if isinstance(content, list) and all(isinstance(line, str) for line in content):
        return "\n".join(content)
    if not isinstance(content, str):
        msg = f"File content must be a string or a legacy list of strings, got {type(content).__name__}."
        raise TypeError(msg)
    return content


def sanitize_tool_call_id(tool_call_id: str) -> str:
    r"""Sanitize tool_call_id to prevent path traversal and separator issues.

    Replaces dangerous characters (., /, \) with underscores.
    """
    return tool_call_id.replace(".", "_").replace("/", "_").replace("\\", "_")


def format_content_with_line_numbers(
    content: str | list[str],
    start_line: int = 1,
) -> str:
    """Format file content with line numbers.

    Chunks lines longer than `MAX_LINE_LENGTH` with continuation markers
    (e.g., `5.1`, `5.2`). Line markers are separated from source content
    with two spaces so source tabs cannot be confused with a gutter separator.

    Args:
        content: File content as string or list of lines
        start_line: Starting line number

    Returns:
        Formatted content with line numbers and continuation markers
    """
    if isinstance(content, str):
        lines = content.split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
    else:
        lines = content

    rows: list[tuple[str, str]] = []
    marker_width = 0
    for i, line in enumerate(lines):
        line_num = i + start_line
        # One slice per MAX_LINE_LENGTH chunk; short lines yield a single chunk.
        # `or [line]` keeps a row for a blank line, whose empty range would
        # otherwise drop it, so it still gets a gutter.
        chunks = [line[s : s + MAX_LINE_LENGTH] for s in range(0, len(line), MAX_LINE_LENGTH)] or [line]

        for chunk_idx, chunk in enumerate(chunks):
            marker = str(line_num) if chunk_idx == 0 else f"{line_num}.{chunk_idx}"
            rows.append((marker, chunk))
            marker_width = max(marker_width, len(marker))

    # The two-space marker/source separator is a load-bearing contract shared by
    # two downstream parsers that must stay in sync with the separator emitted
    # here:
    #   - `ReadFileContinuationNoticeMiddleware._is_numbered_read_file_row`
    #     (profiles/harness/_nvidia_nemotron_3_ultra.py) counts source rows to
    #     decide whether to append the continuation notice.
    #   - `ToolCallMessage._compact_line_gutter` (the agents-code TUI, in a
    #     separate package: libs/code/.../tui/widgets/messages.py) re-justifies
    #     the gutter for display.
    # Both also tolerate the legacy `cat -n` tab. Shrinking this separator below
    # two spaces (or otherwise diverging) would silently break them; the
    # producer->consumer round-trip tests in both packages guard against that.
    return "\n".join(f"{marker:>{marker_width}}  {line}" for marker, line in rows)


def check_empty_content(content: str) -> str | None:
    """Check if content is empty and return warning message.

    Args:
        content: Content to check

    Returns:
        Warning message if empty, `None` otherwise
    """
    if not content or content.strip() == "":
        return EMPTY_CONTENT_WARNING
    return None


def _get_file_type(path: str) -> FileType:
    """Classify a file by its extension.

    Args:
        path: File path to classify.

    Returns:
        One of `"text"`, `"image"`, `"audio"`, `"video"`, or `"file"`.

            Defaults to `"text"` for unrecognized extensions.
    """
    return _EXTENSION_TO_FILE_TYPE.get(PurePosixPath(path).suffix.lower(), "text")


_VIDEO_EXTRA_EXTENSIONS: frozenset[str] = frozenset({".mkv"})
"""Video container extensions handled outside the Google-derived multimodal map.

These are intentionally absent from `_EXTENSION_TO_FILE_TYPE`, so a `read_file`
without the optional `[video]` extra returns them as a generic file block rather
than a native video block. Backends must still read them as binary — never
text-decode them — and `read_file` layers frame extraction on top only when the
`[video]` dependencies are installed.
"""


def _get_backend_read_file_type(path: str) -> FileType:
    """Classify a file for backend reads, forcing known video containers to binary.

    Backends decide binary-vs-text on `_get_file_type(...) != "text"`. Extensions
    in `_VIDEO_EXTRA_EXTENSIONS` are absent from `_EXTENSION_TO_FILE_TYPE`, so
    `_get_file_type` alone would treat them as text and corrupt the bytes (a raw
    UTF-8 decode of a video, or line-slicing a base64 blob). Classify them as
    `"video"` here so the binary read path runs on every backend.

    Args:
        path: File path to classify.

    Returns:
        `"video"` for `_VIDEO_EXTRA_EXTENSIONS`; otherwise the shared
            `_get_file_type` classification.
    """
    if PurePosixPath(path).suffix.lower() in _VIDEO_EXTRA_EXTENSIONS:
        return "video"
    return _get_file_type(path)


def file_data_to_string(file_data: FileData) -> str:
    """Convert current or legacy persisted file content to a string.

    Args:
        file_data: File data whose content is a string or legacy list of strings.

    Returns:
        Content as a single string.

    Raises:
        TypeError: If content is neither a string nor a legacy list of strings.
    """
    return _normalize_content(file_data)


def create_file_data(
    content: str,
    created_at: str | None = None,
    encoding: str = "utf-8",
) -> FileData:
    """Create a `FileData` object with timestamps.

    Args:
        content: File content as string (plain text or base64-encoded binary).
        created_at: Optional creation timestamp (ISO format).
        encoding: Content encoding — `"utf-8"` for text, `"base64"` for binary.

    Returns:
        FileD`ata dict with content, encoding, and timestamps.
    """
    now = datetime.now(UTC).isoformat()

    return {
        "content": content,
        "encoding": encoding,
        "created_at": created_at or now,
        "modified_at": now,
    }


def update_file_data(file_data: FileData, content: str) -> FileData:
    """Update `FileData` with new content, preserving creation timestamp.

    Args:
        file_data: Existing `FileData` dict
        content: New content as string

    Returns:
        Updated `FileData` dict
    """
    now = datetime.now(UTC).isoformat()

    result = FileData(
        content=content,
        encoding=file_data.get("encoding", "utf-8"),
    )
    if "created_at" in file_data:
        result["created_at"] = file_data["created_at"]
    result["modified_at"] = now
    return result


def _copy_file_data_with_content(file_data: FileData, content: str) -> FileData:
    """Clone `file_data` with replaced content, preserving timestamps when present.

    Unlike `update_file_data`, this carries `created_at`/`modified_at` through
    verbatim rather than restamping `modified_at`, since slicing a read window
    does not mutate the underlying file.

    Args:
        file_data: Source `FileData` whose encoding and timestamps are copied.
        content: Replacement content for the returned copy.

    Returns:
        A new `FileData` with `content` set and metadata carried over.
    """
    sliced_fd = FileData(
        content=content,
        encoding=file_data.get("encoding", "utf-8"),
    )
    if "created_at" in file_data:
        sliced_fd["created_at"] = file_data["created_at"]
    if "modified_at" in file_data:
        sliced_fd["modified_at"] = file_data["modified_at"]
    return sliced_fd


def slice_read_response(
    file_data: FileData,
    offset: int,
    limit: int,
) -> ReadResult:
    """Slice file data to the requested line range without formatting.

    The returned `ReadResult` carries the raw (unformatted) window in
    `file_data`; line-number formatting is applied downstream by the
    middleware layer.

    Args:
        file_data: `FileData` dict.
        offset: Line offset (0-indexed).
        limit: Maximum number of lines.

    Returns:
        `ReadResult` with the sliced raw content and pagination metadata
            (`total_lines`, `start_line`, `end_line`, `next_offset`). The
            pagination fields are left unset for empty or whitespace-only
            content. `error` is set instead when the offset exceeds the file
            length.
    """
    content = file_data_to_string(file_data)

    if not content or content.strip() == "":
        return ReadResult(file_data=_copy_file_data_with_content(file_data, content))

    # `splitlines(keepends=True)` retains each line's terminator, including
    # the absence of one on the final line. Joining with `""` therefore
    # round-trips the trailing-newline state of the file faithfully —
    # required so `edit()` can report EOF-newline mismatches accurately. It
    # also splits on CR / CRLF, so line indexing matches the LF-normalized
    # form without first rewriting the whole (potentially huge) string.
    lines = content.splitlines(keepends=True)
    start_idx = offset
    end_idx = min(start_idx + limit, len(lines))
    total_lines = len(lines)

    if start_idx >= total_lines:
        return ReadResult(error=f"Line offset {offset} exceeds file length ({total_lines} lines)")

    # Normalize line endings to LF, but only across the requested window.
    # State/Store backends may carry CRLF or CR content as written;
    # downstream tooling (edit match, grep, format) assumes LF.
    sliced = "".join(lines[start_idx:end_idx]).replace("\r\n", "\n").replace("\r", "\n")
    next_offset = end_idx if end_idx < total_lines else None
    return ReadResult(
        file_data=_copy_file_data_with_content(file_data, sliced),
        total_lines=total_lines,
        start_line=start_idx + 1,
        end_line=end_idx,
        next_offset=next_offset,
    )


def perform_string_replacement(
    content: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,  # noqa: FBT001, FBT002
) -> tuple[str, int] | str:
    """Perform string replacement with occurrence validation.

    Args:
        content: Original content
        old_string: String to replace
        new_string: Replacement string
        replace_all: Whether to replace all occurrences

    Returns:
        Tuple of `(new_content, occurrences)` on success, or error message string
    """
    occurrences = content.count(old_string)

    if occurrences == 0:
        # Detect a common EOF mismatch: `old_string` carries a trailing
        # newline that the file lacks at the same position. Models infer a
        # terminator on what looks like a "well-formed" line; exact-match
        # consumers must surface a precise hint rather than silently relax
        # the contract — silent recovery on a stripped key risks corrupting
        # interior text that happens to share a prefix.
        if old_string.endswith("\n") and len(old_string) > 1 and content.endswith(old_string.removesuffix("\n")):
            stripped = old_string.removesuffix("\n")
            stripped_count = content.count(stripped)
            if stripped_count == 1:
                return (
                    "Error: old_string ends with a newline, but the file does "
                    "not end with a newline. Retry with the trailing newline "
                    "removed from old_string (and from new_string if it also "
                    "ends with a newline)."
                )
            # Stripped key is ambiguous: the model needs both fixes at once
            # (drop the newline AND add surrounding context).
            return (
                f"Error: old_string ends with a newline, but the file does "
                f"not end with a newline. With the trailing newline removed, "
                f"old_string would appear {stripped_count} times in the file. "
                f"Retry with the trailing newline removed and add surrounding "
                f"context so the match is unique."
            )
        return f"Error: String not found in file: '{old_string}'"

    if occurrences > 1 and not replace_all:
        return (
            f"Error: String '{old_string}' appears {occurrences} times in file. "
            f"Use replace_all=True to replace all instances, or provide a more specific string with surrounding context."
        )

    new_content = content.replace(old_string, new_string)
    return new_content, occurrences


@overload
def truncate_if_too_long(result: list[str]) -> list[str]: ...


@overload
def truncate_if_too_long(result: str) -> str: ...


def truncate_if_too_long(result: list[str] | str) -> list[str] | str:
    """Truncate list or string result if it exceeds token limit (rough estimate: 4 chars/token)."""
    if isinstance(result, list):
        total_chars = sum(len(item) for item in result)
        if total_chars > TOOL_RESULT_TOKEN_LIMIT * 4:
            return result[: len(result) * TOOL_RESULT_TOKEN_LIMIT * 4 // total_chars] + [TRUNCATION_GUIDANCE]  # noqa: RUF005  # Concatenation preferred for clarity
        return result
    # string
    if len(result) > TOOL_RESULT_TOKEN_LIMIT * 4:
        return result[: TOOL_RESULT_TOKEN_LIMIT * 4] + "\n" + TRUNCATION_GUIDANCE
    return result


# Characters that mark a glob path component as a wildcard segment for the
# purposes of `_glob_anchor`. Keep in sync with the wcmatch flags used by the
# filesystem middleware (`BRACE | GLOBSTAR`).
_GLOB_WILDCARD_CHARS = frozenset("*?[{")


def _glob_anchor(pattern: str) -> str:
    """Return the longest leading directory of `pattern` with no wildcards.

    For `/secrets/**` returns `/secrets`; for `/a/*/b` returns `/a`; for a
    pattern with a wildcard at or near the root (`/**/secrets`, `/*/foo`)
    falls back to `/`. The root fallback causes overlap checks to match
    *any* subtree — conservative over-gating, since we cannot statically
    pin down where the rule could resolve. Callers wanting precise gating
    should anchor the rule's leading components.
    """
    parts = PurePosixPath(to_posix_path(pattern)).parts
    safe: list[str] = []
    for part in parts:
        if any(c in _GLOB_WILDCARD_CHARS for c in part):
            break
        safe.append(part)
    if not safe:
        return "/"
    return str(PurePosixPath(*safe))


def _paths_overlap(call_path: str, rule_anchor: str) -> bool:
    """Return True if the subtree at `call_path` intersects the subtree at `rule_anchor`.

    Two subtrees overlap when one is a (component-wise) prefix of the other,
    or they're equal. Comparison runs on `PurePosixPath` components, so
    `/secret` does not overlap `/secrets`. The root `/` overlaps everything.
    """
    a = PurePosixPath(call_path)
    b = PurePosixPath(rule_anchor)
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def to_posix_path(path: str) -> str:
    r"""Normalize backslash separators to forward slashes for `PurePosixPath` use.

    Backends running on Windows return OS-native paths using backslashes.
    `PurePosixPath` treats backslashes as literal filename characters,
    so `PurePosixPath(r"C:\a\b").name` yields the full string instead
    of `"b"`. Normalize before constructing a `PurePosixPath`.

    This is best-effort: a POSIX directory literally named with a backslash
    will also be rewritten. That trade-off is accepted because such filenames
    are vanishingly rare in practice and the alternative (gating on `os.sep`)
    fails when a Windows-style path is handed to a non-Windows process.

    Args:
        path: Path string that may use backslash separators.

    Returns:
        The same path with every `\\` replaced by `/`.

            Inputs that already use forward slashes are returned unchanged.
    """
    return path.replace("\\", "/")


def validate_path(path: str, *, allowed_prefixes: Sequence[str] | None = None) -> str:
    r"""Validate and normalize file path for security.

    Ensures paths are safe to use by preventing directory traversal attacks
    and enforcing consistent formatting. All paths are normalized to use
    forward slashes and start with a leading slash.

    This function is designed for virtual filesystem paths and rejects
    Windows absolute paths (e.g., `C:/...`, `F:/...`) to maintain consistency
    and prevent path format ambiguity.

    Args:
        path: The path to validate and normalize.
        allowed_prefixes: Optional list of allowed path prefixes.

            If provided, the normalized path must start with one of
            these prefixes.

    Returns:
        Normalized canonical path starting with `/` and using forward slashes.

    Raises:
        ValueError: If path contains traversal sequences (`..` or `~`), is a
            Windows absolute path (e.g., `C:/...`), or does not start with an
            allowed prefix when `allowed_prefixes` is specified.

    Example:
        ```python
        validate_path("foo/bar")  # Returns: "/foo/bar"
        validate_path("/./foo//bar")  # Returns: "/foo/bar"
        validate_path("../etc/passwd")  # Raises ValueError
        validate_path(r"C:\\Users\\file.txt")  # Raises ValueError
        validate_path("/data/file.txt", allowed_prefixes=["/data/"])  # OK
        validate_path("/etc/file.txt", allowed_prefixes=["/data/"])  # Raises ValueError
        ```
    """
    # Check for traversal as a path component (not substring) to avoid
    # false-positive rejection of legitimate filenames like "foo..bar.txt"
    parts = PurePosixPath(to_posix_path(path)).parts
    if ".." in parts or path.startswith("~"):
        msg = f"Path traversal not allowed: {path}"
        raise ValueError(msg)

    # Reject Windows absolute paths (e.g., C:\..., D:/...)
    if re.match(r"^[a-zA-Z]:", path):
        msg = f"Windows absolute paths are not supported: {path}. Please use virtual paths starting with / (e.g., /workspace/file.txt)"
        raise ValueError(msg)

    normalized = os.path.normpath(path)
    normalized = normalized.replace("\\", "/")

    if not normalized.startswith("/"):
        normalized = f"/{normalized}"

    # Defense-in-depth: verify normpath didn't produce traversal
    if ".." in normalized.split("/"):
        msg = f"Path traversal detected after normalization: {path} -> {normalized}"
        raise ValueError(msg)

    if allowed_prefixes is not None and not any(normalized.startswith(prefix) for prefix in allowed_prefixes):
        msg = f"Path must start with one of {allowed_prefixes}: {path}"
        raise ValueError(msg)

    return normalized


def _normalize_path(path: str | None) -> str:
    """Normalize a path to canonical form.

    Converts path to absolute form starting with /, removes trailing slashes
    (except for root), and validates that the path is not empty.

    Args:
        path: Path to normalize (None defaults to "/")

    Returns:
        Normalized path starting with / (without trailing slash unless it's root)

    Raises:
        ValueError: If path is invalid (empty string after strip)

    Example:
        _normalize_path(None) -> "/"
        _normalize_path("/dir/") -> "/dir"
        _normalize_path("dir") -> "/dir"
        _normalize_path("/") -> "/"
    """
    path = path or "/"
    if not path or path.strip() == "":
        msg = "Path cannot be empty"
        raise ValueError(msg)

    normalized = path if path.startswith("/") else "/" + path

    # Only root should have trailing slash
    if normalized != "/" and normalized.endswith("/"):
        normalized = normalized.rstrip("/")

    return normalized


def _filter_files_by_path(files: dict[str, Any], normalized_path: str) -> dict[str, Any]:
    """Filter files dict by normalized path, handling exact file matches and directory prefixes.

    Expects a normalized path from `_normalize_path` (no trailing slash except root).

    Args:
        files: Dictionary mapping file paths to file data
        normalized_path: Normalized path from `_normalize_path` (e.g., "/", "/dir", "/dir/file")

    Returns:
        Filtered dictionary of files matching the path

    Example:
        files = {"/dir/file": {...}, "/dir/other": {...}}
        _filter_files_by_path(files, "/dir/file")  # Returns {"/dir/file": {...}}
        _filter_files_by_path(files, "/dir")       # Returns both files
    """
    # Check if path matches an exact file
    if normalized_path in files:
        return {normalized_path: files[normalized_path]}

    # Otherwise treat as directory prefix
    if normalized_path == "/":
        # Root directory - match all files starting with /
        return {fp: fd for fp, fd in files.items() if fp.startswith("/")}
    # Non-root directory - add trailing slash for prefix matching
    dir_prefix = normalized_path + "/"
    return {fp: fd for fp, fd in files.items() if fp.startswith(dir_prefix)}


def _relative_to_root(file_path: str, normalized_path: str) -> str:
    """Return `file_path` relative to a normalized grep/glob search root.

    Args:
        file_path: Absolute file path (e.g. "/src/app/main.py").
        normalized_path: Normalized search root from `_normalize_path`.

    Returns:
        POSIX path relative to the search root (e.g. "src/app/main.py").

            When `file_path` equals the search root (an exact-file search),
            returns just the basename.
    """
    if normalized_path == "/":
        return file_path[1:]
    if file_path == normalized_path:
        return file_path.rsplit("/", maxsplit=1)[-1]
    return file_path[len(normalized_path) + 1 :]


def _glob_search_files(
    files: dict[str, Any],
    pattern: str,
    path: str | None = None,
) -> str:
    r"""Search files dict for paths matching glob pattern.

    Args:
        files: Dictionary of file paths to FileData.
        pattern: Glob pattern (e.g., `"*.py"`, `"**/*.ts"`).
        path: Base path to search from. `None` defaults to root.

    Returns:
        Newline-separated file paths, sorted by modification time (most recent first).

            `"No files found"` if no matches.

    Example:
        ```python
        files = {"/src/main.py": FileData(...), "/test.py": FileData(...)}
        _glob_search_files(files, "*.py", "/")
        # Returns: "/test.py\n/src/main.py" (sorted by modified_at)
        ```
    """
    try:
        normalized_path = _normalize_path(path)
    except ValueError:
        return "No files found"

    filtered = _filter_files_by_path(files, normalized_path)

    # Respect standard glob semantics:
    # - Patterns without path separators (e.g., "*.py") match only in the current
    #   directory (non-recursive) relative to `path`.
    # - Use "**" explicitly for recursive matching.
    # Strip leading "/" from pattern since matching is done against relative paths.
    effective_pattern = pattern.lstrip("/")

    matches = []
    for file_path, file_data in filtered.items():
        # Compute relative path for glob matching
        # If normalized_path is "/dir", we want "/dir/file.txt" -> "file.txt"
        # If normalized_path is "/dir/file.txt" (exact file), we want "file.txt"
        if normalized_path == "/":
            relative = file_path[1:]  # Remove leading slash
        elif file_path == normalized_path:
            # Exact file match - use just the filename
            relative = file_path.split("/")[-1]
        else:
            # Directory prefix - strip the directory path
            relative = file_path[len(normalized_path) + 1 :]  # +1 for the slash

        if wcglob.globmatch(relative, effective_pattern, flags=wcglob.BRACE | wcglob.GLOBSTAR):
            matches.append((file_path, file_data["modified_at"]))

    matches.sort(key=lambda x: x[1], reverse=True)

    if not matches:
        return "No files found"

    return "\n".join(fp for fp, _ in matches)


def _format_grep_results(
    results: dict[str, list[tuple[int, str]]],
    output_mode: Literal["files_with_matches", "content", "count"],
) -> str:
    """Format grep search results based on output mode.

    Args:
        results: Dictionary mapping file paths to list of `(line_num, line_content)` tuples
        output_mode: Output format

    Returns:
        Formatted string output
    """
    if output_mode == "files_with_matches":
        return "\n".join(sorted(results.keys()))
    if output_mode == "count":
        lines = []
        for file_path in sorted(results.keys()):
            count = len(results[file_path])
            lines.append(f"{file_path}: {count}")
        return "\n".join(lines)
    lines = []
    for file_path in sorted(results.keys()):
        lines.append(f"{file_path}:")
        for line_num, line in results[file_path]:
            lines.append(f"  {line_num}: {line}")
    return "\n".join(lines)


# -------- Structured helpers for composition --------


def grep_matches_from_files(
    files: dict[str, Any],
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
    *,
    max_count: int | None = None,
) -> GrepResult:
    """Return structured grep matches from an in-memory files mapping.

    Performs literal text search (not regex).

    Returns a `GrepResult` with matches on success. When `max_count` is set, at
    most that many matches are returned; if more exist the scan stops and the
    result is flagged `truncated=True`. Exactly `max_count` matches with none
    dropped is reported complete (`truncated=False`).

    We deliberately do not raise here to keep backends non-throwing in tool
    contexts and preserve user-facing error messages.
    """
    try:
        normalized_path = _normalize_path(path)
    except ValueError:
        return GrepResult(matches=[])

    filtered = _filter_files_by_path(files, normalized_path)

    if glob:
        matcher = compile_grep_include_glob(glob)
        filtered = {fp: fd for fp, fd in filtered.items() if matcher(_relative_to_root(fp, normalized_path))}

    matches: list[GrepMatch] = []
    for file_path, file_data in filtered.items():
        content_str = _normalize_content(file_data)
        for line_num, line in enumerate(content_str.split("\n"), 1):
            if pattern in line:  # Simple substring search for literal matching
                if max_count is not None and len(matches) >= max_count:
                    # A further match beyond `max_count` proves more exist; stop
                    # and flag truncation. Checked before appending so exactly
                    # `max_count` matches is reported complete, not truncated.
                    return GrepResult(matches=matches, truncated=True)
                matches.append({"path": file_path, "line": int(line_num), "text": line})
    return GrepResult(matches=matches)


def build_grep_results_dict(matches: list[GrepMatch]) -> dict[str, list[tuple[int, str]]]:
    """Group structured matches into the legacy dict form used by formatters."""
    grouped: dict[str, list[tuple[int, str]]] = {}
    for m in matches:
        grouped.setdefault(m["path"], []).append((m["line"], m["text"]))
    return grouped


def format_grep_matches(
    matches: list[GrepMatch],
    output_mode: Literal["files_with_matches", "content", "count"],
) -> str:
    """Format structured grep matches using existing formatting logic."""
    if not matches:
        return "No matches found"

    # Presence of the context keys signals "context mode" for the whole result;
    # the producer sets both keys on every match or none. `_format_grep_with_context`
    # still tolerates a hand-built mix of matches with and without context, because
    # `format_grep_matches` is public and may be handed such input.
    if output_mode != "content" or not any("context_before" in match or "context_after" in match for match in matches):
        return _format_grep_results(build_grep_results_dict(matches), output_mode)
    return _format_grep_with_context(matches)


def _format_grep_with_context(matches: list[GrepMatch]) -> str:
    """Render `content`-mode grep output including surrounding context lines.

    Matched lines are marked with `:` and context lines with `-`. Non-adjacent
    line groups within a file are separated by a `--` line, mirroring `grep -C`.
    """
    matches_by_path: dict[str, list[GrepMatch]] = {}
    for match in matches:
        matches_by_path.setdefault(match["path"], []).append(match)

    lines: list[str] = []
    for file_path in sorted(matches_by_path):
        file_matches = matches_by_path[file_path]
        matching_lines = {match["line"] for match in file_matches}
        displayed_lines: dict[int, str] = {}
        for match in file_matches:
            for context_line in match.get("context_before", []):
                displayed_lines[context_line["line"]] = context_line["text"]
            displayed_lines[match["line"]] = match["text"]
            for context_line in match.get("context_after", []):
                displayed_lines[context_line["line"]] = context_line["text"]

        lines.append(f"{file_path}:")
        for group_index, group in enumerate(_group_adjacent_lines(displayed_lines)):
            if group_index:
                lines.append("  --")
            for line_num, text in group:
                separator = ":" if line_num in matching_lines else "-"
                lines.append(f"  {line_num}{separator} {text}")
    return "\n".join(lines)


def _group_adjacent_lines(displayed_lines: dict[int, str]) -> list[list[tuple[int, str]]]:
    """Split `{line_number: text}` into runs of consecutive line numbers."""
    groups: list[list[tuple[int, str]]] = []
    for item in sorted(displayed_lines.items()):
        if not groups or item[0] > groups[-1][-1][0] + 1:
            groups.append([item])
        else:
            groups[-1].append(item)
    return groups


_REGEX_SIGNAL_RE = re.compile(
    r"\|"  # alternation
    r"|\.\*"  # `.*` wildcard
    r"|\.\+"  # `.+` wildcard
    r"|\\[.wWdDsSbB(){}\[\]|+*?^$]"  # escaped regex metacharacters / classes
)
"""Strong signals that a pattern was written as a regex rather than literal text.

Deliberately conservative: bare `.`, `(`, `)`, `[`, `]`, `?`, `^`, `$` are
omitted because they appear routinely in literal code searches (e.g.
`self.tools`, `def __init__(self):`, `arr[0]`), which would cause false hints.
"""


def _looks_like_regex(pattern: str) -> bool:
    """Heuristically detect regex syntax in a pattern meant for literal grep."""
    return bool(_REGEX_SIGNAL_RE.search(pattern))


def regex_literal_hint(pattern: str) -> str | None:
    """Return a hint when a pattern looks like an (unsupported) regex.

    `grep` matches literal text, so regex metacharacters are searched verbatim
    and silently miss. Callers gate this on a no-match result; the function
    itself only inspects the pattern.

    Args:
        pattern: The literal grep pattern to inspect for regex signals.

    Returns:
        A one-line hint steering the caller toward literal search, or `None`
            when the pattern has no regex signals.
    """
    if not _looks_like_regex(pattern):
        return None
    return (
        "Note: grep matches literal text, not regex, so characters like "
        "`|`, `.*`, and `\\.` are searched verbatim. Search for the literal "
        "text you need instead; for `|` alternation, run a separate search "
        "per alternative."
    )

"""Protocol definition for pluggable memory backends.

This module defines the `BackendProtocol` that all backend implementations
must follow. Backends can store files in different locations (state, filesystem,
database, etc.) and provide a uniform interface for file operations.
"""

import abc
import asyncio
import inspect
import logging
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Final, Literal, NotRequired

from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

DEFAULT_GREP_TIMEOUT: Final = 15
"""Default timeout in seconds for one sync grep phase."""

ASYNC_GREP_TIMEOUT: Final = (2 * DEFAULT_GREP_TIMEOUT) + 5
"""Timeout in seconds for the async grep wrapper.

This gives `FilesystemBackend` enough headroom to finish the worst-case sync
path: ripgrep timeout, then Python fallback timeout.
"""

FileOperationError = Literal[
    "file_not_found",
    "permission_denied",
    "is_directory",
    "invalid_path",
]
"""Standardized error codes for file upload/download operations.

These represent common, recoverable errors that an LLM can understand and
potentially fix:

- `file_not_found`: The requested file doesn't exist (download)
- `permission_denied`: Access denied for the operation
- `is_directory`: Attempted to download a directory as a file
- `invalid_path`: Path syntax is malformed or contains invalid characters
"""

# Named constants for each `FileOperationError` literal. Use these instead of
# bare string literals at producer/consumer sites so a rename in one place
# surfaces as a type error (rather than silently reverting to a fallback branch).
FILE_NOT_FOUND: Final = "file_not_found"
PERMISSION_DENIED: Final = "permission_denied"
IS_DIRECTORY: Final = "is_directory"
INVALID_PATH: Final = "invalid_path"


@dataclass
class FileDownloadResponse:
    """Result of a single file download operation.

    The response is designed to allow partial success in batch operations.

    The errors are standardized using `FileOperationError` literals for certain
    recoverable conditions for use cases that involve LLMs performing
    file operations.

    Examples:
        >>> # Success
        >>> FileDownloadResponse(path="/app/config.json", content=b"{...}", error=None)
        >>> # Failure
        >>> FileDownloadResponse(path="/wrong/path.txt", content=None, error="file_not_found")
    """

    path: str
    """The file path that was requested. Included for easy correlation when
    processing batch results, especially useful for error messages."""

    content: bytes | None = None
    """File contents as bytes on success, `None` on failure."""

    error: FileOperationError | str | None = None
    """A `FileOperationError` literal for known conditions, or a
    backend-specific error string when the failure cannot be normalized.

    `None` on success.
    """


@dataclass
class FileUploadResponse:
    """Result of a single file upload operation.

    The response is designed to allow partial success in batch operations.

    The errors are standardized using `FileOperationError` literals for certain
    recoverable conditions for use cases that involve LLMs performing
    file operations.

    Examples:
        >>> # Success
        >>> FileUploadResponse(path="/app/data.txt", error=None)
        >>> # Failure
        >>> FileUploadResponse(path="/readonly/file.txt", error="permission_denied")
    """

    path: str
    """The file path that was requested.

    Included for easy correlation when processing batch results and for clear
    error messages.
    """

    error: FileOperationError | str | None = None
    """A `FileOperationError` literal for known conditions, or a
    backend-specific error string when the failure cannot be normalized.

    `None` on success.
    """


class FileInfo(TypedDict):
    """Structured file listing info.

    Minimal contract used across backends. Only `path` is required.
    Other fields are best-effort and may be absent depending on backend.
    """

    path: str
    """Absolute or relative file path."""

    is_dir: NotRequired[bool]
    """Whether the entry is a directory."""

    size: NotRequired[int]
    """File size in bytes (approximate)."""

    modified_at: NotRequired[str]
    """ISO 8601 timestamp of last modification, if known."""


class ContextLine(TypedDict):
    """A non-matching line surrounding a grep match, used for `context_lines`."""

    line: int
    """1-indexed line number of the context line."""

    text: str
    """Content of the context line."""


class GrepMatch(TypedDict):
    """A single match from a grep search."""

    path: str
    """Path to the file containing the match."""

    line: int
    """1-indexed line number of the match."""

    text: str
    """Content of the matching line."""

    context_before: NotRequired[list[ContextLine]]
    """Context lines before the match.

    Present (alongside `context_after`) only when a backend was asked for
    `context_lines > 0` (via a backend-specific argument, e.g.
    `FilesystemBackend.grep`); both keys are set together on every match or on
    none. An empty list means no context lines were available on that side: the
    match sits at the file boundary, the adjacent line was itself a match
    (matches are never repeated as context), or the file could not be re-read
    (in which case the failure is reported in `GrepResult.error`).
    """

    context_after: NotRequired[list[ContextLine]]
    """Context lines after the match. See `context_before` for presence rules."""


class FileData(TypedDict):
    """Data structure for storing file contents with metadata."""

    content: str
    """File content as a plain string (utf-8 text or base64-encoded binary)."""

    encoding: str
    """Content encoding: `"utf-8"` for text, `"base64"` for binary."""

    created_at: NotRequired[str]
    """ISO 8601 timestamp of file creation."""

    modified_at: NotRequired[str]
    """ISO 8601 timestamp of last modification."""


@dataclass
class ReadResult:
    """Result from backend read operations."""

    error: str | None = None
    """Error message on failure, `None` on success."""

    file_data: FileData | None = None
    """File data on success, `None` on failure."""

    total_lines: int | None = None
    """Total number of source lines when the backend can determine it."""

    start_line: int | None = None
    """1-indexed first source line returned in `file_data`."""

    end_line: int | None = None
    """1-indexed last source line returned in `file_data`."""

    next_offset: int | None = None
    """0-indexed offset for the next unread source line."""

    def __post_init__(self) -> None:
        """Reject malformed pagination-field combinations at construction.

        The window fields are not independent: `start_line`/`end_line` are a
        pair, and neither `next_offset` nor `total_lines` describes anything
        without the window it refers to. Beyond co-presence, the values must
        agree numerically: a window runs forward (`1 <= start_line <=
        end_line`), the file is at least as long as the window
        (`total_lines >= end_line`), and the resume point is the 0-indexed line
        immediately after the last one shown (`next_offset == end_line`, since
        `end_line` is 1-indexed). Fail loudly here to keep a backend from
        emitting a `next_offset` that would silently skip unshown source lines
        once it reaches the middleware.
        """
        if (self.start_line is None) != (self.end_line is None):
            msg = "ReadResult.start_line and end_line must be set together or both left unset"
            raise ValueError(msg)
        if self.next_offset is not None and self.start_line is None:
            msg = "ReadResult.next_offset requires start_line and end_line to be set"
            raise ValueError(msg)
        if self.total_lines is not None and self.start_line is None:
            msg = "ReadResult.total_lines requires start_line and end_line to be set"
            raise ValueError(msg)

        # Numeric consistency of a present window. `start_line`/`end_line` are
        # bound together above, so testing `start_line` covers both.
        if self.start_line is not None and self.end_line is not None:
            if self.start_line < 1 or self.end_line < self.start_line:
                msg = f"ReadResult window must satisfy 1 <= start_line <= end_line, got start_line={self.start_line}, end_line={self.end_line}"
                raise ValueError(msg)
            if self.total_lines is not None and self.total_lines < self.end_line:
                msg = f"ReadResult.total_lines ({self.total_lines}) cannot be less than end_line ({self.end_line})"
                raise ValueError(msg)
            if self.next_offset is not None and self.next_offset != self.end_line:
                msg = f"ReadResult.next_offset ({self.next_offset}) must equal end_line ({self.end_line}), the 0-indexed line after the last shown"
                raise ValueError(msg)


@dataclass
class WriteResult:
    """Result from backend `write` operations.

    Attributes:
        error: Error message on failure, `None` on success.
        path: Absolute path of written file, `None` on failure.

    Examples:
        >>> WriteResult(path="/f.txt")
        >>> WriteResult(error="File exists")
    """

    error: str | None = None
    path: str | None = None


@dataclass
class EditResult:
    """Result from backend `edit` operations.

    Attributes:
        error: Error message on failure, `None` on success.
        path: Absolute path of edited file, `None` on failure.
        occurrences: Number of replacements made, `None` on failure.

    Examples:
        >>> EditResult(path="/f.txt", occurrences=1)
        >>> EditResult(error="File not found")
    """

    error: str | None = None
    path: str | None = None
    occurrences: int | None = None


@dataclass
class DeleteResult:
    """Result from backend delete operations.

    Attributes:
        error: Error message on failure, None on success.
        path: Absolute path of the deleted file, None on failure.

    Examples:
        >>> DeleteResult(path="/f.txt")
        >>> DeleteResult(error="File not found")
    """

    error: str | None = None
    path: str | None = None


@dataclass
class LsResult:
    """Result from backend `ls` operations.

    Attributes:
        error: Error message on failure, `None` on success.
        entries: List of file info dicts on success, `None` on failure.
    """

    error: str | None = None
    entries: list["FileInfo"] | None = None


@dataclass
class GrepResult:
    """Result from backend `grep` operations.

    Attributes:
        error: Error message on failure, `None` on success.
        matches: List of grep match dicts. Populated on success and, when the
            search was cut short, with whatever was found before stopping.
            `None` only on a hard failure.
        truncated: True when the search stopped early (e.g. hit its time limit)
            and `matches` is therefore incomplete but still valid.
    """

    error: str | None = None
    matches: list["GrepMatch"] | None = None
    truncated: bool = False


def _apply_grep_max_count(result: GrepResult, max_count: int | None) -> GrepResult:
    """Enforce a match cap after a backend search has completed."""
    if max_count is None or result.matches is None or len(result.matches) <= max_count:
        return result
    return GrepResult(error=result.error, matches=result.matches[:max_count], truncated=True)


@dataclass
class GlobResult:
    """Result from backend `glob` operations.

    Attributes:
        error: Error message on failure, `None` on success.
        matches: List of matching file info dicts. Populated on success and,
            when the walk was cut short, with whatever was found before
            stopping. `None` only on a hard failure.
        truncated: True when the walk stopped early (e.g. hit its time limit)
            and `matches` is therefore incomplete but still valid.
    """

    error: str | None = None
    matches: list["FileInfo"] | None = None
    truncated: bool = False


# @abstractmethod to avoid breaking subclasses that only implement a subset
class BackendProtocol(abc.ABC):  # noqa: B024
    r"""Protocol for pluggable memory backends (single, unified).

    Backends can store files in different locations (state, filesystem,
    database, etc.) and provide a uniform interface for file operations.

    File operations (`grep`, `glob`, `ls`, `read`, etc.) live on this base
    protocol rather than only on `SandboxBackendProtocol` because not every
    backend has a shell. `StateBackend` and `StoreBackend` store files in
    in-memory state or a remote store with no process to exec into, so they
    implement `grep`/`glob` in pure Python and have no `execute` at all.
    Even on shell-capable backends, the tools are not just convenience
    wrappers around `execute`: they enforce literal-only matching (not
    regex), return structured `GrepResult`/`GlobResult` objects, support
    `max_count` truncation, and pass through filesystem permission rules —
    none of which raw `execute` + shell `grep`/`find` provides. Agent-facing
    prompt guidance should therefore recommend these tools only when they
    are actually registered, and never assume a shell is available as a
    fallback.

    All file data is represented as dicts with the following structure:

    ```python
    {
        "content": str,  # Text content (utf-8) or base64-encoded binary
        "encoding": str,  # "utf-8" for text, "base64" for binary data
        "created_at": str,  # ISO format timestamp
        "modified_at": str,  # ISO format timestamp
    }
    ```
    """

    def ls(self, path: str) -> "LsResult":
        """List all files in a directory with metadata.

        Args:
            path: Absolute path to the directory to list. Must start with `'/'`.

        Returns:
            `LsResult` with directory entries or error.

        Raises:
            NotImplementedError: If the backend does not implement `ls`.
        """
        raise NotImplementedError

    async def als(self, path: str) -> "LsResult":
        """Async version of `ls`."""
        return await asyncio.to_thread(self.ls, path)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """Read file content for the requested line range.

        Args:
            file_path: Absolute path to the file to read. Must start with `'/'`.
            offset: Line number to start reading from (0-indexed).
            limit: Maximum number of lines to read.

        Returns:
            `ReadResult` with raw (unformatted) content for the requested window,
                or an error if the file doesn't exist or can't be read.

                Line-number formatting is applied downstream by the filesystem
                middleware (`format_content_with_line_numbers`), not by backends:
                it adds the gutter, starts numbering at `offset + 1`, and splits
                lines longer than 5000 characters into continuation rows
                (e.g., `5.1`, `5.2`).
        """
        raise NotImplementedError

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """Async version of read."""
        return await asyncio.to_thread(self.read, file_path, offset, limit)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> "GrepResult":
        """Search for a literal text pattern in files.

        Args:
            pattern: Literal string to search for (NOT regex).

                Performs exact substring matching within file content.

                Example: `"TODO"` matches any line containing `"TODO"`

            path: Optional directory path to search in.

                If `None`, searches in current working directory.

                Example: `'/workspace/src'`

            glob: Optional glob pattern to filter which FILES to search.

                Filters by filename/path, not content.

                Supports standard glob wildcards:

                - `*` matches any characters in filename
                - `**` matches any directories recursively
                - `?` matches single character
                - `[abc]` matches one character from set

            max_count: Optional total cap on the number of matches returned
                across all files.

                `None` (the default) preserves existing backend behavior and
                returns every match. When set to an int, at most that many
                matches are returned; if more exist the search stops and the
                result is flagged with `GrepResult.truncated=True`. Exactly
                `max_count` matches with none dropped is reported complete
                (`truncated=False`). Interpreted as a total cap, not a per-file
                cap.

        Examples:
            - `'*.py'` - only search Python files
            - `'**/*.txt'` - search all `.txt` files recursively
            - `'src/**/*.js'` - search JS files under src/
            - `'test[0-9].txt'` - search `test0.txt`, `test1.txt`, etc.

        Returns:
            `GrepResult` with matches or error.

        Raises:
            NotImplementedError: If the backend does not implement `grep`.
        """
        raise NotImplementedError

    async def agrep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> "GrepResult":
        """Async version of `grep`.

        Wraps the sync call with an async timeout as a safety net. The timeout
        bounds how long the caller waits; it does not stop the worker thread
        created by `asyncio.to_thread`.

        `max_count` is forwarded when the concrete `grep` accepts it (so the
        search can bound itself); backends that don't accept it run uncapped and
        are trimmed afterward. Either way the return value is always passed
        through `_apply_grep_max_count` (a no-op when already within the cap), so
        callers get the same guarantee regardless of which path runs.
        """
        grep_kwargs = {"max_count": max_count} if _method_accepts_max_count(type(self), "grep") else {}
        grep_call = partial(self.grep, pattern, path, glob, **grep_kwargs)
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(grep_call),
                timeout=ASYNC_GREP_TIMEOUT,
            )
            return _apply_grep_max_count(result, max_count)
        except TimeoutError:
            logger.warning(
                "agrep timed out after %ds (pattern=%r, path=%r, glob=%r)",
                ASYNC_GREP_TIMEOUT,
                pattern,
                path,
                glob,
            )
            return GrepResult(
                error=f"Error: grep timed out after {ASYNC_GREP_TIMEOUT}s. Try a more specific pattern or a narrower path.",
            )

    def glob(self, pattern: str, path: str | None = None) -> "GlobResult":
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern with wildcards to match file paths.

                Supports standard glob syntax:

                - `*` matches any characters within a filename/directory
                - `**` matches any directories recursively
                - `?` matches a single character
                - `[abc]` matches one character from set

            path: Optional base directory to search from.

                If omitted, the backend chooses its default search root.

                The pattern is applied relative to this path.

        Returns:
            `GlobResult` with matching files or error.

        Raises:
            NotImplementedError: If the backend does not implement `glob`.
        """
        raise NotImplementedError

    async def aglob(self, pattern: str, path: str | None = None) -> "GlobResult":
        """Async version of `glob`."""
        return await asyncio.to_thread(self.glob, pattern, path)

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Write content to a file, creating it or overwriting it if it already exists.

        Args:
            file_path: Absolute path where the file should be written.

                Must start with '/'.
            content: String content to write to the file.

        Returns:
            WriteResult
        """
        raise NotImplementedError

    async def awrite(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Async version of write."""
        return await asyncio.to_thread(self.write, file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """Perform exact string replacements in an existing file.

        Args:
            file_path: Absolute path to the file to edit. Must start with `'/'`.
            old_string: Exact string to search for and replace.

                Must match exactly including whitespace and indentation.
            new_string: String to replace old_string with.

                Must be different from old_string.
            replace_all: If True, replace all occurrences.

                If `False` (default), `old_string` must be unique in the file or
                the edit fails.

        Returns:
            EditResult
        """
        raise NotImplementedError

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """Async version of edit."""
        return await asyncio.to_thread(self.edit, file_path, old_string, new_string, replace_all)

    def delete(self, file_path: str) -> DeleteResult:
        """Delete a path, recursively removing anything nested under it.

        This method is optional. Backends that do not implement it inherit this
        default, which raises `NotImplementedError`. Callers that need to support
        a mix of backends should guard with
        [`supports_delete`][deepagents.backends.protocol.supports_delete] before
        calling, or catch `NotImplementedError`.

        Deletion is recursive: it removes `file_path` plus everything nested
        under it. On hierarchical backends (e.g.
        [`FilesystemBackend`][deepagents.backends.filesystem.FilesystemBackend])
        that means a directory and its contents; on key-value backends it means
        the exact key plus every key sharing the `file_path` + "/" prefix.

        Args:
            file_path: Absolute path to delete (a file, or a directory/prefix to
                remove recursively). Must start with '/'.

        Returns:
            `DeleteResult` with the deleted path on success, or an error if
                nothing exists at or under the path or removal fails.

        Raises:
            NotImplementedError: If the backend does not implement `delete`.
        """
        raise NotImplementedError

    async def adelete(self, file_path: str) -> DeleteResult:
        """Async version of `delete`."""
        return await asyncio.to_thread(self.delete, file_path)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to the sandbox.

        This API is designed to allow developers to use it either directly or by
        exposing it to LLMs via custom tools.

        Args:
            files: List of (path, content) tuples to upload.

        Returns:
            List of `FileUploadResponse` objects, one per input file.

                Response order matches input order (`response[i] for files[i]`).

                Check the error field to determine success/failure per file.

        Examples:
            ```python
            responses = sandbox.upload_files(
                [
                    ("/app/config.json", b"{...}"),
                    ("/app/data.txt", b"content"),
                ]
            )
            ```
        """
        raise NotImplementedError

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Async version of upload_files."""
        return await asyncio.to_thread(self.upload_files, files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the sandbox.

        This API is designed to allow developers to use it either directly or
        by exposing it to LLMs via custom tools.

        Args:
            paths: List of file paths to download.

        Returns:
            List of `FileDownloadResponse` objects, one per input path.

                Response order matches input order (`response[i] for paths[i]`).

                Check the error field to determine success/failure per file.
        """
        raise NotImplementedError

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Async version of download_files."""
        return await asyncio.to_thread(self.download_files, paths)


@dataclass
class ExecuteResponse:
    """Result of code execution.

    Simplified schema optimized for LLM consumption.
    """

    output: str
    """Combined stdout and stderr output of the executed command."""

    exit_code: int | None = None
    """The process exit code.

    0 indicates success, non-zero indicates failure.
    """

    truncated: bool = False
    """Whether the output was truncated due to backend limitations."""


@dataclass(frozen=True, slots=True)
class ExecuteOffloadResult:
    """Result of [`BaseSandbox.execute_with_offload`][deepagents.backends.sandbox.BaseSandbox.execute_with_offload].

    `offloaded` describes the capture mechanism and is kept off `ExecuteResponse`
    (which an ordinary `execute` never sets).
    """

    offloaded: bool
    """Whether the output was left at the capture path.

    When `True`, `response.output` holds only a head/tail preview and the full
    output lives at the capture path on the sandbox filesystem. When `False`,
    `response.output` is the complete output.
    """

    response: ExecuteResponse
    """The command result. `response.truncated` indicates the output hit the size cap."""


class SandboxBackendProtocol(BackendProtocol):
    """Extension of `BackendProtocol` that adds shell command execution.

    Designed for backends running in isolated environments (containers, VMs,
    remote hosts).

    Adds `execute()`/`aexecute()` for shell commands and an `id` property.

    See `BaseSandbox` for a base class that implements all inherited file
    operations by delegating to `execute()`.
    """

    @property
    def id(self) -> str:
        """Unique identifier for the sandbox backend instance."""
        raise NotImplementedError

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command in the sandbox environment.

        Simplified interface optimized for LLM consumption.

        Args:
            command: Full shell command string to execute.
            timeout: Maximum time in seconds to wait for the command to complete.

                If None, uses the backend's default timeout.

                Callers should provide non-negative integer values for portable
                behavior across backends. A value of 0 may disable timeouts on
                backends that support no-timeout execution.

        Returns:
            `ExecuteResponse` with combined output, exit code, and truncation flag.
        """
        raise NotImplementedError

    async def aexecute(
        self,
        command: str,
        *,
        # ASYNC109 - timeout is a semantic parameter forwarded to the sync
        # implementation, not an asyncio.timeout() contract.
        timeout: int | None = None,  # noqa: ASYNC109
    ) -> ExecuteResponse:
        """Async version of execute."""
        # The middleware layer validates timeout support before calling, so
        # this guard only protects direct callers bypassing the middleware.
        if timeout is not None and execute_accepts_timeout(type(self)):
            return await asyncio.to_thread(self.execute, command, timeout=timeout)
        return await asyncio.to_thread(self.execute, command)


@lru_cache(maxsize=256)
def _method_accepts_max_count(cls: type[BackendProtocol], method_name: Literal["grep", "agrep"]) -> bool:
    """Check whether a backend method accepts the optional `max_count` keyword."""
    try:
        sig = inspect.signature(getattr(cls, method_name))
    except (AttributeError, ValueError, TypeError):
        logger.warning(
            "Could not inspect signature of %s.%s; assuming max_count is not supported. "
            "The cap will be enforced after the search instead of bounding it, so a huge "
            "result set is fully materialized before being trimmed.",
            cls.__qualname__,
            method_name,
            exc_info=True,
        )
        return False
    return "max_count" in sig.parameters or any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in sig.parameters.values())


@lru_cache(maxsize=128)
def execute_accepts_timeout(cls: type[SandboxBackendProtocol]) -> bool:
    """Check whether a backend class's `execute` accepts a `timeout` kwarg.

    Older backend packages didn't lower-bound their SDK dependency, so they
    may not accept the `timeout` keyword added to
    [`SandboxBackendProtocol`][deepagents.backends.protocol.SandboxBackendProtocol].

    Results are cached per class to avoid repeated introspection overhead.
    """
    try:
        sig = inspect.signature(cls.execute)
    except (ValueError, TypeError):
        logger.warning(
            "Could not inspect signature of %s.execute; assuming timeout is not supported. This may indicate a backend packaging issue.",
            cls.__qualname__,
            exc_info=True,
        )
        return False
    else:
        return "timeout" in sig.parameters


def _supports_delete(backend: BackendProtocol) -> bool:
    """Check whether a backend implements `delete`.

    `delete` is optional: backends that don't override it inherit the
    `NotImplementedError` default from
    [`BackendProtocol`][deepagents.backends.protocol.BackendProtocol]. This
    helper lets callers detect support without invoking the method and
    triggering the error.

    Args:
        backend: The backend instance to check.

    Returns:
        True if the backend overrides `delete`, False otherwise.
    """
    return type(backend).delete is not BackendProtocol.delete

"""Anthropic text editor and memory tool middleware.

This module provides client-side implementations of Anthropic's text editor and
memory tools using schema-less tool definitions and tool call interception.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

from langchain_core.messages import ToolMessage
from langgraph.types import Command
from typing_extensions import NotRequired, TypedDict

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain.tools.tool_node import ToolCallRequest

# Tool type constants
TEXT_EDITOR_TOOL_TYPE = "text_editor_20250728"
TEXT_EDITOR_TOOL_NAME = "str_replace_based_edit_tool"
MEMORY_TOOL_TYPE = "memory_20250818"
MEMORY_TOOL_NAME = "memory"

MEMORY_SYSTEM_PROMPT = """IMPORTANT: ALWAYS VIEW YOUR MEMORY DIRECTORY BEFORE \
DOING ANYTHING ELSE.
MEMORY PROTOCOL:
1. Use the `view` command of your `memory` tool to check for earlier progress.
2. ... (work on the task) ...
   - As you make progress, record status / progress / thoughts etc in your memory.
ASSUME INTERRUPTION: Your context window might be reset at any moment, so you risk \
losing any progress that is not recorded in your memory directory."""


class FileData(TypedDict):
    """Data structure for storing file contents."""

    content: list[str]
    """Lines of the file."""

    created_at: str
    """ISO 8601 timestamp of file creation."""

    modified_at: str
    """ISO 8601 timestamp of last modification."""


def files_reducer(
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


class AnthropicToolsState(AgentState):
    """State schema for Anthropic text editor and memory tools."""

    text_editor_files: NotRequired[Annotated[dict[str, FileData], files_reducer]]
    """Virtual file system for text editor tools."""

    memory_files: NotRequired[Annotated[dict[str, FileData], files_reducer]]
    """Virtual file system for memory tools."""


def _validate_path(path: str, *, allowed_prefixes: Sequence[str] | None = None) -> str:
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


def _list_directory(files: dict[str, FileData], path: str) -> list[str]:
    """List files in a directory.

    Args:
        files: Files dict.
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


class _StateClaudeFileToolMiddleware(AgentMiddleware):
    """Base class for state-based file tool middleware (internal)."""

    state_schema = AnthropicToolsState

    def __init__(
        self,
        *,
        tool_type: str,
        tool_name: str,
        state_key: str,
        allowed_path_prefixes: Sequence[str] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize the middleware.

        Args:
            tool_type: Tool type identifier.
            tool_name: Tool name.
            state_key: State key for file storage.
            allowed_path_prefixes: Optional list of allowed path prefixes.
            system_prompt: Optional system prompt to inject.
        """
        self.tool_type = tool_type
        self.tool_name = tool_name
        self.state_key = state_key
        self.allowed_prefixes = allowed_path_prefixes
        self.system_prompt = system_prompt

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject tool and optional system prompt."""
        # Add tool
        tools = list(request.tools or [])
        tools.append(
            {
                "type": self.tool_type,
                "name": self.tool_name,
            }
        )
        request.tools = tools

        # Inject system prompt if provided
        if self.system_prompt:
            request.system_prompt = (
                request.system_prompt + "\n\n" + self.system_prompt
                if request.system_prompt
                else self.system_prompt
            )

        return handler(request)

    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]
    ) -> ToolMessage | Command:
        """Intercept tool calls."""
        tool_call = request.tool_call
        tool_name = tool_call.get("name")

        if tool_name != self.tool_name:
            return handler(request)

        # Handle tool call
        try:
            args = tool_call.get("args", {})
            command = args.get("command")
            state = request.state

            if command == "view":
                return self._handle_view(args, state, tool_call["id"])
            if command == "create":
                return self._handle_create(args, state, tool_call["id"])
            if command == "str_replace":
                return self._handle_str_replace(args, state, tool_call["id"])
            if command == "insert":
                return self._handle_insert(args, state, tool_call["id"])
            if command == "delete":
                return self._handle_delete(args, state, tool_call["id"])
            if command == "rename":
                return self._handle_rename(args, state, tool_call["id"])

            msg = f"Unknown command: {command}"
            return ToolMessage(
                content=msg,
                tool_call_id=tool_call["id"],
                name=tool_name,
                status="error",
            )
        except (ValueError, FileNotFoundError) as e:
            return ToolMessage(
                content=str(e),
                tool_call_id=tool_call["id"],
                name=tool_name,
                status="error",
            )

    def _handle_view(
        self, args: dict, state: AnthropicToolsState, tool_call_id: str | None
    ) -> Command:
        """Handle view command."""
        path = args["path"]
        normalized_path = _validate_path(path, allowed_prefixes=self.allowed_prefixes)

        files = cast("dict[str, Any]", state.get(self.state_key, {}))
        file_data = files.get(normalized_path)

        if file_data is None:
            # Try directory listing
            matching = _list_directory(files, normalized_path)

            if matching:
                content = "\n".join(matching)
                return Command(
                    update={
                        "messages": [
                            ToolMessage(
                                content=content,
                                tool_call_id=tool_call_id,
                                name=self.tool_name,
                            )
                        ]
                    }
                )

            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)

        # Format file content with line numbers
        lines_content = file_data["content"]
        formatted_lines = [f"{i + 1}|{line}" for i, line in enumerate(lines_content)]
        content = "\n".join(formatted_lines)

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=content,
                        tool_call_id=tool_call_id,
                        name=self.tool_name,
                    )
                ]
            }
        )

    def _handle_create(
        self, args: dict, state: AnthropicToolsState, tool_call_id: str | None
    ) -> Command:
        """Handle create command."""
        path = args["path"]
        file_text = args["file_text"]

        normalized_path = _validate_path(path, allowed_prefixes=self.allowed_prefixes)

        # Get existing files
        files = cast("dict[str, Any]", state.get(self.state_key, {}))
        existing = files.get(normalized_path)

        # Create file data
        now = datetime.now(timezone.utc).isoformat()
        created_at = existing["created_at"] if existing else now

        content_lines = file_text.split("\n")

        return Command(
            update={
                self.state_key: {
                    normalized_path: {
                        "content": content_lines,
                        "created_at": created_at,
                        "modified_at": now,
                    }
                },
                "messages": [
                    ToolMessage(
                        content=f"File created: {path}",
                        tool_call_id=tool_call_id,
                        name=self.tool_name,
                    )
                ],
            }
        )

    def _handle_str_replace(
        self, args: dict, state: AnthropicToolsState, tool_call_id: str | None
    ) -> Command:
        """Handle str_replace command."""
        path = args["path"]
        old_str = args["old_str"]
        new_str = args.get("new_str", "")

        normalized_path = _validate_path(path, allowed_prefixes=self.allowed_prefixes)

        # Read file
        files = cast("dict[str, Any]", state.get(self.state_key, {}))
        file_data = files.get(normalized_path)
        if file_data is None:
            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)

        lines_content = file_data["content"]
        content = "\n".join(lines_content)

        # Replace string
        if old_str not in content:
            msg = f"String not found in file: {old_str}"
            raise ValueError(msg)

        new_content = content.replace(old_str, new_str, 1)
        new_lines = new_content.split("\n")

        # Update file
        now = datetime.now(timezone.utc).isoformat()

        return Command(
            update={
                self.state_key: {
                    normalized_path: {
                        "content": new_lines,
                        "created_at": file_data["created_at"],
                        "modified_at": now,
                    }
                },
                "messages": [
                    ToolMessage(
                        content=f"String replaced in {path}",
                        tool_call_id=tool_call_id,
                        name=self.tool_name,
                    )
                ],
            }
        )

    def _handle_insert(
        self, args: dict, state: AnthropicToolsState, tool_call_id: str | None
    ) -> Command:
        """Handle insert command."""
        path = args["path"]
        insert_line = args["insert_line"]
        text_to_insert = args["new_str"]

        normalized_path = _validate_path(path, allowed_prefixes=self.allowed_prefixes)

        # Read file
        files = cast("dict[str, Any]", state.get(self.state_key, {}))
        file_data = files.get(normalized_path)
        if file_data is None:
            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)

        lines_content = file_data["content"]
        new_lines = text_to_insert.split("\n")

        # Insert after insert_line (0-indexed)
        updated_lines = lines_content[:insert_line] + new_lines + lines_content[insert_line:]

        # Update file
        now = datetime.now(timezone.utc).isoformat()

        return Command(
            update={
                self.state_key: {
                    normalized_path: {
                        "content": updated_lines,
                        "created_at": file_data["created_at"],
                        "modified_at": now,
                    }
                },
                "messages": [
                    ToolMessage(
                        content=f"Text inserted in {path}",
                        tool_call_id=tool_call_id,
                        name=self.tool_name,
                    )
                ],
            }
        )

    def _handle_delete(
        self,
        args: dict,
        state: AnthropicToolsState,  # noqa: ARG002
        tool_call_id: str | None,
    ) -> Command:
        """Handle delete command."""
        path = args["path"]

        normalized_path = _validate_path(path, allowed_prefixes=self.allowed_prefixes)

        return Command(
            update={
                self.state_key: {normalized_path: None},
                "messages": [
                    ToolMessage(
                        content=f"File deleted: {path}",
                        tool_call_id=tool_call_id,
                        name=self.tool_name,
                    )
                ],
            }
        )

    def _handle_rename(
        self, args: dict, state: AnthropicToolsState, tool_call_id: str | None
    ) -> Command:
        """Handle rename command."""
        old_path = args["old_path"]
        new_path = args["new_path"]

        normalized_old = _validate_path(old_path, allowed_prefixes=self.allowed_prefixes)
        normalized_new = _validate_path(new_path, allowed_prefixes=self.allowed_prefixes)

        # Read file
        files = cast("dict[str, Any]", state.get(self.state_key, {}))
        file_data = files.get(normalized_old)
        if file_data is None:
            msg = f"File not found: {old_path}"
            raise ValueError(msg)

        # Update timestamp
        now = datetime.now(timezone.utc).isoformat()
        file_data_copy = file_data.copy()
        file_data_copy["modified_at"] = now

        return Command(
            update={
                self.state_key: {
                    normalized_old: None,
                    normalized_new: file_data_copy,
                },
                "messages": [
                    ToolMessage(
                        content=f"File renamed: {old_path} -> {new_path}",
                        tool_call_id=tool_call_id,
                        name=self.tool_name,
                    )
                ],
            }
        )


class StateClaudeTextEditorMiddleware(_StateClaudeFileToolMiddleware):
    """State-based text editor tool middleware.

    Provides Anthropic's text_editor tool using LangGraph state for storage.
    Files persist for the conversation thread.

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain.agents.middleware import StateTextEditorToolMiddleware

        agent = create_agent(
            model=model,
            tools=[],
            middleware=[StateTextEditorToolMiddleware()],
        )
        ```
    """

    def __init__(
        self,
        *,
        allowed_path_prefixes: Sequence[str] | None = None,
    ) -> None:
        """Initialize the text editor middleware.

        Args:
            allowed_path_prefixes: Optional list of allowed path prefixes.
                If specified, only paths starting with these prefixes are allowed.
        """
        super().__init__(
            tool_type=TEXT_EDITOR_TOOL_TYPE,
            tool_name=TEXT_EDITOR_TOOL_NAME,
            state_key="text_editor_files",
            allowed_path_prefixes=allowed_path_prefixes,
        )


class StateClaudeMemoryMiddleware(_StateClaudeFileToolMiddleware):
    """State-based memory tool middleware.

    Provides Anthropic's memory tool using LangGraph state for storage.
    Files persist for the conversation thread. Enforces /memories prefix
    and injects Anthropic's recommended system prompt.

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain.agents.middleware import StateMemoryToolMiddleware

        agent = create_agent(
            model=model,
            tools=[],
            middleware=[StateMemoryToolMiddleware()],
        )
        ```
    """

    def __init__(
        self,
        *,
        allowed_path_prefixes: Sequence[str] | None = None,
        system_prompt: str = MEMORY_SYSTEM_PROMPT,
    ) -> None:
        """Initialize the memory middleware.

        Args:
            allowed_path_prefixes: Optional list of allowed path prefixes.
                Defaults to ["/memories"].
            system_prompt: System prompt to inject. Defaults to Anthropic's
                recommended memory prompt.
        """
        super().__init__(
            tool_type=MEMORY_TOOL_TYPE,
            tool_name=MEMORY_TOOL_NAME,
            state_key="memory_files",
            allowed_path_prefixes=allowed_path_prefixes or ["/memories"],
            system_prompt=system_prompt,
        )


class _FilesystemClaudeFileToolMiddleware(AgentMiddleware):
    """Base class for filesystem-based file tool middleware (internal)."""

    def __init__(
        self,
        *,
        tool_type: str,
        tool_name: str,
        root_path: str,
        allowed_prefixes: list[str] | None = None,
        max_file_size_mb: int = 10,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize the middleware.

        Args:
            tool_type: Tool type identifier.
            tool_name: Tool name.
            root_path: Root directory for file operations.
            allowed_prefixes: Optional list of allowed virtual path prefixes.
            max_file_size_mb: Maximum file size in MB.
            system_prompt: Optional system prompt to inject.
        """
        self.tool_type = tool_type
        self.tool_name = tool_name
        self.root_path = Path(root_path).resolve()
        self.allowed_prefixes = allowed_prefixes or ["/"]
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.system_prompt = system_prompt

        # Create root directory if it doesn't exist
        self.root_path.mkdir(parents=True, exist_ok=True)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject tool and optional system prompt."""
        # Add tool
        tools = list(request.tools or [])
        tools.append(
            {
                "type": self.tool_type,
                "name": self.tool_name,
            }
        )
        request.tools = tools

        # Inject system prompt if provided
        if self.system_prompt:
            request.system_prompt = (
                request.system_prompt + "\n\n" + self.system_prompt
                if request.system_prompt
                else self.system_prompt
            )

        return handler(request)

    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]
    ) -> ToolMessage | Command:
        """Intercept tool calls."""
        tool_call = request.tool_call
        tool_name = tool_call.get("name")

        if tool_name != self.tool_name:
            return handler(request)

        # Handle tool call
        try:
            args = tool_call.get("args", {})
            command = args.get("command")

            if command == "view":
                return self._handle_view(args, tool_call["id"])
            if command == "create":
                return self._handle_create(args, tool_call["id"])
            if command == "str_replace":
                return self._handle_str_replace(args, tool_call["id"])
            if command == "insert":
                return self._handle_insert(args, tool_call["id"])
            if command == "delete":
                return self._handle_delete(args, tool_call["id"])
            if command == "rename":
                return self._handle_rename(args, tool_call["id"])

            msg = f"Unknown command: {command}"
            return ToolMessage(
                content=msg,
                tool_call_id=tool_call["id"],
                name=tool_name,
                status="error",
            )
        except (ValueError, FileNotFoundError) as e:
            return ToolMessage(
                content=str(e),
                tool_call_id=tool_call["id"],
                name=tool_name,
                status="error",
            )

    def _validate_and_resolve_path(self, path: str) -> Path:
        """Validate and resolve a virtual path to filesystem path.

        Args:
            path: Virtual path (e.g., /file.txt or /src/main.py).

        Returns:
            Resolved absolute filesystem path within root_path.

        Raises:
            ValueError: If path contains traversal attempts, escapes root directory,
                or violates allowed_prefixes restrictions.
        """
        # Normalize path
        if not path.startswith("/"):
            path = "/" + path

        # Check for path traversal
        if ".." in path or "~" in path:
            msg = "Path traversal not allowed"
            raise ValueError(msg)

        # Convert virtual path to filesystem path
        # Remove leading / and resolve relative to root
        relative = path.lstrip("/")
        full_path = (self.root_path / relative).resolve()

        # Ensure path is within root
        try:
            full_path.relative_to(self.root_path)
        except ValueError:
            msg = f"Path outside root directory: {path}"
            raise ValueError(msg) from None

        # Check allowed prefixes
        virtual_path = "/" + str(full_path.relative_to(self.root_path))
        if self.allowed_prefixes:
            allowed = any(
                virtual_path.startswith(prefix) or virtual_path == prefix.rstrip("/")
                for prefix in self.allowed_prefixes
            )
            if not allowed:
                msg = f"Path must start with one of: {self.allowed_prefixes}"
                raise ValueError(msg)

        return full_path

    def _handle_view(self, args: dict, tool_call_id: str | None) -> Command:
        """Handle view command."""
        path = args["path"]
        full_path = self._validate_and_resolve_path(path)

        if not full_path.exists() or not full_path.is_file():
            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)

        # Check file size
        if full_path.stat().st_size > self.max_file_size_bytes:
            msg = f"File too large: {path} exceeds {self.max_file_size_bytes / 1024 / 1024}MB"
            raise ValueError(msg)

        # Read file
        try:
            content = full_path.read_text()
        except UnicodeDecodeError as e:
            msg = f"Cannot decode file {path}: {e}"
            raise ValueError(msg) from e

        # Format with line numbers
        lines = content.split("\n")
        # Remove trailing newline's empty string if present
        if lines and lines[-1] == "":
            lines = lines[:-1]
        formatted_lines = [f"{i + 1}|{line}" for i, line in enumerate(lines)]
        formatted_content = "\n".join(formatted_lines)

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=formatted_content,
                        tool_call_id=tool_call_id,
                        name=self.tool_name,
                    )
                ]
            }
        )

    def _handle_create(self, args: dict, tool_call_id: str | None) -> Command:
        """Handle create command."""
        path = args["path"]
        file_text = args["file_text"]

        full_path = self._validate_and_resolve_path(path)

        # Create parent directories
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        full_path.write_text(file_text + "\n")

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"File created: {path}",
                        tool_call_id=tool_call_id,
                        name=self.tool_name,
                    )
                ]
            }
        )

    def _handle_str_replace(self, args: dict, tool_call_id: str | None) -> Command:
        """Handle str_replace command."""
        path = args["path"]
        old_str = args["old_str"]
        new_str = args.get("new_str", "")

        full_path = self._validate_and_resolve_path(path)

        if not full_path.exists():
            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)

        # Read file
        content = full_path.read_text()

        # Replace string
        if old_str not in content:
            msg = f"String not found in file: {old_str}"
            raise ValueError(msg)

        new_content = content.replace(old_str, new_str, 1)

        # Write back
        full_path.write_text(new_content)

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"String replaced in {path}",
                        tool_call_id=tool_call_id,
                        name=self.tool_name,
                    )
                ]
            }
        )

    def _handle_insert(self, args: dict, tool_call_id: str | None) -> Command:
        """Handle insert command."""
        path = args["path"]
        insert_line = args["insert_line"]
        text_to_insert = args["new_str"]

        full_path = self._validate_and_resolve_path(path)

        if not full_path.exists():
            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)

        # Read file
        content = full_path.read_text()
        lines = content.split("\n")
        # Handle trailing newline
        if lines and lines[-1] == "":
            lines = lines[:-1]
            had_trailing_newline = True
        else:
            had_trailing_newline = False

        new_lines = text_to_insert.split("\n")

        # Insert after insert_line (0-indexed)
        updated_lines = lines[:insert_line] + new_lines + lines[insert_line:]

        # Write back
        new_content = "\n".join(updated_lines)
        if had_trailing_newline:
            new_content += "\n"
        full_path.write_text(new_content)

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Text inserted in {path}",
                        tool_call_id=tool_call_id,
                        name=self.tool_name,
                    )
                ]
            }
        )

    def _handle_delete(self, args: dict, tool_call_id: str | None) -> Command:
        """Handle delete command."""
        import shutil

        path = args["path"]
        full_path = self._validate_and_resolve_path(path)

        if full_path.is_file():
            full_path.unlink()
        elif full_path.is_dir():
            shutil.rmtree(full_path)
        # If doesn't exist, silently succeed

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"File deleted: {path}",
                        tool_call_id=tool_call_id,
                        name=self.tool_name,
                    )
                ]
            }
        )

    def _handle_rename(self, args: dict, tool_call_id: str | None) -> Command:
        """Handle rename command."""
        old_path = args["old_path"]
        new_path = args["new_path"]

        old_full = self._validate_and_resolve_path(old_path)
        new_full = self._validate_and_resolve_path(new_path)

        if not old_full.exists():
            msg = f"File not found: {old_path}"
            raise ValueError(msg)

        # Create parent directory for new path
        new_full.parent.mkdir(parents=True, exist_ok=True)

        # Rename
        old_full.rename(new_full)

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"File renamed: {old_path} -> {new_path}",
                        tool_call_id=tool_call_id,
                        name=self.tool_name,
                    )
                ]
            }
        )


class FilesystemClaudeTextEditorMiddleware(_FilesystemClaudeFileToolMiddleware):
    """Filesystem-based text editor tool middleware.

    Provides Anthropic's text_editor tool using local filesystem for storage.
    User handles persistence via volumes, git, or other mechanisms.

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain.agents.middleware import FilesystemTextEditorToolMiddleware

        agent = create_agent(
            model=model,
            tools=[],
            middleware=[FilesystemTextEditorToolMiddleware(root_path="/workspace")],
        )
        ```
    """

    def __init__(
        self,
        *,
        root_path: str,
        allowed_prefixes: list[str] | None = None,
        max_file_size_mb: int = 10,
    ) -> None:
        """Initialize the text editor middleware.

        Args:
            root_path: Root directory for file operations.
            allowed_prefixes: Optional list of allowed virtual path prefixes (default: ["/"]).
            max_file_size_mb: Maximum file size in MB (default: 10).
        """
        super().__init__(
            tool_type=TEXT_EDITOR_TOOL_TYPE,
            tool_name=TEXT_EDITOR_TOOL_NAME,
            root_path=root_path,
            allowed_prefixes=allowed_prefixes,
            max_file_size_mb=max_file_size_mb,
        )


class FilesystemClaudeMemoryMiddleware(_FilesystemClaudeFileToolMiddleware):
    """Filesystem-based memory tool middleware.

    Provides Anthropic's memory tool using local filesystem for storage.
    User handles persistence via volumes, git, or other mechanisms.
    Enforces /memories prefix and injects Anthropic's recommended system prompt.

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain.agents.middleware import FilesystemMemoryToolMiddleware

        agent = create_agent(
            model=model,
            tools=[],
            middleware=[FilesystemMemoryToolMiddleware(root_path="/workspace")],
        )
        ```
    """

    def __init__(
        self,
        *,
        root_path: str,
        allowed_prefixes: list[str] | None = None,
        max_file_size_mb: int = 10,
        system_prompt: str = MEMORY_SYSTEM_PROMPT,
    ) -> None:
        """Initialize the memory middleware.

        Args:
            root_path: Root directory for file operations.
            allowed_prefixes: Optional list of allowed virtual path prefixes.
                Defaults to ["/memories"].
            max_file_size_mb: Maximum file size in MB (default: 10).
            system_prompt: System prompt to inject. Defaults to Anthropic's
                recommended memory prompt.
        """
        super().__init__(
            tool_type=MEMORY_TOOL_TYPE,
            tool_name=MEMORY_TOOL_NAME,
            root_path=root_path,
            allowed_prefixes=allowed_prefixes or ["/memories"],
            max_file_size_mb=max_file_size_mb,
            system_prompt=system_prompt,
        )


__all__ = [
    "AnthropicToolsState",
    "FileData",
    "FilesystemClaudeMemoryMiddleware",
    "FilesystemClaudeTextEditorMiddleware",
    "StateClaudeMemoryMiddleware",
    "StateClaudeTextEditorMiddleware",
]

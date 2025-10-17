"""Middleware for providing filesystem tools to an agent."""
# ruff: noqa: E501

from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any

from typing_extensions import NotRequired

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.config import get_config
from langgraph.runtime import Runtime, get_runtime
from langgraph.store.base import BaseStore, Item
from langgraph.types import Command

from langchain.agents._internal.file_utils import (
    FileData,
    append_memories_prefix,
    check_empty_content,
    create_file_data,
    file_data_reducer,
    file_data_to_string,
    format_content_with_line_numbers,
    has_memories_prefix,
    strip_memories_prefix,
    update_file_data,
    validate_path,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.tools.tool_node import InjectedState

# Constants
LONGTERM_MEMORY_PREFIX = "/memories/"
DEFAULT_READ_OFFSET = 0
DEFAULT_READ_LIMIT = 2000


class FilesystemState(AgentState):
    """State for the filesystem middleware."""

    files: Annotated[NotRequired[dict[str, FileData]], file_data_reducer]
    """Files in the filesystem."""


LIST_FILES_TOOL_DESCRIPTION = """Lists all files in the filesystem, optionally filtering by directory.

Usage:
- The list_files tool will return a list of all files in the filesystem.
- You can optionally provide a path parameter to list files in a specific directory.
- This is very useful for exploring the file system and finding the right file to read or edit.
- You should almost ALWAYS use this tool before using the Read or Edit tools."""
LIST_FILES_TOOL_DESCRIPTION_LONGTERM_SUPPLEMENT = f"\n- Files from the longterm filesystem will be prefixed with the {LONGTERM_MEMORY_PREFIX} path."

READ_FILE_TOOL_DESCRIPTION = """Reads a file from the filesystem. You can access any file directly by using this tool.
Assume this tool is able to read all files on the machine. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.

Usage:
- The file_path parameter must be an absolute path, not a relative path
- By default, it reads up to 2000 lines starting from the beginning of the file
- You can optionally specify a line offset and limit (especially handy for long files), but it's recommended to read the whole file by not providing these parameters
- Any lines longer than 2000 characters will be truncated
- Results are returned using cat -n format, with line numbers starting at 1
- You have the capability to call multiple tools in a single response. It is always better to speculatively read multiple files as a batch that are potentially useful.
- If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents.
- You should ALWAYS make sure a file has been read before editing it."""
READ_FILE_TOOL_DESCRIPTION_LONGTERM_SUPPLEMENT = f"\n- file_paths prefixed with the {LONGTERM_MEMORY_PREFIX} path will be read from the longterm filesystem."

EDIT_FILE_TOOL_DESCRIPTION = """Performs exact string replacements in files.

Usage:
- You must use your `Read` tool at least once in the conversation before editing. This tool will error if you attempt an edit without reading the file.
- When editing text from Read tool output, ensure you preserve the exact indentation (tabs/spaces) as it appears AFTER the line number prefix. The line number prefix format is: spaces + line number + tab. Everything after that tab is the actual file content to match. Never include any part of the line number prefix in the old_string or new_string.
- ALWAYS prefer editing existing files. NEVER write new files unless explicitly required.
- Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.
- The edit will FAIL if `old_string` is not unique in the file. Either provide a larger string with more surrounding context to make it unique or use `replace_all` to change every instance of `old_string`.
- Use `replace_all` for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance."""
EDIT_FILE_TOOL_DESCRIPTION_LONGTERM_SUPPLEMENT = f"\n- You can edit files in the longterm filesystem by prefixing the filename with the {LONGTERM_MEMORY_PREFIX} path."

WRITE_FILE_TOOL_DESCRIPTION = """Writes to a new file in the filesystem.

Usage:
- The file_path parameter must be an absolute path, not a relative path
- The content parameter must be a string
- The write_file tool will create the a new file.
- Prefer to edit existing files over creating new ones when possible.
- file_paths prefixed with the /memories/ path will be written to the longterm filesystem."""
WRITE_FILE_TOOL_DESCRIPTION_LONGTERM_SUPPLEMENT = f"\n- file_paths prefixed with the {LONGTERM_MEMORY_PREFIX} path will be written to the longterm filesystem."

FILESYSTEM_SYSTEM_PROMPT = """## Filesystem Tools `ls`, `read_file`, `write_file`, `edit_file`

You have access to a filesystem which you can interact with using these tools.
All file paths must start with a /.

- ls: list all files in the filesystem
- read_file: read a file from the filesystem
- write_file: write to a file in the filesystem
- edit_file: edit a file in the filesystem"""
FILESYSTEM_SYSTEM_PROMPT_LONGTERM_SUPPLEMENT = f"""

You also have access to a longterm filesystem in which you can store files that you want to keep around for longer than the current conversation.
In order to interact with the longterm filesystem, you can use those same tools, but filenames must be prefixed with the {LONGTERM_MEMORY_PREFIX} path.
Remember, to interact with the longterm filesystem, you must prefix the filename with the {LONGTERM_MEMORY_PREFIX} path."""


def _get_namespace() -> tuple[str] | tuple[str, str]:
    """Get the namespace for longterm filesystem storage.

    Returns a tuple for organizing files in the store. If an assistant_id is available
    in the config metadata, returns a 2-tuple of (assistant_id, "filesystem") to provide
    per-assistant isolation. Otherwise, returns a 1-tuple of ("filesystem",) for shared storage.

    Returns:
        Namespace tuple for store operations, either `(assistant_id, "filesystem")` or `("filesystem",)`.
    """
    namespace = "filesystem"
    config = get_config()
    if config is None:
        return (namespace,)
    assistant_id = config.get("metadata", {}).get("assistant_id")
    if assistant_id is None:
        return (namespace,)
    return (assistant_id, "filesystem")


def _get_store(runtime: Runtime[Any]) -> BaseStore:
    """Get the store from the runtime, raising an error if unavailable.

    Args:
        runtime: The LangGraph runtime containing the store.

    Returns:
        The BaseStore instance for longterm file storage.

    Raises:
        ValueError: If longterm memory is enabled but no store is available in runtime.
    """
    if runtime.store is None:
        msg = "Longterm memory is enabled, but no store is available"
        raise ValueError(msg)
    return runtime.store


def _convert_store_item_to_file_data(store_item: Item) -> FileData:
    """Convert a store Item to FileData format.

    Args:
        store_item: The store Item containing file data.

    Returns:
        FileData with content, created_at, and modified_at fields.

    Raises:
        ValueError: If required fields are missing or have incorrect types.
    """
    if "content" not in store_item.value or not isinstance(store_item.value["content"], list):
        msg = f"Store item does not contain valid content field. Got: {store_item.value.keys()}"
        raise ValueError(msg)
    if "created_at" not in store_item.value or not isinstance(store_item.value["created_at"], str):
        msg = f"Store item does not contain valid created_at field. Got: {store_item.value.keys()}"
        raise ValueError(msg)
    if "modified_at" not in store_item.value or not isinstance(
        store_item.value["modified_at"], str
    ):
        msg = f"Store item does not contain valid modified_at field. Got: {store_item.value.keys()}"
        raise ValueError(msg)
    return FileData(
        content=store_item.value["content"],
        created_at=store_item.value["created_at"],
        modified_at=store_item.value["modified_at"],
    )


def _convert_file_data_to_store_item(file_data: FileData) -> dict[str, Any]:
    """Convert FileData to a dict suitable for store.put().

    Args:
        file_data: The FileData to convert.

    Returns:
        Dictionary with content, created_at, and modified_at fields.
    """
    return {
        "content": file_data["content"],
        "created_at": file_data["created_at"],
        "modified_at": file_data["modified_at"],
    }


def _get_file_data_from_state(state: FilesystemState, file_path: str) -> FileData:
    """Retrieve file data from the agent's state.

    Args:
        state: The current filesystem state.
        file_path: The path of the file to retrieve.

    Returns:
        The FileData for the requested file.

    Raises:
        ValueError: If the file is not found in state.
    """
    mock_filesystem = state.get("files", {})
    if file_path not in mock_filesystem:
        msg = f"File '{file_path}' not found"
        raise ValueError(msg)
    return mock_filesystem[file_path]


def _ls_tool_generator(
    custom_description: str | None = None, *, long_term_memory: bool
) -> BaseTool:
    """Generate the ls (list files) tool.

    Args:
        custom_description: Optional custom description for the tool.
        long_term_memory: Whether to enable longterm memory support.

    Returns:
        Configured ls tool that lists files from state and optionally from longterm store.
    """
    tool_description = LIST_FILES_TOOL_DESCRIPTION
    if custom_description:
        tool_description = custom_description
    elif long_term_memory:
        tool_description += LIST_FILES_TOOL_DESCRIPTION_LONGTERM_SUPPLEMENT

    def _get_filenames_from_state(state: FilesystemState) -> list[str]:
        """Extract list of filenames from the filesystem state.

        Args:
            state: The current filesystem state.

        Returns:
            List of file paths in the state.
        """
        files_dict = state.get("files", {})
        return list(files_dict.keys())

    def _filter_files_by_path(filenames: list[str], path: str | None) -> list[str]:
        """Filter filenames by path prefix.

        Args:
            filenames: List of file paths to filter.
            path: Optional path prefix to filter by.

        Returns:
            Filtered list of file paths matching the prefix.
        """
        if path is None:
            return filenames
        normalized_path = validate_path(path)
        return [f for f in filenames if f.startswith(normalized_path)]

    if long_term_memory:

        @tool(description=tool_description)
        def ls(
            state: Annotated[FilesystemState, InjectedState], path: str | None = None
        ) -> list[str]:
            files = _get_filenames_from_state(state)
            # Add filenames from longterm memory
            runtime = get_runtime()
            store = _get_store(runtime)
            namespace = _get_namespace()
            longterm_files = store.search(namespace)
            longterm_files_prefixed = [append_memories_prefix(f.key) for f in longterm_files]
            files.extend(longterm_files_prefixed)
            return _filter_files_by_path(files, path)
    else:

        @tool(description=tool_description)
        def ls(
            state: Annotated[FilesystemState, InjectedState], path: str | None = None
        ) -> list[str]:
            files = _get_filenames_from_state(state)
            return _filter_files_by_path(files, path)

    return ls


def _read_file_tool_generator(
    custom_description: str | None = None, *, long_term_memory: bool
) -> BaseTool:
    """Generate the read_file tool.

    Args:
        custom_description: Optional custom description for the tool.
        long_term_memory: Whether to enable longterm memory support.

    Returns:
        Configured read_file tool that reads files from state and optionally from longterm store.
    """
    tool_description = READ_FILE_TOOL_DESCRIPTION
    if custom_description:
        tool_description = custom_description
    elif long_term_memory:
        tool_description += READ_FILE_TOOL_DESCRIPTION_LONGTERM_SUPPLEMENT

    def _read_file_data_content(file_data: FileData, offset: int, limit: int) -> str:
        """Read and format file content with line numbers.

        Args:
            file_data: The file data to read.
            offset: Line offset to start reading from (0-indexed).
            limit: Maximum number of lines to read.

        Returns:
            Formatted file content with line numbers, or an error message.
        """
        content = file_data_to_string(file_data)
        empty_msg = check_empty_content(content)
        if empty_msg:
            return empty_msg
        lines = content.splitlines()
        start_idx = offset
        end_idx = min(start_idx + limit, len(lines))
        if start_idx >= len(lines):
            return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"
        selected_lines = lines[start_idx:end_idx]
        return format_content_with_line_numbers(
            selected_lines, format_style="tab", start_line=start_idx + 1
        )

    if long_term_memory:

        @tool(description=tool_description)
        def read_file(
            file_path: str,
            state: Annotated[FilesystemState, InjectedState],
            offset: int = DEFAULT_READ_OFFSET,
            limit: int = DEFAULT_READ_LIMIT,
        ) -> str:
            file_path = validate_path(file_path)
            if has_memories_prefix(file_path):
                stripped_file_path = strip_memories_prefix(file_path)
                runtime = get_runtime()
                store = _get_store(runtime)
                namespace = _get_namespace()
                item: Item | None = store.get(namespace, stripped_file_path)
                if item is None:
                    return f"Error: File '{file_path}' not found"
                file_data = _convert_store_item_to_file_data(item)
            else:
                try:
                    file_data = _get_file_data_from_state(state, file_path)
                except ValueError as e:
                    return str(e)
            return _read_file_data_content(file_data, offset, limit)

    else:

        @tool(description=tool_description)
        def read_file(
            file_path: str,
            state: Annotated[FilesystemState, InjectedState],
            offset: int = DEFAULT_READ_OFFSET,
            limit: int = DEFAULT_READ_LIMIT,
        ) -> str:
            file_path = validate_path(file_path)
            try:
                file_data = _get_file_data_from_state(state, file_path)
            except ValueError as e:
                return str(e)
            return _read_file_data_content(file_data, offset, limit)

    return read_file


def _write_file_tool_generator(
    custom_description: str | None = None, *, long_term_memory: bool
) -> BaseTool:
    """Generate the write_file tool.

    Args:
        custom_description: Optional custom description for the tool.
        long_term_memory: Whether to enable longterm memory support.

    Returns:
        Configured write_file tool that creates new files in state or longterm store.
    """
    tool_description = WRITE_FILE_TOOL_DESCRIPTION
    if custom_description:
        tool_description = custom_description
    elif long_term_memory:
        tool_description += WRITE_FILE_TOOL_DESCRIPTION_LONGTERM_SUPPLEMENT

    def _write_file_to_state(
        state: FilesystemState, tool_call_id: str, file_path: str, content: str
    ) -> Command | str:
        """Write a new file to the filesystem state.

        Args:
            state: The current filesystem state.
            tool_call_id: ID of the tool call for generating ToolMessage.
            file_path: The path where the file should be written.
            content: The content to write to the file.

        Returns:
            Command to update state with new file, or error string if file exists.
        """
        mock_filesystem = state.get("files", {})
        existing = mock_filesystem.get(file_path)
        if existing:
            return f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path."
        new_file_data = create_file_data(content)
        return Command(
            update={
                "files": {file_path: new_file_data},
                "messages": [ToolMessage(f"Updated file {file_path}", tool_call_id=tool_call_id)],
            }
        )

    if long_term_memory:

        @tool(description=tool_description)
        def write_file(
            file_path: str,
            content: str,
            state: Annotated[FilesystemState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command | str:
            file_path = validate_path(file_path)
            if has_memories_prefix(file_path):
                stripped_file_path = strip_memories_prefix(file_path)
                runtime = get_runtime()
                store = _get_store(runtime)
                namespace = _get_namespace()
                if store.get(namespace, stripped_file_path) is not None:
                    return f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path."
                new_file_data = create_file_data(content)
                store.put(
                    namespace, stripped_file_path, _convert_file_data_to_store_item(new_file_data)
                )
                return f"Updated longterm memories file {file_path}"
            return _write_file_to_state(state, tool_call_id, file_path, content)

    else:

        @tool(description=tool_description)
        def write_file(
            file_path: str,
            content: str,
            state: Annotated[FilesystemState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command | str:
            file_path = validate_path(file_path)
            return _write_file_to_state(state, tool_call_id, file_path, content)

    return write_file


def _edit_file_tool_generator(
    custom_description: str | None = None, *, long_term_memory: bool
) -> BaseTool:
    """Generate the edit_file tool.

    Args:
        custom_description: Optional custom description for the tool.
        long_term_memory: Whether to enable longterm memory support.

    Returns:
        Configured edit_file tool that performs string replacements in files.
    """
    tool_description = EDIT_FILE_TOOL_DESCRIPTION
    if custom_description:
        tool_description = custom_description
    elif long_term_memory:
        tool_description += EDIT_FILE_TOOL_DESCRIPTION_LONGTERM_SUPPLEMENT

    def _perform_file_edit(
        file_data: FileData,
        old_string: str,
        new_string: str,
        *,
        replace_all: bool = False,
    ) -> tuple[FileData, str] | str:
        """Perform string replacement on file data.

        Args:
            file_data: The file data to edit.
            old_string: String to find and replace.
            new_string: Replacement string.
            replace_all: If True, replace all occurrences.

        Returns:
            Tuple of (updated_file_data, success_message) on success,
            or error string on failure.
        """
        content = file_data_to_string(file_data)
        occurrences = content.count(old_string)
        if occurrences == 0:
            return f"Error: String not found in file: '{old_string}'"
        if occurrences > 1 and not replace_all:
            return f"Error: String '{old_string}' appears {occurrences} times in file. Use replace_all=True to replace all instances, or provide a more specific string with surrounding context."
        new_content = content.replace(old_string, new_string)
        new_file_data = update_file_data(file_data, new_content)
        result_msg = f"Successfully replaced {occurrences} instance(s) of the string"
        return new_file_data, result_msg

    if long_term_memory:

        @tool(description=tool_description)
        def edit_file(
            file_path: str,
            old_string: str,
            new_string: str,
            state: Annotated[FilesystemState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
            *,
            replace_all: bool = False,
        ) -> Command | str:
            file_path = validate_path(file_path)
            is_longterm_memory = has_memories_prefix(file_path)

            # Retrieve file data from appropriate storage
            if is_longterm_memory:
                stripped_file_path = strip_memories_prefix(file_path)
                runtime = get_runtime()
                store = _get_store(runtime)
                namespace = _get_namespace()
                item: Item | None = store.get(namespace, stripped_file_path)
                if item is None:
                    return f"Error: File '{file_path}' not found"
                file_data = _convert_store_item_to_file_data(item)
            else:
                try:
                    file_data = _get_file_data_from_state(state, file_path)
                except ValueError as e:
                    return str(e)

            # Perform the edit
            result = _perform_file_edit(file_data, old_string, new_string, replace_all=replace_all)
            if isinstance(result, str):  # Error message
                return result

            new_file_data, result_msg = result
            full_msg = f"{result_msg} in '{file_path}'"

            # Save to appropriate storage
            if is_longterm_memory:
                store.put(
                    namespace, stripped_file_path, _convert_file_data_to_store_item(new_file_data)
                )
                return full_msg

            return Command(
                update={
                    "files": {file_path: new_file_data},
                    "messages": [ToolMessage(full_msg, tool_call_id=tool_call_id)],
                }
            )
    else:

        @tool(description=tool_description)
        def edit_file(
            file_path: str,
            old_string: str,
            new_string: str,
            state: Annotated[FilesystemState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
            *,
            replace_all: bool = False,
        ) -> Command | str:
            file_path = validate_path(file_path)

            # Retrieve file data from state
            try:
                file_data = _get_file_data_from_state(state, file_path)
            except ValueError as e:
                return str(e)

            # Perform the edit
            result = _perform_file_edit(file_data, old_string, new_string, replace_all=replace_all)
            if isinstance(result, str):  # Error message
                return result

            new_file_data, result_msg = result
            full_msg = f"{result_msg} in '{file_path}'"

            return Command(
                update={
                    "files": {file_path: new_file_data},
                    "messages": [ToolMessage(full_msg, tool_call_id=tool_call_id)],
                }
            )

    return edit_file


TOOL_GENERATORS = {
    "ls": _ls_tool_generator,
    "read_file": _read_file_tool_generator,
    "write_file": _write_file_tool_generator,
    "edit_file": _edit_file_tool_generator,
}


def _get_filesystem_tools(
    custom_tool_descriptions: dict[str, str] | None = None, *, long_term_memory: bool
) -> list[BaseTool]:
    """Get filesystem tools.

    Args:
        custom_tool_descriptions: Optional custom descriptions for tools.
        long_term_memory: Whether to enable longterm memory support.

    Returns:
        List of configured filesystem tools (ls, read_file, write_file, edit_file).
    """
    if custom_tool_descriptions is None:
        custom_tool_descriptions = {}
    tools = []
    for tool_name, tool_generator in TOOL_GENERATORS.items():
        tool = tool_generator(
            custom_tool_descriptions.get(tool_name), long_term_memory=long_term_memory
        )
        tools.append(tool)
    return tools


class FilesystemMiddleware(AgentMiddleware):
    """Middleware for providing filesystem tools to an agent.

    This middleware adds four filesystem tools to the agent: ls, read_file, write_file,
    and edit_file. Files can be stored in two locations:
    - Short-term: In the agent's state (ephemeral, lasts only for the conversation)
    - Long-term: In a persistent store (persists across conversations when enabled)

    Args:
        long_term_memory: Whether to enable longterm memory support.
        system_prompt_extension: Optional custom system prompt override.
        custom_tool_descriptions: Optional custom tool descriptions override.

    Raises:
        ValueError: If longterm memory is enabled but no store is available.

    Example:
        ```python
        from langchain.agents.middleware.filesystem import FilesystemMiddleware
        from langchain.agents import create_agent

        # Short-term memory only
        agent = create_agent(middleware=[FilesystemMiddleware(long_term_memory=False)])

        # With long-term memory
        agent = create_agent(middleware=[FilesystemMiddleware(long_term_memory=True)])
        ```
    """

    state_schema = FilesystemState

    def __init__(
        self,
        *,
        long_term_memory: bool = False,
        system_prompt_extension: str | None = None,
        custom_tool_descriptions: dict[str, str] | None = None,
    ) -> None:
        """Initialize the filesystem middleware.

        Args:
            long_term_memory: Whether to enable longterm memory support.
            system_prompt_extension: Optional custom system prompt override.
            custom_tool_descriptions: Optional custom tool descriptions override.
        """
        self.long_term_memory = long_term_memory
        self.system_prompt_extension = FILESYSTEM_SYSTEM_PROMPT
        if system_prompt_extension is not None:
            self.system_prompt_extension = system_prompt_extension
        elif long_term_memory:
            self.system_prompt_extension += FILESYSTEM_SYSTEM_PROMPT_LONGTERM_SUPPLEMENT

        self.tools = _get_filesystem_tools(
            custom_tool_descriptions, long_term_memory=long_term_memory
        )

    def before_model_call(self, request: ModelRequest, runtime: Runtime[Any]) -> ModelRequest:
        """Validate that store is available if longterm memory is enabled.

        Args:
            request: The model request being processed.
            runtime: The LangGraph runtime.

        Returns:
            The unmodified model request.

        Raises:
            ValueError: If long_term_memory is True but runtime.store is None.
        """
        if self.long_term_memory and runtime.store is None:
            msg = "Longterm memory is enabled, but no store is available"
            raise ValueError(msg)
        return request

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Update the system prompt to include instructions on using the filesystem.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        if self.system_prompt_extension is not None:
            request.system_prompt = (
                request.system_prompt + "\n\n" + self.system_prompt_extension
                if request.system_prompt
                else self.system_prompt_extension
            )
        return handler(request)

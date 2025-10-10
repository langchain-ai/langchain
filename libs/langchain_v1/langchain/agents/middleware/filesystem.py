"""Middleware for providing filesystem tools to an agent."""
# ruff: noqa: E501

from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any, NotRequired

if TYPE_CHECKING:
    from langgraph.runtime import Runtime
    from langgraph.store.base import Item

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.runtime import Runtime, get_runtime
from langgraph.types import Command

from langchain.agents.middleware.file_utils import (
    FileData,
    apply_string_replacement,
    check_empty_content,
    create_file_data,
    file_data_reducer,
    file_data_to_string,
    format_content_with_line_numbers,
    list_directory,
    update_file_data,
)
from langchain.agents.middleware.types import AgentMiddleware, AgentState, ModelRequest
from langchain.tools.tool_node import InjectedState


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
LIST_FILES_TOOL_DESCRIPTION_LONGTERM_SUPPLEMENT = (
    "\n- Files from the longterm filesystem will be prefixed with the memories/ path."
)

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
READ_FILE_TOOL_DESCRIPTION_LONGTERM_SUPPLEMENT = (
    "\n- file_paths prefixed with the memories/ path will be read from the longterm filesystem."
)

EDIT_FILE_TOOL_DESCRIPTION = """Performs exact string replacements in files.

Usage:
- You must use your `Read` tool at least once in the conversation before editing. This tool will error if you attempt an edit without reading the file.
- When editing text from Read tool output, ensure you preserve the exact indentation (tabs/spaces) as it appears AFTER the line number prefix. The line number prefix format is: spaces + line number + tab. Everything after that tab is the actual file content to match. Never include any part of the line number prefix in the old_string or new_string.
- ALWAYS prefer editing existing files. NEVER write new files unless explicitly required.
- Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.
- The edit will FAIL if `old_string` is not unique in the file. Either provide a larger string with more surrounding context to make it unique or use `replace_all` to change every instance of `old_string`.
- Use `replace_all` for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance."""
EDIT_FILE_TOOL_DESCRIPTION_LONGTERM_SUPPLEMENT = "\n- You can edit files in the longterm filesystem by prefixing the filename with the memories/ path."

WRITE_FILE_TOOL_DESCRIPTION = """Writes to a new file in the filesystem.

Usage:
- The file_path parameter must be an absolute path, not a relative path
- The content parameter must be a string
- The write_file tool will create the a new file.
- Prefer to edit existing files over creating new ones when possible.
- file_paths prefixed with the memories/ path will be written to the longterm filesystem."""
WRITE_FILE_TOOL_DESCRIPTION_LONGTERM_SUPPLEMENT = (
    "\n- file_paths prefixed with the memories/ path will be written to the longterm filesystem."
)

FILESYSTEM_SYSTEM_PROMPT = """## Filesystem Tools `ls`, `read_file`, `write_file`, `edit_file`

You have access to a filesystem which you can interact with using these tools.
Do not prepend a / to file_paths.

- ls: list all files in the filesystem
- read_file: read a file from the filesystem
- write_file: write to a file in the filesystem
- edit_file: edit a file in the filesystem"""
FILESYSTEM_SYSTEM_PROMPT_LONGTERM_SUPPLEMENT = """

You also have access to a longterm filesystem in which you can store files that you want to keep around for longer than the current conversation.
In order to interact with the longterm filesystem, you can use those same tools, but filenames must be prefixed with the memories/ path.
Remember, to interact with the longterm filesystem, you must prefix the filename with the memories/ path."""


def _has_memories_prefix(file_path: str) -> bool:
    return file_path.startswith("memories/")


def _append_memories_prefix(file_path: str) -> str:
    return f"memories/{file_path}"


def _strip_memories_prefix(file_path: str) -> str:
    return file_path.replace("memories/", "")


def _get_namespace(runtime: Runtime[Any]) -> tuple[str] | tuple[str, str]:
    namespace = "filesystem"
    if runtime.context is None:
        return (namespace,)
    assistant_id = runtime.context.get("assistant_id")
    if assistant_id is None:
        return (namespace,)
    return (assistant_id, "filesystem")


def _ls_tool_generator(
    custom_description: str | None = None, *, has_longterm_memory: bool
) -> BaseTool:
    tool_description = LIST_FILES_TOOL_DESCRIPTION
    if custom_description:
        tool_description = custom_description
    elif has_longterm_memory:
        tool_description += LIST_FILES_TOOL_DESCRIPTION_LONGTERM_SUPPLEMENT

    if has_longterm_memory:
        # Tool with Long-term memory
        @tool(description=tool_description)
        def ls(
            state: Annotated[FilesystemState, InjectedState], path: str | None = None
        ) -> list[str]:
            files_dict = state.get("files", {})

            # If path is provided, filter by directory
            if path is not None:
                from langchain.agents.middleware.file_utils import validate_path

                normalized_path = validate_path(path)
                files = list_directory(files_dict, normalized_path)
            else:
                files = list(files_dict.keys())

            runtime = get_runtime()
            store = runtime.store
            if store is None:
                msg = "Longterm memory is enabled, but no store is available"
                raise ValueError(msg)
            namespace = _get_namespace(runtime)
            file_data_list = store.search(namespace)
            memories_files = [_append_memories_prefix(f.key) for f in file_data_list]
            files.extend(memories_files)
            return files
    else:
        # Tool without long-term memory
        @tool(description=tool_description)
        def ls(
            state: Annotated[FilesystemState, InjectedState], path: str | None = None
        ) -> list[str]:
            files_dict = state.get("files", {})

            # If path is provided, filter by directory
            if path is not None:
                from langchain.agents.middleware.file_utils import validate_path

                normalized_path = validate_path(path)
                return list_directory(files_dict, normalized_path)

            return list(files_dict.keys())

    return ls


def _read_file_tool_generator(
    custom_description: str | None = None, *, has_longterm_memory: bool
) -> BaseTool:
    tool_description = READ_FILE_TOOL_DESCRIPTION
    if custom_description:
        tool_description = custom_description
    elif has_longterm_memory:
        tool_description += READ_FILE_TOOL_DESCRIPTION_LONGTERM_SUPPLEMENT

    if has_longterm_memory:
        # Tool with Long-term memory
        @tool(description=tool_description)
        def read_file(
            file_path: str,
            state: Annotated[FilesystemState, InjectedState],
            offset: int = 0,
            limit: int = 2000,
        ) -> str:
            if _has_memories_prefix(file_path):
                stripped_file_path = _strip_memories_prefix(file_path)
                runtime = get_runtime()
                store = runtime.store
                if store is None:
                    msg = "Longterm memory is enabled, but no store is available"
                    raise ValueError(msg)
                namespace = _get_namespace(runtime)
                item: Item | None = store.get(namespace, stripped_file_path)
                if item is None:
                    return f"Error: File '{file_path}' not found"
                content: str = str(item.value["content"])
            else:
                mock_filesystem = state.get("files", {})
                if file_path not in mock_filesystem:
                    return f"Error: File '{file_path}' not found"
                file_data = mock_filesystem[file_path]
                content = file_data_to_string(file_data)

            # Check for empty content
            empty_msg = check_empty_content(content)
            if empty_msg:
                return empty_msg

            lines = content.splitlines()
            start_idx = offset
            end_idx = min(start_idx + limit, len(lines))
            if start_idx >= len(lines):
                return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"

            # Use shared formatting for the selected range
            selected_lines = lines[start_idx:end_idx]
            return format_content_with_line_numbers(
                selected_lines, format_style="tab", start_line=start_idx + 1
            )
    else:
        # Tool without long-term memory
        @tool(description=tool_description)
        def read_file(
            file_path: str,
            state: Annotated[FilesystemState, InjectedState],
            offset: int = 0,
            limit: int = 2000,
        ) -> str:
            mock_filesystem = state.get("files", {})
            if file_path not in mock_filesystem:
                return f"Error: File '{file_path}' not found"
            file_data = mock_filesystem[file_path]
            content = file_data_to_string(file_data)

            # Check for empty content
            empty_msg = check_empty_content(content)
            if empty_msg:
                return empty_msg

            lines = content.splitlines()
            start_idx = offset
            end_idx = min(start_idx + limit, len(lines))
            if start_idx >= len(lines):
                return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"

            # Use shared formatting for the selected range
            selected_lines = lines[start_idx:end_idx]
            return format_content_with_line_numbers(
                selected_lines, format_style="tab", start_line=start_idx + 1
            )

    return read_file


def _write_file_tool_generator(
    custom_description: str | None = None, *, has_longterm_memory: bool
) -> BaseTool:
    tool_description = WRITE_FILE_TOOL_DESCRIPTION
    if custom_description:
        tool_description = custom_description
    elif has_longterm_memory:
        tool_description += WRITE_FILE_TOOL_DESCRIPTION_LONGTERM_SUPPLEMENT

    if has_longterm_memory:
        # Tool with Long-term memory
        @tool(description=tool_description)
        def write_file(
            file_path: str,
            content: str,
            state: Annotated[FilesystemState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command:
            if _has_memories_prefix(file_path):
                stripped_file_path = _strip_memories_prefix(file_path)
                runtime = get_runtime()
                store = runtime.store
                if store is None:
                    msg = "Longterm memory is enabled, but no store is available"
                    raise ValueError(msg)
                namespace = _get_namespace(runtime)
                store.put(namespace, stripped_file_path, {"content": content})
                return Command(
                    update={
                        "messages": [
                            ToolMessage(
                                f"Updated longterm memories file {file_path}",
                                tool_call_id=tool_call_id,
                            )
                        ]
                    }
                )
            mock_filesystem = state.get("files", {})
            existing = mock_filesystem.get(file_path)

            # Create or update FileData
            if existing:
                new_file_data = update_file_data(existing, content)
            else:
                new_file_data = create_file_data(content)

            return Command(
                update={
                    "files": {file_path: new_file_data},
                    "messages": [
                        ToolMessage(f"Updated file {file_path}", tool_call_id=tool_call_id)
                    ],
                }
            )
    else:
        # Tool without long-term memory
        @tool(description=tool_description)
        def write_file(
            file_path: str,
            content: str,
            state: Annotated[FilesystemState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command:
            mock_filesystem = state.get("files", {})
            existing = mock_filesystem.get(file_path)

            # Create or update FileData
            if existing:
                new_file_data = update_file_data(existing, content)
            else:
                new_file_data = create_file_data(content)

            return Command(
                update={
                    "files": {file_path: new_file_data},
                    "messages": [
                        ToolMessage(f"Updated file {file_path}", tool_call_id=tool_call_id)
                    ],
                }
            )

    return write_file


def _edit_file_tool_generator(
    custom_description: str | None = None, *, has_longterm_memory: bool
) -> BaseTool:
    tool_description = EDIT_FILE_TOOL_DESCRIPTION
    if custom_description:
        tool_description = custom_description
    elif has_longterm_memory:
        tool_description += EDIT_FILE_TOOL_DESCRIPTION_LONGTERM_SUPPLEMENT

    if has_longterm_memory:
        # Tool with Long-term memory
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
            if _has_memories_prefix(file_path):
                stripped_file_path = _strip_memories_prefix(file_path)
                runtime = get_runtime()
                store = runtime.store
                if store is None:
                    msg = "Longterm memory is enabled, but no store is available"
                    raise ValueError(msg)
                namespace = _get_namespace(runtime)
                item: Item | None = store.get(namespace, stripped_file_path)
                if item is None:
                    return f"Error: File '{file_path}' not found"
                content: str = str(item.value["content"])
                if old_string not in content:
                    return f"Error: String not found in file: '{old_string}'"
                if not replace_all:
                    occurrences = content.count(old_string)
                    if occurrences > 1:
                        return f"Error: String '{old_string}' appears {occurrences} times in file. Use replace_all=True to replace all instances, or provide a more specific string with surrounding context."
                    if occurrences == 0:
                        return f"Error: String not found in file: '{old_string}'"
                new_content = content.replace(old_string, new_string, 1)
                replacement_count = 1
                store.put(namespace, stripped_file_path, {"content": new_content})
                return f"Successfully replaced {replacement_count} instance(s) of the string in '{file_path}'"
            mock_filesystem = state.get("files", {})
            if file_path not in mock_filesystem:
                return f"Error: File '{file_path}' not found"
            file_data = mock_filesystem[file_path]
            content = file_data_to_string(file_data)

            # Check if string exists
            if old_string not in content:
                return f"Error: String not found in file: '{old_string}'"

            # Apply replacement
            new_content, replacement_count = apply_string_replacement(
                content, old_string, new_string, replace_all=replace_all
            )

            # Update file data
            new_file_data = update_file_data(file_data, new_content)

            result_msg = f"Successfully replaced {replacement_count} instance(s) of the string in '{file_path}'"
            return Command(
                update={
                    "files": {file_path: new_file_data},
                    "messages": [ToolMessage(result_msg, tool_call_id=tool_call_id)],
                }
            )
    else:
        # Tool without long-term memory
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
            mock_filesystem = state.get("files", {})
            if file_path not in mock_filesystem:
                return f"Error: File '{file_path}' not found"
            file_data = mock_filesystem[file_path]
            content = file_data_to_string(file_data)

            # Check if string exists
            if old_string not in content:
                return f"Error: String not found in file: '{old_string}'"

            # Apply replacement
            new_content, replacement_count = apply_string_replacement(
                content, old_string, new_string, replace_all=replace_all
            )

            # Update file data
            new_file_data = update_file_data(file_data, new_content)

            result_msg = f"Successfully replaced {replacement_count} instance(s) of the string in '{file_path}'"
            return Command(
                update={
                    "files": {file_path: new_file_data},
                    "messages": [ToolMessage(result_msg, tool_call_id=tool_call_id)],
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
    custom_tool_descriptions: dict[str, str] | None = None, *, has_longterm_memory: bool
) -> list[BaseTool]:
    """Get filesystem tools.

    Args:
        has_longterm_memory: Whether to enable longterm memory support.
        custom_tool_descriptions: Optional custom descriptions for tools.

    Returns:
        List of configured filesystem tools.
    """
    if custom_tool_descriptions is None:
        custom_tool_descriptions = {}
    tools = []
    for tool_name, tool_generator in TOOL_GENERATORS.items():
        tool = tool_generator(
            custom_tool_descriptions.get(tool_name), has_longterm_memory=has_longterm_memory
        )
        tools.append(tool)
    return tools


class FilesystemMiddleware(AgentMiddleware):
    """Middleware for providing filesystem tools to an agent.

    Args:
        use_longterm_memory: Whether to enable longterm memory support.
        system_prompt_extension: Optional custom system prompt.
        custom_tool_descriptions: Optional custom tool descriptions.

    Returns:
        List of configured filesystem tools.

    Raises:
        ValueError: If longterm memory is enabled but no store is available.

    Example:
        ```python
        from langchain.agents.middleware.filesystem import FilesystemMiddleware
        from langchain.agents import create_agent

        agent = create_agent(middleware=[FilesystemMiddleware(use_longterm_memory=False)])
        ```
    """

    state_schema = FilesystemState

    def __init__(
        self,
        *,
        use_longterm_memory: bool = False,
        system_prompt_extension: str | None = None,
        custom_tool_descriptions: dict[str, str] | None = None,
    ) -> None:
        """Initialize the filesystem middleware.

        Args:
            use_longterm_memory: Whether to enable longterm memory support.
            system_prompt_extension: Optional custom system prompt.
            custom_tool_descriptions: Optional custom tool descriptions.
        """
        self.system_prompt_extension = FILESYSTEM_SYSTEM_PROMPT
        if system_prompt_extension is not None:
            self.system_prompt_extension = system_prompt_extension
        elif use_longterm_memory:
            self.system_prompt_extension += FILESYSTEM_SYSTEM_PROMPT_LONGTERM_SUPPLEMENT

        self.tools = _get_filesystem_tools(
            custom_tool_descriptions, has_longterm_memory=use_longterm_memory
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], AIMessage],
    ) -> AIMessage:
        """Update the system prompt to include instructions on using the filesystem."""
        if self.system_prompt_extension is not None:
            request.system_prompt = (
                request.system_prompt + "\n\n" + self.system_prompt_extension
                if request.system_prompt
                else self.system_prompt_extension
            )
        return handler(request)

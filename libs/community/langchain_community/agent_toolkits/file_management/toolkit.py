from __future__ import annotations

from typing import List, Optional

from langchain_core.pydantic_v1 import root_validator

from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.tools import BaseTool
from langchain_community.tools.file_management.copy import CopyFileTool
from langchain_community.tools.file_management.delete import DeleteFileTool
from langchain_community.tools.file_management.file_search import FileSearchTool
from langchain_community.tools.file_management.list_dir import ListDirectoryTool
from langchain_community.tools.file_management.move import MoveFileTool
from langchain_community.tools.file_management.read import ReadFileTool
from langchain_community.tools.file_management.write import WriteFileTool

_FILE_TOOLS = {
    # "Type[Runnable[Any, Any]]" has no attribute "__fields__"  [attr-defined]
    tool_cls.__fields__["name"].default: tool_cls  # type: ignore[attr-defined]
    for tool_cls in [
        CopyFileTool,
        DeleteFileTool,
        FileSearchTool,
        MoveFileTool,
        ReadFileTool,
        WriteFileTool,
        ListDirectoryTool,
    ]
}


class FileManagementToolkit(BaseToolkit):
    """Toolkit for interacting with local files.

    *Security Notice*: This toolkit provides methods to interact with local files.
        If providing this toolkit to an agent on an LLM, ensure you scope
        the agent's permissions to only include the necessary permissions
        to perform the desired operations.

        By **default** the agent will have access to all files within
        the root dir and will be able to Copy, Delete, Move, Read, Write
        and List files in that directory.

        Consider the following:
        - Limit access to particular directories using `root_dir`.
        - Use filesystem permissions to restrict access and permissions to only
          the files and directories required by the agent.
        - Limit the tools available to the agent to only the file operations
          necessary for the agent's intended use.
        - Sandbox the agent by running it in a container.

        See https://python.langchain.com/docs/security for more information.
    """

    root_dir: Optional[str] = None
    """If specified, all file operations are made relative to root_dir."""
    selected_tools: Optional[List[str]] = None
    """If provided, only provide the selected tools. Defaults to all."""

    @root_validator
    def validate_tools(cls, values: dict) -> dict:
        selected_tools = values.get("selected_tools") or []
        for tool_name in selected_tools:
            if tool_name not in _FILE_TOOLS:
                raise ValueError(
                    f"File Tool of name {tool_name} not supported."
                    f" Permitted tools: {list(_FILE_TOOLS)}"
                )
        return values

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        allowed_tools = self.selected_tools or _FILE_TOOLS.keys()
        tools: List[BaseTool] = []
        for tool in allowed_tools:
            tool_cls = _FILE_TOOLS[tool]
            tools.append(tool_cls(root_dir=self.root_dir))  # type: ignore
        return tools


__all__ = ["FileManagementToolkit"]

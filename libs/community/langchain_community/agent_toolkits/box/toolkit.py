from __future__ import annotations

from typing import Dict, List, Optional, Type

from langchain_core.pydantic_v1 import root_validator
from langchain_core.tools import BaseToolkit

from langchain_community.tools import BaseTool
from langchain_community.tools.box.box_file_search import BoxFileSearchTool
from langchain_community.tools.box.box_ai_ask import BoxAIAskTool
from langchain_community.tools.box.box_text_rep import BoxTextRepTool
from langchain_community.tools.box.box_folder_contents import BoxFolderContentsTool


_BOX_TOOLS: List[Type[BaseTool]] = [
    BoxFileSearchTool,
    BoxAIAskTool,
    BoxTextRepTool,
    BoxFolderContentsTool,
]
_BOX_TOOLS_MAP: Dict[str, Type[BaseTool]] = {
    tool_cls.__fields__["name"].default: tool_cls for tool_cls in _BOX_TOOLS
}


class BoxToolkit(BaseToolkit):
    """Toolkit for interacting with Box files.
    """

    selected_tools: Optional[List[str]] = None
    """If provided, only provide the selected tools. Defaults to all."""

    @root_validator
    def validate_tools(cls, values: dict) -> dict:
        selected_tools = values.get("selected_tools") or []
        for tool_name in selected_tools:
            if tool_name not in _BOX_TOOLS_MAP:
                raise ValueError(
                    f"File Tool of name {tool_name} not supported."
                    f" Permitted tools: {list(_BOX_TOOLS_MAP)}"
                )
        return values

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        allowed_tools = self.selected_tools or _BOX_TOOLS_MAP
        tools: List[BaseTool] = []
        for tool in allowed_tools:
            tool_cls = _BOX_TOOLS_MAP[tool]
            tools.append(tool_cls(root_dir=self.root_dir))  # type: ignore[call-arg]
        return tools


__all__ = ["BoxToolkit"]

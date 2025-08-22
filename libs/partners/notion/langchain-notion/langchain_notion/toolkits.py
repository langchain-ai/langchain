"""LangchainNotion toolkit for grouping Notion tools for agent use."""

from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseToolkit

from langchain_notion.notion_wrapper import NotionWrapper
from langchain_notion.tools import (
    CreatePage,
    GetPage,
    LangchainNotionTool,
    SearchQuery,
    UpdatePage,
)


class LangchainNotionToolkit(BaseToolkit):
    """
    Toolkit that groups Notion tools (search, get, create, update) for use with LangChain agents.

    Requires NOTION_API_KEY to be set in the environment or passed to NotionWrapper.
    """

    def __init__(self, tools: List[BaseTool]):
        self._tools = tools

    # NOTE: Do not assign a mutable default like [] at the class level.
    # BaseToolkit handles storing tools on the instance created below.

    @classmethod
    def from_notion_wrapper(
        cls,
        api: NotionWrapper,
        *,
        include_write_tools: bool = True,
    ) -> "LangchainNotionToolkit":
        """
        Instantiate the toolkit with Notion tools using a NotionWrapper instance.

        Args:
            api: Authenticated NotionWrapper instance.
            include_write_tools: If False, only read tools are included (no create/update).

        Returns:
            LangchainNotionToolkit with the selected tools registered.
        """
        if not isinstance(api, NotionWrapper):
            raise TypeError("api must be an instance of NotionWrapper")

        ops: List[Dict[str, Any]] = [
            {
                "mode": "search_pages",
                "name": "Search Pages",
                "description": "Search Notion pages by keywords.",
                "args_schema": SearchQuery,
            },
            {
                "mode": "get_page",
                "name": "Get Page",
                "description": "Retrieve a page's title, URL, and properties.",
                "args_schema": GetPage,
            },
        ]

        if include_write_tools:
            ops.extend(
                [
                    {
                        "mode": "create_page",
                        "name": "Create Page",
                        "description": "Create a new page in Notion.",
                        "args_schema": CreatePage,
                    },
                    {
                        "mode": "update_page",
                        "name": "Update Page",
                        "description": "Update an existing Notion page's properties.",
                        "args_schema": UpdatePage,
                    },
                ]
            )

        tools: List[BaseTool] = [
            LangchainNotionTool(
                mode=op["mode"],
                name=op["name"],
                description=op["description"],
                args_schema=op["args_schema"],
                api=api,
            )
            for op in ops
        ]

        return cls(tools=tools)  # type: ignore[arg-type]

    def get_tools(self) -> List[BaseTool]:
        """Return the list of Notion tools in this toolkit."""
        return self._tools

    # Small convenience helper for tests/examples
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """Return a tool by its name, or None if not found."""
        return next((t for t in self.tools if t.name == name), None)

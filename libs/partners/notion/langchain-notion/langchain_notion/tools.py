"""LangchainNotion tools for interacting with Notion via LangChain."""

from typing import Any, Dict, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_notion.notion_wrapper import NotionWrapper

# ---------- Input Schemas ----------


class SearchQuery(BaseModel):
    """Input schema for searching Notion pages by keyword(s)."""

    # query: str = Field(..., description="Keyword(s) to search for pages.")
    tool_input: str = Field(..., description="Keyword(s) to search for pages.")


class GetPage(BaseModel):
    """Input schema for retrieving a Notion page by ID."""

    # page_id: str = Field(..., description="The Notion page ID.")
    tool_input: str = Field(..., description="The Notion page ID.")


class CreatePage(BaseModel):
    """Input schema for creating a new Notion page."""

    # parent must match Notion API shape: {"database_id": "..."} or {"page_id": "..."}
    tool_input: Dict[str, Any] = Field(
        ...,
        description="Input data for creating a page, must include 'parent' and 'properties'. "
        "Example: {'parent': {'database_id': '...'}, 'properties': {...}}",
    )


class UpdatePage(BaseModel):
    """Input schema for updating an existing Notion page."""

    tool_input: Dict[str, Any] = Field(
        ...,
        description="Input data for updating a page, must include 'page_id' and 'properties'. "
        "Example: {'page_id': '...', 'properties': {...}}",
    )


# ---------- Tool ----------


class LangchainNotionTool(BaseTool):  # type: ignore[override]
    """
    Tool for interacting with Notion via NotionWrapper.
    Supports searching, retrieving, creating, and updating Notion pages.

    Notes:
    - Prefer calling tools with .invoke({...}) where the dict matches the schema.
    - This tool also accepts a plain string for Search (convenience).
    """

    # Optional metadata (kept for agent routing / display)
    mode: str
    name: str = ""
    description: str = ""

    # Wrapper instance (env-based default if not provided)
    api: NotionWrapper = Field(default_factory=NotionWrapper)

    # Pydantic model describing inputs for this specific tool instance
    args_schema: Optional[Type[BaseModel]] = None

    # IMPORTANT: BaseTool expects `tool_input` as the first arg
    def _run(
        self,
        tool_input: Any,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **_: Any,
    ) -> str:
        """
        Dispatch to the appropriate NotionWrapper method based on args_schema.
        Accepts either dict inputs (preferred) or plain strings where noted.
        """
        data = tool_input

        # --- Search ---
        # if self.args_schema is SearchQuery:
        if self.args_schema is SearchQuery:
            print(data)
            if isinstance(data, dict):
                query = data.get("query")
                # query = kwargs["query"]
                if not query:
                    return "Error: 'query' is required."
                return self.api.search_pages(query)
            # convenience for plain-string search
            return self.api.search_pages(str(data))

        # --- Get Page ---
        if self.args_schema is GetPage:
            print("@@@@@@@@@@@@@", data)
            if isinstance(data, dict):
                page_id = data.get("page_id")
            else:
                page_id = data
            if not page_id:
                return "Error: 'page_id' is required."
            return self.api.get_page(page_id)

        # --- Create Page ---
        if self.args_schema is CreatePage:
            print(data)
            if not isinstance(data, dict):
                return "Error: expected {'parent': {...}, 'properties': {...}}."
            parent = data.get("parent")
            properties = data.get("properties")
            if not properties:
                return "Error: 'properties' is required."
            # print("parent:", parent)
            # print("properties:", properties)
            return self.api.create_page(parent, properties)

        # --- Update Page ---
        if self.args_schema is UpdatePage:
            if not isinstance(data, dict):
                return "Error: expected {'page_id': '...','properties': {...}}."
            page_id = data.get("page_id")
            properties = data.get("properties")
            if not page_id:
                return "Error: 'page_id' is required."
            if not properties:
                return "Error: 'properties' is required."
            return self.api.update_page(page_id, properties)

        return "Unsupported arguments for this tool."

    async def _arun(  # optional async mirror
        self,
        tool_input: Any,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **_: Any,
    ) -> str:
        return self._run(tool_input, run_manager)

"""LangChain tools for Plasmate web browsing."""

from __future__ import annotations

from typing import Any, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_plasmate._utilities import fetch_som, som_to_context, som_to_text


class PlasmateFetchInput(BaseModel):
    """Input for the PlasmateFetchTool."""

    url: str = Field(description="The URL to fetch and extract content from.")


class PlasmateFetchTool(BaseTool):  # type: ignore[override]
    """Fetch a web page and return its content as structured SOM text.

    Uses Plasmate to compile HTML into a Semantic Object Model (SOM),
    reducing token usage by 90%+ compared to raw HTML while preserving
    content, structure, and interactive element information.

    Setup:
        Install ``langchain-plasmate`` and the ``plasmate`` binary.

        .. code-block:: bash

            pip install langchain-plasmate
            cargo install plasmate  # or: curl -fsSL https://plasmate.app/install.sh | sh

    Instantiation:
        .. code-block:: python

            from langchain_plasmate import PlasmateFetchTool

            tool = PlasmateFetchTool()

    Invocation with args:
        .. code-block:: python

            tool.invoke({"url": "https://news.ycombinator.com"})

    Invocation with ToolCall:
        .. code-block:: python

            tool.invoke(
                {
                    "args": {"url": "https://example.com"},
                    "id": "1",
                    "name": tool.name,
                    "type": "tool_call",
                }
            )
    """

    name: str = "plasmate_fetch"
    description: str = (
        "Fetch a web page and return its content in a structured, "
        "token-efficient format. Use this to read web pages, extract "
        "text, find links, and identify interactive elements. "
        "Returns content with 10x fewer tokens than raw HTML."
    )
    args_schema: Type[BaseModel] = PlasmateFetchInput
    plasmate_bin: Optional[str] = None
    output_format: str = "context"  # "context", "markdown", or "raw"

    def _run(
        self,
        url: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Fetch a URL and return SOM content."""
        som_data = fetch_som(url, plasmate_bin=self.plasmate_bin)
        if self.output_format == "markdown":
            return som_to_text(som_data)
        elif self.output_format == "raw":
            import json

            return json.dumps(som_data, indent=2)
        else:
            return som_to_context(som_data)


class PlasmateNavigateInput(BaseModel):
    """Input for the PlasmateNavigateTool."""

    url: str = Field(description="The URL to navigate to.")
    extract_links: bool = Field(
        default=False,
        description="Whether to include a detailed list of all links on the page.",
    )


class PlasmateNavigateTool(BaseTool):  # type: ignore[override]
    """Navigate to a web page and return structured content with interactive elements.

    Similar to PlasmateFetchTool but optimized for agent navigation workflows.
    Returns interactive elements prominently so the agent can decide what to
    click, type, or interact with.

    Setup:
        Install ``langchain-plasmate`` and the ``plasmate`` binary.

        .. code-block:: bash

            pip install langchain-plasmate
            cargo install plasmate

    Instantiation:
        .. code-block:: python

            from langchain_plasmate import PlasmateNavigateTool

            tool = PlasmateNavigateTool()

    Invocation:
        .. code-block:: python

            tool.invoke({"url": "https://example.com", "extract_links": True})
    """

    name: str = "plasmate_navigate"
    description: str = (
        "Navigate to a URL and get a structured view of the page including "
        "all interactive elements (buttons, links, forms, inputs). "
        "Use this when you need to understand what actions are available on a page."
    )
    args_schema: Type[BaseModel] = PlasmateNavigateInput
    plasmate_bin: Optional[str] = None

    def _run(
        self,
        url: str,
        extract_links: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Navigate to a URL and return interactive page content."""
        som_data = fetch_som(url, plasmate_bin=self.plasmate_bin)
        return som_to_context(som_data)


def get_plasmate_tools(
    plasmate_bin: Optional[str] = None,
) -> list[BaseTool]:
    """Get all Plasmate tools for use with a LangChain agent.

    Args:
        plasmate_bin: Path to plasmate binary. Auto-detected if not provided.

    Returns:
        List of Plasmate tools.
    """
    return [
        PlasmateFetchTool(plasmate_bin=plasmate_bin),
        PlasmateNavigateTool(plasmate_bin=plasmate_bin),
    ]

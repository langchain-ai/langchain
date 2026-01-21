"""MCP Discovery Tool for LangChain.

This module provides a tool for discovering Model Context Protocol (MCP) servers
using natural language queries against the MCP Discovery API.

Example:
    Basic usage with an agent::

        from langchain_classic.tools.mcp_discovery import MCPDiscoveryTool

        tool = MCPDiscoveryTool()
        result = tool.run({"need": "send emails", "limit": 5})
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import requests
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForToolRun


class MCPDiscoveryInput(BaseModel):
    """Input schema for MCP Discovery Tool.

    Attributes:
        need: Natural language description of the capability required.
        limit: Maximum number of MCP server recommendations to return.
    """

    need: str = Field(
        description=(
            "Natural language description of the capability required. "
            "Example: 'I need to send emails' or 'database with authentication'."
        )
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of MCP server recommendations to return.",
    )


class MCPDiscoveryTool(BaseTool):
    """Tool for discovering MCP servers using natural language queries.

    This tool enables agents to dynamically discover Model Context Protocol (MCP)
    servers based on natural language requirements. It queries the MCP Discovery
    API and returns ranked recommendations with install commands and confidence
    scores.

    Attributes:
        name: The name of this tool.
        description: A description of what the tool does.
        args_schema: The Pydantic model for input validation.
        api_url: The MCP Discovery API endpoint URL.
        timeout: Request timeout in seconds.

    Example:
        Basic instantiation and usage::

            tool = MCPDiscoveryTool()
            result = tool.run({"need": "send emails", "limit": 3})
            print(result)
    """

    name: str = "mcp_discovery"
    description: str = (
        "Discover Model Context Protocol (MCP) servers using natural language. "
        "Returns ranked MCP server recommendations with install commands and "
        "confidence scores."
    )

    args_schema: type[BaseModel] = MCPDiscoveryInput

    api_url: str = "https://mcp-discovery-production.up.railway.app/api/v1/discover"
    timeout: int = 10

    def _run(
        self,
        need: str,
        limit: int = 5,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Execute MCP Discovery synchronously.

        Args:
            need: Natural language description of the required capability.
            limit: Maximum number of recommendations to return.
            _run_manager: Optional callback manager for tool run events.

        Returns:
            A human-readable summary of discovered MCP servers suitable for
            agent reasoning, or an error message if the request fails.

        Raises:
            No exceptions are raised; errors are returned as string messages.
        """
        try:
            response = requests.post(
                self.api_url,
                json={"need": need, "limit": limit},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()

        except requests.RequestException as exc:
            return f"Failed to contact MCP Discovery service: {exc}"

        # Validate response structure
        recommendations: list[dict[str, Any]] = payload.get("recommendations", [])
        if not isinstance(recommendations, list) or not recommendations:
            return f"No MCP servers found for requirement: '{need}'."

        return self._format_recommendations(recommendations, need)

    async def _arun(
        self,
        need: str,
        limit: int = 5,
        _run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Execute MCP Discovery asynchronously.

        Uses aiohttp to avoid blocking event loops in async contexts.

        Args:
            need: Natural language description of the required capability.
            limit: Maximum number of recommendations to return.
            _run_manager: Optional callback manager for tool run events.

        Returns:
            A human-readable summary of discovered MCP servers suitable for
            agent reasoning, or an error message if the request fails.

        Raises:
            No exceptions are raised; errors are returned as string messages.
        """
        import aiohttp

        try:
            async with (
                aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as session,
                session.post(
                    self.api_url,
                    json={"need": need, "limit": limit},
                ) as response,
            ):
                response.raise_for_status()
                payload: dict[str, Any] = await response.json()

        except (aiohttp.ClientError, TimeoutError) as exc:
            return f"Failed to contact MCP Discovery service: {exc}"

        recommendations: list[dict[str, Any]] = payload.get("recommendations", [])
        if not isinstance(recommendations, list) or not recommendations:
            return f"No MCP servers found for requirement: '{need}'."

        return self._format_recommendations(recommendations, need)

    @staticmethod
    def _format_recommendations(
        recommendations: list[dict[str, Any]],
        need: str,
    ) -> str:
        """Format MCP server recommendations for agent consumption.

        Args:
            recommendations: List of server recommendation dictionaries from API.
            need: The original natural language requirement.

        Returns:
            A formatted string containing server details including name,
            category, description, install command, and confidence score.
        """
        blocks: list[str] = []

        for server in recommendations:
            name: str = server.get("name", "Unknown server")
            description: str = server.get("description", "No description provided.")
            install: str = server.get("install_command", "No install command provided.")
            confidence: float | int | None = server.get("confidence")

            confidence_str: str = (
                f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "N/A"
            )

            category: str = server.get("category", "unknown")

            blocks.append(
                "\n".join(
                    [
                        f"Server: {name}",
                        f"Category: {category}",
                        f"Description: {description}",
                        f"Install command: {install}",
                        f"Confidence score: {confidence_str}",
                    ]
                )
            )

        header: str = f"MCP servers matching requirement '{need}':"
        return header + "\n\n" + "\n\n".join(blocks)

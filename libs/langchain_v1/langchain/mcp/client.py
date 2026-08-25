"""Client for connecting to multiple MCP servers and loading LC tools/resources.

This module provides the `MultiServerMCPClient` class for managing connections
to multiple MCP servers and loading tools, prompts, and resources from them.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any

from langchain_core.documents.base import Blob
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from mcp import ClientSession
from mcp.client.session import ProgressFnT
from typing_extensions import Self

from langchain.mcp.prompts import load_mcp_prompt
from langchain.mcp.resources import load_mcp_resources
from langchain.mcp.sessions import (
    Connection,
    McpHttpClientFactory,
    SSEConnection,
    StdioConnection,
    StreamableHttpConnection,
    create_session,
)
from langchain.mcp.tools import load_mcp_tools

ASYNC_CONTEXT_MANAGER_ERROR = (
    "As of langchain-mcp-adapters 0.1.0, MultiServerMCPClient cannot be used as a "
    "context manager (e.g., async with MultiServerMCPClient(...)). "
    "Instead, you can do one of the following:\n"
    "1. client = MultiServerMCPClient(...)\n"
    "   tools = await client.get_tools()\n"
    "2. client = MultiServerMCPClient(...)\n"
    "   async with client.session(server_name) as session:\n"
    "       tools = await load_mcp_tools(session)"
)


class MultiServerMCPClient:
    """Client for connecting to multiple MCP servers.

    Loads LangChain-compatible tools, prompts and resources from MCP servers.
    """

    def __init__(
        self,
        connections: dict[str, Connection] | None = None,
        *,
        progress_callback: ProgressFnT | None = None,
        tool_name_prefix: bool = False,
    ) -> None:
        """Initialize a `MultiServerMCPClient` with MCP servers connections.

        Args:
            connections: A `dict` mapping server names to connection configurations. If
                `None`, no initial connections are established.
            progress_callback: Optional handler for the servers' progress notifications.
            tool_name_prefix: If `True`, tool names are prefixed with the server name
                using an underscore separator (e.g., `"weather_search"` instead of
                `"search"`). This helps avoid conflicts when multiple servers have tools
                with the same name. Defaults to `False`.

        !!! example "Basic usage (starting a new session on each tool call)"

            ```python
            from langchain.mcp.client import MultiServerMCPClient

            client = MultiServerMCPClient(
                {
                    "math": {
                        "command": "python",
                        # Make sure to update to the full absolute path to your
                        # math_server.py file
                        "args": ["/path/to/math_server.py"],
                        "transport": "stdio",
                    },
                    "weather": {
                        # Make sure you start your weather server on port 8000
                        "url": "http://localhost:8000/mcp",
                        "transport": "http",
                    },
                }
            )
            all_tools = await client.get_tools()
            ```

        !!! example "Explicitly starting a session"

            ```python
            from langchain.mcp.client import MultiServerMCPClient
            from langchain.mcp.tools import load_mcp_tools

            client = MultiServerMCPClient({...})
            async with client.session("math") as session:
                tools = await load_mcp_tools(session)
            ```
        """
        self.connections: dict[str, Connection] = connections if connections is not None else {}
        self.progress_callback = progress_callback
        self.tool_name_prefix = tool_name_prefix

    @asynccontextmanager
    async def session(
        self,
        server_name: str,
        *,
        auto_initialize: bool = True,
    ) -> AsyncIterator[ClientSession]:
        """Connect to an MCP server and initialize a session.

        Args:
            server_name: Name to identify this server connection
            auto_initialize: Whether to automatically initialize the session

        Raises:
            ValueError: If the server name is not found in the connections

        Yields:
            An initialized `ClientSession`

        """
        if server_name not in self.connections:
            msg = (
                f"Couldn't find a server with name '{server_name}', "
                f"expected one of '{list(self.connections.keys())}'"
            )
            raise ValueError(msg)

        async with create_session(self.connections[server_name]) as session:
            if auto_initialize:
                await session.initialize()
            yield session

    async def get_tools(self, *, server_name: str | None = None) -> list[BaseTool]:
        """Get a list of all tools from all connected servers.

        Args:
            server_name: Optional name of the server to get tools from.
                If `None`, all tools from all servers will be returned.

        !!! note

            A new session will be created for each tool call

        Returns:
            A list of LangChain [tools](https://docs.langchain.com/oss/python/langchain/tools)

        """
        if server_name is not None:
            if server_name not in self.connections:
                msg = (
                    f"Couldn't find a server with name '{server_name}', "
                    f"expected one of '{list(self.connections.keys())}'"
                )
                raise ValueError(msg)
            return await load_mcp_tools(
                None,
                connection=self.connections[server_name],
                progress_callback=self.progress_callback,
                server_name=server_name,
                tool_name_prefix=self.tool_name_prefix,
            )

        all_tools: list[BaseTool] = []
        load_mcp_tool_tasks = []
        for name, connection in self.connections.items():
            load_mcp_tool_task = asyncio.create_task(
                load_mcp_tools(
                    None,
                    connection=connection,
                    progress_callback=self.progress_callback,
                    server_name=name,
                    tool_name_prefix=self.tool_name_prefix,
                )
            )
            load_mcp_tool_tasks.append(load_mcp_tool_task)
        tools_list = await asyncio.gather(*load_mcp_tool_tasks)
        for tools in tools_list:
            all_tools.extend(tools)
        return all_tools

    async def get_prompt(
        self,
        server_name: str,
        prompt_name: str,
        *,
        arguments: dict[str, Any] | None = None,
    ) -> list[HumanMessage | AIMessage]:
        """Get a prompt from a given MCP server."""
        async with self.session(server_name) as session:
            return await load_mcp_prompt(session, prompt_name, arguments=arguments)

    async def get_resources(
        self,
        server_name: str | None = None,
        *,
        uris: str | list[str] | None = None,
    ) -> list[Blob]:
        """Get resources from MCP server(s).

        Args:
            server_name: Optional name of the server to get resources from.
                If `None`, all resources from all servers will be returned.
            uris: Optional resource URI or list of URIs to load. If not provided,
                all resources will be loaded.

        Returns:
            A list of LangChain [Blob][langchain_core.documents.base.Blob] objects.

        """
        if server_name is not None:
            if server_name not in self.connections:
                msg = (
                    f"Couldn't find a server with name '{server_name}', "
                    f"expected one of '{list(self.connections.keys())}'"
                )
                raise ValueError(msg)
            async with self.session(server_name) as session:
                return await load_mcp_resources(session, uris=uris)

        async def _load_resources_from_server(name: str) -> list[Blob]:
            async with self.session(name) as session:
                return await load_mcp_resources(session, uris=uris)

        all_resources: list[Blob] = []
        load_tasks = [
            asyncio.create_task(_load_resources_from_server(name)) for name in self.connections
        ]
        resources_list = await asyncio.gather(*load_tasks)
        for resources in resources_list:
            all_resources.extend(resources)
        return all_resources

    async def __aenter__(self) -> Self:
        """Async context manager entry point.

        Raises:
            NotImplementedError: Context manager support has been removed.
        """
        raise NotImplementedError(ASYNC_CONTEXT_MANAGER_ERROR)

    def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit point.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.

        Raises:
            NotImplementedError: Context manager support has been removed.
        """
        raise NotImplementedError(ASYNC_CONTEXT_MANAGER_ERROR)


__all__ = [
    "McpHttpClientFactory",
    "MultiServerMCPClient",
    "SSEConnection",
    "StdioConnection",
    "StreamableHttpConnection",
]

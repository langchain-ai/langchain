"""Client for connecting to multiple MCP servers and loading LC tools/resources.

This module provides the `MultiServerMCPClient` class for managing connections
to multiple MCP servers and loading tools, prompts, and resources from them.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from langchain_core.documents.base import Blob
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from mcp import ClientSession
from typing_extensions import Self

from langchain.mcp.callbacks import CallbackContext, Callbacks
from langchain.mcp.interceptors import ToolCallInterceptor
from langchain.mcp.prompts import load_mcp_prompt
from langchain.mcp.resources import load_mcp_resources
from langchain.mcp.sessions import Connection, create_session
from langchain.mcp.tools import load_mcp_tools


class MultiServerMCPClient:
    """Client for connecting to multiple MCP servers.

    Loads LangChain-compatible tools, prompts and resources from MCP servers.
    """

    def __init__(
        self,
        connections: dict[str, Connection] | None = None,
        *,
        callbacks: Callbacks | None = None,
        tool_interceptors: list[ToolCallInterceptor] | None = None,
        tool_name_prefix: bool = False,
    ) -> None:
        """Initialize a `MultiServerMCPClient` with MCP servers connections.

        Args:
            connections: A `dict` mapping server names to connection configurations. If
                `None`, no initial connections are established.
            callbacks: Optional callbacks for handling notifications and events.
            tool_interceptors: Optional list of tool call interceptors for modifying
                requests and responses.
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
        self.callbacks = callbacks or Callbacks()
        self.tool_interceptors = tool_interceptors or []
        self.tool_name_prefix = tool_name_prefix
        self._sessions: dict[str, ClientSession] = {}
        self._exit_stack: AsyncExitStack | None = None

    @asynccontextmanager
    async def session(self, server_name: str) -> AsyncIterator[ClientSession]:
        """Connect to an MCP server and initialize a session.

        Args:
            server_name: Name to identify this server connection

        Raises:
            ValueError: If the server name is not found in the connections

        Yields:
            An initialized `ClientSession`

        """
        self._require_server(server_name)

        if (held := self._sessions.get(server_name)) is not None:
            yield held
            return

        async with create_session(
            self.connections[server_name],
            mcp_callbacks=self.callbacks.to_mcp_format(
                context=CallbackContext(server_name=server_name)
            ),
        ) as session:
            yield session

    def _require_server(self, server_name: str) -> None:
        """Raise if the server was not configured.

        Raises:
            ValueError: If the server name is not found in the connections.
        """
        if server_name not in self.connections:
            msg = (
                f"Couldn't find a server with name '{server_name}', "
                f"expected one of '{list(self.connections.keys())}'"
            )
            raise ValueError(msg)

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
            self._require_server(server_name)
            return await load_mcp_tools(
                self._sessions.get(server_name),
                connection=self.connections[server_name],
                callbacks=self.callbacks,
                server_name=server_name,
                tool_interceptors=self.tool_interceptors,
                tool_name_prefix=self.tool_name_prefix,
            )

        all_tools: list[BaseTool] = []
        load_mcp_tool_tasks = []
        for name, connection in self.connections.items():
            load_mcp_tool_task = asyncio.create_task(
                load_mcp_tools(
                    self._sessions.get(name),
                    connection=connection,
                    callbacks=self.callbacks,
                    server_name=name,
                    tool_interceptors=self.tool_interceptors,
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
        """Connect to every server and keep the connections open.

        Optional: without it a session is opened per operation, which still works but
        repeats the handshake on every call, and respawns the subprocess for stdio
        servers. Entering once and reusing is the reason to do it.
        """
        stack = AsyncExitStack()
        try:
            for name, connection in self.connections.items():
                self._sessions[name] = await stack.enter_async_context(
                    create_session(
                        connection,
                        mcp_callbacks=self.callbacks.to_mcp_format(
                            context=CallbackContext(server_name=name)
                        ),
                    )
                )
        except BaseException:
            self._sessions.clear()
            await stack.aclose()
            raise
        self._exit_stack = stack
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Close every connection opened on entry."""
        stack, self._exit_stack = self._exit_stack, None
        self._sessions.clear()
        if stack is not None:
            await stack.aclose()


__all__ = [
    "Callbacks",
    "Connection",
    "MultiServerMCPClient",
]

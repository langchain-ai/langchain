"""Utility classes and functions for stateful MCP agent creation.

This module provides wrapper classes and factory functions to simplify the creation
of agents with stateful MCP (Model Context Protocol) tools. It handles session
lifecycle management automatically, ensuring that browser sessions and other
stateful connections persist across multiple tool invocations.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph

    from langchain.agents.middleware.types import AgentMiddleware


class StatefulMCPAgentExecutor:
    """Wrapper class that manages MCP session lifecycle for agents.

    This class ensures that MCP tools maintain persistent sessions across
    multiple invocations, solving the common issue of browser sessions
    terminating between tool calls.

    Example:
        ```python
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain.agents.mcp_utils import StatefulMCPAgentExecutor

        client = MultiServerMCPClient(
            {
                "playwright": {
                    "command": "npx",
                    "args": ["@playwright/mcp@latest"],
                    "transport": "stdio",
                }
            }
        )

        async with StatefulMCPAgentExecutor(
            client=client,
            server_name="playwright",
            model="gpt-4",
        ) as executor:
            result = await executor.ainvoke(
                {"messages": [{"role": "user", "content": "Navigate and interact with a webpage"}]}
            )
        ```
    """

    def __init__(
        self,
        client: Any,  # MultiServerMCPClient
        server_name: str,
        model: str | BaseChatModel,
        *,
        system_prompt: str | None = None,
        middleware: Sequence[AgentMiddleware] | None = None,
        checkpointer: Any | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        debug: bool = False,
    ) -> None:
        """Initialize the stateful MCP agent executor.

        Args:
            client: MultiServerMCPClient instance for MCP connections.
            server_name: Name of the MCP server to create a session for.
            model: Language model for the agent (string ID or model instance).
            system_prompt: Optional system prompt for the agent.
            middleware: Optional sequence of middleware to apply.
            checkpointer: Optional checkpointer for agent state persistence.
            interrupt_before: Optional list of node names to interrupt before.
            interrupt_after: Optional list of node names to interrupt after.
            debug: Whether to enable debug mode.
        """
        self.client = client
        self.server_name = server_name
        self.model = model
        self.system_prompt = system_prompt
        self.middleware = middleware or []
        self.checkpointer = checkpointer
        self.interrupt_before = interrupt_before
        self.interrupt_after = interrupt_after
        self.debug = debug

        self._session = None
        self._agent: CompiledStateGraph | None = None
        self._tools: list[BaseTool] | None = None

    async def __aenter__(self) -> StatefulMCPAgentExecutor:
        """Async context manager entry: create session and initialize agent."""
        try:
            # Import here to avoid circular dependencies
            from langchain_mcp_adapters.tools import load_mcp_tools

            from langchain.agents import create_agent

            # Create persistent MCP session
            self._session = await self.client.session(self.server_name).__aenter__()

            # Load tools with the persistent session
            self._tools = await load_mcp_tools(self._session)

            # Initialize model
            if isinstance(self.model, str):
                from langchain.chat_models import init_chat_model

                model = init_chat_model(self.model)
            else:
                model = self.model

            # Bind tools to model
            model_with_tools = model.bind_tools(self._tools)

            # Create agent with stateful tools
            self._agent = create_agent(
                model_with_tools,
                self._tools,
                system_prompt=self.system_prompt,
                middleware=list(self.middleware),
                checkpointer=self.checkpointer,
                interrupt_before=self.interrupt_before,
                interrupt_after=self.interrupt_after,
                debug=self.debug,
            )

            return self

        except Exception:
            # Clean up session if initialization fails
            if self._session:
                await self._session.__aexit__(None, None, None)
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit: cleanup session."""
        if self._session:
            await self._session.__aexit__(exc_type, exc_val, exc_tb)

    async def ainvoke(
        self,
        input: dict[str, Any],
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Asynchronously invoke the agent with the given input.

        Args:
            input: Input dictionary containing messages.
            config: Optional configuration dictionary.
            **kwargs: Additional keyword arguments.

        Returns:
            Agent response dictionary.

        Raises:
            RuntimeError: If agent is not initialized (not in context manager).
        """
        if self._agent is None:
            msg = (
                "Agent not initialized. Use StatefulMCPAgentExecutor as a context manager:\n"
                "async with StatefulMCPAgentExecutor(...) as executor:\n"
                "    result = await executor.ainvoke(...)"
            )
            raise RuntimeError(msg)

        return await self._agent.ainvoke(input, config, **kwargs)

    def invoke(
        self,
        input: dict[str, Any],
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Synchronously invoke the agent with the given input.

        Args:
            input: Input dictionary containing messages.
            config: Optional configuration dictionary.
            **kwargs: Additional keyword arguments.

        Returns:
            Agent response dictionary.

        Raises:
            RuntimeError: If agent is not initialized (not in context manager).
        """
        if self._agent is None:
            msg = (
                "Agent not initialized. Use StatefulMCPAgentExecutor as a context manager:\n"
                "async with StatefulMCPAgentExecutor(...) as executor:\n"
                "    result = await executor.ainvoke(...)"
            )
            raise RuntimeError(msg)

        return self._agent.invoke(input, config, **kwargs)

    async def astream(
        self,
        input: dict[str, Any],
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream agent responses asynchronously.

        Args:
            input: Input dictionary containing messages.
            config: Optional configuration dictionary.
            **kwargs: Additional keyword arguments.

        Yields:
            Agent response chunks.

        Raises:
            RuntimeError: If agent is not initialized (not in context manager).
        """
        if self._agent is None:
            msg = (
                "Agent not initialized. Use StatefulMCPAgentExecutor as a context manager:\n"
                "async with StatefulMCPAgentExecutor(...) as executor:\n"
                "    async for chunk in executor.astream(...):\n"
                "        ..."
            )
            raise RuntimeError(msg)

        async for chunk in self._agent.astream(input, config, **kwargs):
            yield chunk

    def stream(
        self,
        input: dict[str, Any],
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        """Stream agent responses synchronously.

        Args:
            input: Input dictionary containing messages.
            config: Optional configuration dictionary.
            **kwargs: Additional keyword arguments.

        Yields:
            Agent response chunks.

        Raises:
            RuntimeError: If agent is not initialized (not in context manager).
        """
        if self._agent is None:
            msg = (
                "Agent not initialized. Use StatefulMCPAgentExecutor as a context manager:\n"
                "async with StatefulMCPAgentExecutor(...) as executor:\n"
                "    for chunk in executor.stream(...):\n"
                "        ..."
            )
            raise RuntimeError(msg)

        for chunk in self._agent.stream(input, config, **kwargs):
            yield chunk

    @property
    def agent(self) -> CompiledStateGraph | None:
        """Get the underlying agent graph."""
        return self._agent

    @property
    def tools(self) -> list[BaseTool] | None:
        """Get the loaded MCP tools."""
        return self._tools


async def create_stateful_mcp_agent(
    client: Any,  # MultiServerMCPClient
    server_name: str,
    model: str | BaseChatModel,
    *,
    system_prompt: str | None = None,
    middleware: Sequence[AgentMiddleware] | None = None,
    checkpointer: Any | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    auto_cleanup: bool = True,
) -> tuple[CompiledStateGraph, Any]:  # (agent, session)
    """Factory function to create an agent with stateful MCP tools.

    This function creates an agent with MCP tools that maintain persistent
    sessions across multiple invocations. It returns both the agent and
    the session object for manual lifecycle management.

    Args:
        client: MultiServerMCPClient instance for MCP connections.
        server_name: Name of the MCP server to create a session for.
        model: Language model for the agent (string ID or model instance).
        system_prompt: Optional system prompt for the agent.
        middleware: Optional sequence of middleware to apply.
        checkpointer: Optional checkpointer for agent state persistence.
        interrupt_before: Optional list of node names to interrupt before.
        interrupt_after: Optional list of node names to interrupt after.
        debug: Whether to enable debug mode.
        auto_cleanup: If False, caller is responsible for session cleanup.

    Returns:
        Tuple of (agent, session). If auto_cleanup is False, the caller
        must call `await session.__aexit__(None, None, None)` when done.

    Example:
        ```python
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain.agents.mcp_utils import create_stateful_mcp_agent

        client = MultiServerMCPClient(
            {
                "playwright": {
                    "command": "npx",
                    "args": ["@playwright/mcp@latest"],
                    "transport": "stdio",
                }
            }
        )

        # With auto cleanup (recommended)
        agent, session = await create_stateful_mcp_agent(
            client=client,
            server_name="playwright",
            model="gpt-4",
        )
        try:
            result = await agent.ainvoke({"messages": [...]})
        finally:
            await session.__aexit__(None, None, None)

        # Or use the StatefulMCPAgentExecutor context manager instead
        ```
    """
    # Import here to avoid circular dependencies
    from langchain_mcp_adapters.tools import load_mcp_tools

    from langchain.agents import create_agent

    # Create persistent MCP session
    session = await client.session(server_name).__aenter__()

    try:
        # Load tools with the persistent session
        tools = await load_mcp_tools(session)

        # Initialize model
        if isinstance(model, str):
            from langchain.chat_models import init_chat_model

            model_instance = init_chat_model(model)
        else:
            model_instance = model

        # Bind tools to model
        model_with_tools = model_instance.bind_tools(tools)

        # Create agent with stateful tools
        agent = create_agent(
            model_with_tools,
            tools,
            system_prompt=system_prompt,
            middleware=list(middleware) if middleware else [],
            checkpointer=checkpointer,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            debug=debug,
        )

        if auto_cleanup:
            # Wrap agent to auto-cleanup session on deletion
            original_del = getattr(agent, "__del__", None)

            def cleanup_on_del(self):
                # Schedule session cleanup
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(session.__aexit__(None, None, None))
                    else:
                        loop.run_until_complete(session.__aexit__(None, None, None))
                except Exception:
                    pass  # Best effort cleanup

                if original_del:
                    original_del()

            agent.__del__ = cleanup_on_del.__get__(agent, type(agent))

        return agent, session

    except Exception:
        # Clean up session if initialization fails
        await session.__aexit__(None, None, None)
        raise


@asynccontextmanager
async def mcp_agent_session(
    client: Any,  # MultiServerMCPClient
    server_name: str,
    model: str | BaseChatModel,
    *,
    system_prompt: str | None = None,
    middleware: Sequence[AgentMiddleware] | None = None,
    checkpointer: Any | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
) -> AsyncIterator[CompiledStateGraph]:
    """Context manager for creating an agent with stateful MCP tools.

    This is a convenience function that automatically manages the session
    lifecycle, ensuring proper cleanup even if an error occurs.

    Args:
        client: MultiServerMCPClient instance for MCP connections.
        server_name: Name of the MCP server to create a session for.
        model: Language model for the agent (string ID or model instance).
        system_prompt: Optional system prompt for the agent.
        middleware: Optional sequence of middleware to apply.
        checkpointer: Optional checkpointer for agent state persistence.
        interrupt_before: Optional list of node names to interrupt before.
        interrupt_after: Optional list of node names to interrupt after.
        debug: Whether to enable debug mode.

    Yields:
        Configured agent with stateful MCP tools.

    Example:
        ```python
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain.agents.mcp_utils import mcp_agent_session

        client = MultiServerMCPClient(
            {
                "playwright": {
                    "command": "npx",
                    "args": ["@playwright/mcp@latest"],
                    "transport": "stdio",
                }
            }
        )

        async with mcp_agent_session(
            client=client,
            server_name="playwright",
            model="gpt-4",
        ) as agent:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": "Navigate to a webpage"}]}
            )
        ```
    """
    agent, session = await create_stateful_mcp_agent(
        client=client,
        server_name=server_name,
        model=model,
        system_prompt=system_prompt,
        middleware=middleware,
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
        auto_cleanup=False,  # We'll handle cleanup manually
    )

    try:
        yield agent
    finally:
        # Ensure session is properly cleaned up
        await session.__aexit__(None, None, None)


__all__ = [
    "StatefulMCPAgentExecutor",
    "create_stateful_mcp_agent",
    "mcp_agent_session",
]

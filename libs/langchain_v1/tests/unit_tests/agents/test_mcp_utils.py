"""Unit tests for MCP utility classes and functions.

This module tests the stateful MCP agent creation utilities, including
session persistence, cleanup, error handling, and state maintenance.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool

from langchain.agents.mcp_utils import (
    StatefulMCPAgentExecutor,
    create_stateful_mcp_agent,
    mcp_agent_session,
)


class MockMCPSession:
    """Mock MCP session for testing."""

    def __init__(self, name: str = "test_session"):
        self.name = name
        self.is_open = False
        self.call_count = 0
        self.cleanup_called = False

    async def __aenter__(self):
        self.is_open = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.is_open = False
        self.cleanup_called = True
        return False


class MockMCPClient:
    """Mock MultiServerMCPClient for testing."""

    def __init__(self):
        self.sessions = {}
        self.session_create_count = 0

    def session(self, server_name: str):
        """Create a mock session context manager."""
        self.session_create_count += 1
        session = MockMCPSession(f"{server_name}_session_{self.session_create_count}")
        self.sessions[server_name] = session
        return session


@tool
def mock_browser_navigate(url: str) -> str:
    """Mock browser navigation tool."""
    return f"Navigated to {url}"


@tool
def mock_browser_click(selector: str) -> str:
    """Mock browser click tool."""
    return f"Clicked on {selector}"


@tool
def mock_browser_type(selector: str, text: str) -> str:
    """Mock browser type tool."""
    return f"Typed '{text}' into {selector}"


class TestStatefulMCPAgentExecutor:
    """Test cases for StatefulMCPAgentExecutor class."""

    @pytest.mark.asyncio
    async def test_session_persistence_across_tool_calls(self):
        """Test that session persists across multiple tool invocations."""
        mock_client = MockMCPClient()
        mock_tools = [mock_browser_navigate, mock_browser_click, mock_browser_type]

        with patch("langchain.agents.mcp_utils.load_mcp_tools") as mock_load_tools:
            mock_load_tools.return_value = mock_tools

            with patch("langchain.agents.mcp_utils.create_agent") as mock_create_agent:
                # Create a mock agent that tracks tool calls
                mock_agent = MagicMock()
                mock_agent.ainvoke = AsyncMock(
                    return_value={"messages": [AIMessage(content="Task completed")]}
                )
                mock_create_agent.return_value = mock_agent

                async with StatefulMCPAgentExecutor(
                    client=mock_client,
                    server_name="playwright",
                    model="gpt-4",
                    system_prompt="Test prompt",
                ) as executor:
                    # Verify session was created and is open
                    assert "playwright" in mock_client.sessions
                    session = mock_client.sessions["playwright"]
                    assert session.is_open

                    # Make multiple invocations
                    await executor.ainvoke(
                        {"messages": [HumanMessage(content="Navigate to example.com")]}
                    )
                    await executor.ainvoke({"messages": [HumanMessage(content="Click button")]})
                    await executor.ainvoke({"messages": [HumanMessage(content="Type text")]})

                    # Verify same session was used (only one session created)
                    assert mock_client.session_create_count == 1
                    assert session.is_open  # Session still open

                # Verify session was cleaned up after context exit
                assert session.cleanup_called
                assert not session.is_open

    @pytest.mark.asyncio
    async def test_session_cleanup_on_agent_termination(self):
        """Test that session is properly cleaned up when agent terminates."""
        mock_client = MockMCPClient()
        mock_tools = [mock_browser_navigate]

        with patch("langchain.agents.mcp_utils.load_mcp_tools") as mock_load_tools:
            mock_load_tools.return_value = mock_tools

            with patch("langchain.agents.mcp_utils.create_agent") as mock_create_agent:
                mock_agent = MagicMock()
                mock_agent.ainvoke = AsyncMock(
                    return_value={"messages": [AIMessage(content="Done")]}
                )
                mock_create_agent.return_value = mock_agent

                # Create executor and verify cleanup
                async with StatefulMCPAgentExecutor(
                    client=mock_client,
                    server_name="test_server",
                    model="gpt-4",
                ) as executor:
                    session = mock_client.sessions["test_server"]
                    assert session.is_open
                    assert not session.cleanup_called

                # After context exit, session should be cleaned up
                assert session.cleanup_called
                assert not session.is_open

    @pytest.mark.asyncio
    async def test_error_handling_when_session_fails(self):
        """Test proper error handling when session creation fails."""
        mock_client = MockMCPClient()

        # Make session creation fail
        original_session = mock_client.session

        def failing_session(server_name):
            if server_name == "failing_server":
                raise ConnectionError("Failed to connect to MCP server")
            return original_session(server_name)

        mock_client.session = failing_session

        with pytest.raises(ConnectionError, match="Failed to connect to MCP server"):
            async with StatefulMCPAgentExecutor(
                client=mock_client,
                server_name="failing_server",
                model="gpt-4",
            ) as executor:
                pass  # Should not reach here

    @pytest.mark.asyncio
    async def test_error_handling_during_tool_loading(self):
        """Test error handling when tool loading fails."""
        mock_client = MockMCPClient()

        with patch("langchain.agents.mcp_utils.load_mcp_tools") as mock_load_tools:
            mock_load_tools.side_effect = RuntimeError("Failed to load tools")

            with pytest.raises(RuntimeError, match="Failed to load tools"):
                async with StatefulMCPAgentExecutor(
                    client=mock_client,
                    server_name="test_server",
                    model="gpt-4",
                ) as executor:
                    pass  # Should not reach here

            # Verify session was cleaned up even though initialization failed
            session = mock_client.sessions["test_server"]
            assert session.cleanup_called

    @pytest.mark.asyncio
    async def test_runtime_error_when_not_in_context(self):
        """Test that RuntimeError is raised when using executor outside context manager."""
        mock_client = MockMCPClient()

        executor = StatefulMCPAgentExecutor(
            client=mock_client,
            server_name="test_server",
            model="gpt-4",
        )

        # Try to use without entering context manager
        with pytest.raises(RuntimeError, match="Agent not initialized"):
            await executor.ainvoke({"messages": []})

        with pytest.raises(RuntimeError, match="Agent not initialized"):
            executor.invoke({"messages": []})

        with pytest.raises(RuntimeError, match="Agent not initialized"):
            async for _ in executor.astream({"messages": []}):
                pass

        with pytest.raises(RuntimeError, match="Agent not initialized"):
            for _ in executor.stream({"messages": []}):
                pass

    @pytest.mark.asyncio
    async def test_tools_maintain_state_between_invocations(self):
        """Test that tools maintain state between invocations."""
        mock_client = MockMCPClient()

        # Create a stateful tool that maintains a counter
        class StatefulTool(BaseTool):
            name: str = "stateful_tool"
            description: str = "A tool that maintains state"

            def __init__(self):
                super().__init__()
                self.call_count = 0
                self.session_id = None

            def _run(self, session_id: str) -> str:
                if self.session_id is None:
                    self.session_id = session_id
                self.call_count += 1
                return f"Call {self.call_count} in session {self.session_id}"

            async def _arun(self, session_id: str) -> str:
                return self._run(session_id)

        stateful_tool = StatefulTool()
        mock_tools = [stateful_tool]

        with patch("langchain.agents.mcp_utils.load_mcp_tools") as mock_load_tools:
            mock_load_tools.return_value = mock_tools

            with patch("langchain.agents.mcp_utils.create_agent") as mock_create_agent:
                # Create mock agent that uses the stateful tool
                mock_agent = MagicMock()

                async def mock_ainvoke(input_dict, config=None, **kwargs):
                    # Simulate tool usage
                    result = stateful_tool._run("session_123")
                    return {"messages": [AIMessage(content=result)]}

                mock_agent.ainvoke = mock_ainvoke
                mock_create_agent.return_value = mock_agent

                async with StatefulMCPAgentExecutor(
                    client=mock_client,
                    server_name="stateful_server",
                    model="gpt-4",
                ) as executor:
                    # Make multiple calls
                    result1 = await executor.ainvoke({"messages": [HumanMessage(content="Call 1")]})
                    result2 = await executor.ainvoke({"messages": [HumanMessage(content="Call 2")]})
                    result3 = await executor.ainvoke({"messages": [HumanMessage(content="Call 3")]})

                    # Verify state was maintained
                    assert "Call 1 in session session_123" in result1["messages"][-1].content
                    assert "Call 2 in session session_123" in result2["messages"][-1].content
                    assert "Call 3 in session session_123" in result3["messages"][-1].content

                    # Verify same session was used throughout
                    assert stateful_tool.call_count == 3
                    assert stateful_tool.session_id == "session_123"


class TestCreateStatefulMCPAgent:
    """Test cases for create_stateful_mcp_agent factory function."""

    @pytest.mark.asyncio
    async def test_agent_creation_with_session(self):
        """Test that agent is created with persistent session."""
        mock_client = MockMCPClient()
        mock_tools = [mock_browser_navigate]

        with patch("langchain.agents.mcp_utils.load_mcp_tools") as mock_load_tools:
            mock_load_tools.return_value = mock_tools

            with patch("langchain.agents.mcp_utils.create_agent") as mock_create_agent:
                mock_agent = MagicMock()
                mock_create_agent.return_value = mock_agent

                agent, session = await create_stateful_mcp_agent(
                    client=mock_client,
                    server_name="test_server",
                    model="gpt-4",
                    system_prompt="Test prompt",
                    auto_cleanup=False,
                )

                # Verify agent was created
                assert agent == mock_agent

                # Verify session is open
                assert session.is_open
                assert not session.cleanup_called

                # Manual cleanup required when auto_cleanup=False
                await session.__aexit__(None, None, None)
                assert session.cleanup_called

    @pytest.mark.asyncio
    async def test_auto_cleanup_mode(self):
        """Test auto-cleanup mode with __del__ injection."""
        mock_client = MockMCPClient()
        mock_tools = [mock_browser_navigate]

        with patch("langchain.agents.mcp_utils.load_mcp_tools") as mock_load_tools:
            mock_load_tools.return_value = mock_tools

            with patch("langchain.agents.mcp_utils.create_agent") as mock_create_agent:
                mock_agent = MagicMock()
                mock_create_agent.return_value = mock_agent

                agent, session = await create_stateful_mcp_agent(
                    client=mock_client,
                    server_name="test_server",
                    model="gpt-4",
                    auto_cleanup=True,  # Enable auto-cleanup
                )

                # Verify __del__ was added to agent
                assert hasattr(agent, "__del__")

                # Session should still be open
                assert session.is_open

                # Manually cleanup for test
                await session.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_error_handling_during_creation(self):
        """Test that session is cleaned up if agent creation fails."""
        mock_client = MockMCPClient()

        with patch("langchain.agents.mcp_utils.load_mcp_tools") as mock_load_tools:
            mock_load_tools.side_effect = ValueError("Tool loading failed")

            with pytest.raises(ValueError, match="Tool loading failed"):
                await create_stateful_mcp_agent(
                    client=mock_client,
                    server_name="test_server",
                    model="gpt-4",
                )

            # Verify session was cleaned up
            session = mock_client.sessions["test_server"]
            assert session.cleanup_called


class TestMCPAgentSession:
    """Test cases for mcp_agent_session context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_lifecycle(self):
        """Test that context manager properly manages session lifecycle."""
        mock_client = MockMCPClient()
        mock_tools = [mock_browser_navigate]

        with patch("langchain.agents.mcp_utils.load_mcp_tools") as mock_load_tools:
            mock_load_tools.return_value = mock_tools

            with patch("langchain.agents.mcp_utils.create_agent") as mock_create_agent:
                mock_agent = MagicMock()
                mock_agent.ainvoke = AsyncMock(
                    return_value={"messages": [AIMessage(content="Done")]}
                )
                mock_create_agent.return_value = mock_agent

                async with mcp_agent_session(
                    client=mock_client,
                    server_name="test_server",
                    model="gpt-4",
                    system_prompt="Test prompt",
                ) as agent:
                    # Verify agent was created
                    assert agent == mock_agent

                    # Verify session is open
                    session = mock_client.sessions["test_server"]
                    assert session.is_open

                    # Use the agent
                    result = await agent.ainvoke({"messages": [HumanMessage(content="Test")]})
                    assert result["messages"][-1].content == "Done"

                # Verify session was cleaned up after context exit
                assert session.cleanup_called
                assert not session.is_open

    @pytest.mark.asyncio
    async def test_error_propagation_and_cleanup(self):
        """Test that errors are propagated and session is still cleaned up."""
        mock_client = MockMCPClient()
        mock_tools = [mock_browser_navigate]

        with patch("langchain.agents.mcp_utils.load_mcp_tools") as mock_load_tools:
            mock_load_tools.return_value = mock_tools

            with patch("langchain.agents.mcp_utils.create_agent") as mock_create_agent:
                mock_agent = MagicMock()
                mock_agent.ainvoke = AsyncMock(side_effect=RuntimeError("Agent failed"))
                mock_create_agent.return_value = mock_agent

                with pytest.raises(RuntimeError, match="Agent failed"):
                    async with mcp_agent_session(
                        client=mock_client,
                        server_name="test_server",
                        model="gpt-4",
                    ) as agent:
                        # This should raise an error
                        await agent.ainvoke({"messages": []})

                # Verify session was still cleaned up despite error
                session = mock_client.sessions["test_server"]
                assert session.cleanup_called
                assert not session.is_open


class TestSessionStatePersistence:
    """Test cases specifically for verifying session state persistence."""

    @pytest.mark.asyncio
    async def test_browser_session_persistence_simulation(self):
        """Simulate browser session persistence across navigation and interactions."""
        mock_client = MockMCPClient()

        # Simulate browser state
        browser_state = {
            "current_url": None,
            "page_elements": [],
            "session_active": False,
        }

        @tool
        def browser_navigate_stateful(url: str) -> str:
            """Navigate to URL and maintain session."""
            if not browser_state["session_active"]:
                browser_state["session_active"] = True
            browser_state["current_url"] = url
            browser_state["page_elements"] = [f"element_{i}" for i in range(5)]
            return f"Navigated to {url}, found {len(browser_state['page_elements'])} elements"

        @tool
        def browser_click_stateful(element_ref: str) -> str:
            """Click element in current session."""
            if not browser_state["session_active"]:
                raise RuntimeError("No active browser session")
            if element_ref not in browser_state["page_elements"]:
                raise RuntimeError(f"Element {element_ref} not found in current page")
            return f"Clicked {element_ref} on {browser_state['current_url']}"

        mock_tools = [browser_navigate_stateful, browser_click_stateful]

        with patch("langchain.agents.mcp_utils.load_mcp_tools") as mock_load_tools:
            mock_load_tools.return_value = mock_tools

            with patch("langchain.agents.mcp_utils.create_agent") as mock_create_agent:
                # Create mock agent that simulates tool usage
                mock_agent = MagicMock()

                async def simulate_browser_interaction(input_dict, config=None, **kwargs):
                    message = input_dict["messages"][-1].content

                    if "navigate" in message.lower():
                        result = browser_navigate_stateful.invoke({"url": "https://example.com"})
                    elif "click" in message.lower():
                        result = browser_click_stateful.invoke({"element_ref": "element_0"})
                    else:
                        result = "Unknown command"

                    return {"messages": [AIMessage(content=result)]}

                mock_agent.ainvoke = simulate_browser_interaction
                mock_create_agent.return_value = mock_agent

                async with StatefulMCPAgentExecutor(
                    client=mock_client,
                    server_name="playwright",
                    model="gpt-4",
                ) as executor:
                    # Navigate to page
                    nav_result = await executor.ainvoke(
                        {"messages": [HumanMessage(content="Navigate to example.com")]}
                    )
                    assert "Navigated to https://example.com" in nav_result["messages"][-1].content
                    assert browser_state["session_active"]

                    # Click element - should work because session is maintained
                    click_result = await executor.ainvoke(
                        {"messages": [HumanMessage(content="Click element_0")]}
                    )
                    assert (
                        "Clicked element_0 on https://example.com"
                        in click_result["messages"][-1].content
                    )

                    # Verify browser state was maintained throughout
                    assert browser_state["current_url"] == "https://example.com"
                    assert "element_0" in browser_state["page_elements"]
                    assert browser_state["session_active"]

    @pytest.mark.asyncio
    async def test_multiple_server_sessions(self):
        """Test managing multiple MCP server sessions simultaneously."""
        mock_client = MockMCPClient()

        # Track which sessions are used
        sessions_used = set()

        async def track_session_tool(session_name: str) -> str:
            sessions_used.add(session_name)
            return f"Used session: {session_name}"

        with patch("langchain.agents.mcp_utils.load_mcp_tools") as mock_load_tools:
            # Return different tools for different servers
            def get_tools_for_server(session):
                if hasattr(session, "name"):
                    if "playwright" in session.name:
                        return [mock_browser_navigate]
                    elif "database" in session.name:
                        return [mock_browser_click]  # Different tool for database
                return []

            mock_load_tools.side_effect = get_tools_for_server

            with patch("langchain.agents.mcp_utils.create_agent") as mock_create_agent:
                mock_agent = MagicMock()
                mock_create_agent.return_value = mock_agent

                # Create executors for different servers
                async with StatefulMCPAgentExecutor(
                    client=mock_client,
                    server_name="playwright",
                    model="gpt-4",
                ) as playwright_executor:
                    async with StatefulMCPAgentExecutor(
                        client=mock_client,
                        server_name="database",
                        model="gpt-4",
                    ) as db_executor:
                        # Verify different sessions were created
                        assert "playwright" in mock_client.sessions
                        assert "database" in mock_client.sessions

                        playwright_session = mock_client.sessions["playwright"]
                        db_session = mock_client.sessions["database"]

                        # Verify both sessions are open
                        assert playwright_session.is_open
                        assert db_session.is_open

                        # Verify sessions are different
                        assert playwright_session != db_session
                        assert playwright_session.name != db_session.name

                # Verify both sessions were cleaned up
                assert playwright_session.cleanup_called
                assert db_session.cleanup_called

"""Integration test demonstrating MCP stateful session management for Playwright.

This test verifies that the stateful MCP session management fixes the browser
session termination issue by maintaining persistent connections across multiple
tool invocations.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool

from langchain.agents import create_agent
from langchain.agents.mcp_utils import (
    StatefulMCPAgentExecutor,
    create_stateful_mcp_agent,
    mcp_agent_session,
)


class MockPlaywrightServer:
    """Mock Playwright MCP server for testing browser automation."""
    
    def __init__(self):
        self.browser_sessions = {}
        self.current_session_id = None
        self.session_counter = 0
        
    def create_session(self, session_id: str) -> Dict[str, Any]:
        """Create a new browser session."""
        self.session_counter += 1
        session = {
            "id": session_id,
            "browser": "chromium",
            "pages": {},
            "current_page": None,
            "is_active": True,
            "creation_order": self.session_counter,
        }
        self.browser_sessions[session_id] = session
        self.current_session_id = session_id
        return session
    
    def get_session(self, session_id: str) -> Dict[str, Any] | None:
        """Get an existing browser session."""
        return self.browser_sessions.get(session_id)
    
    def close_session(self, session_id: str) -> None:
        """Close a browser session."""
        if session_id in self.browser_sessions:
            self.browser_sessions[session_id]["is_active"] = False
            if self.current_session_id == session_id:
                self.current_session_id = None


class MockMCPSession:
    """Mock MCP session that simulates Playwright browser state."""
    
    def __init__(self, server: MockPlaywrightServer, session_name: str):
        self.server = server
        self.session_name = session_name
        self.session_id = f"{session_name}_{id(self)}"
        self.is_open = False
        self.browser_session = None
        
    async def __aenter__(self):
        """Enter the session context and create browser session."""
        self.is_open = True
        self.browser_session = self.server.create_session(self.session_id)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the session context and close browser session."""
        self.is_open = False
        if self.browser_session:
            self.server.close_session(self.session_id)
        return False


class MockMCPClient:
    """Mock MultiServerMCPClient for testing."""
    
    def __init__(self, playwright_server: MockPlaywrightServer):
        self.playwright_server = playwright_server
        self.sessions_created = []
        
    def session(self, server_name: str) -> MockMCPSession:
        """Create a mock session for the specified server."""
        session = MockMCPSession(self.playwright_server, server_name)
        self.sessions_created.append(session)
        return session


def create_playwright_tools(session: MockMCPSession) -> List[BaseTool]:
    """Create mock Playwright tools that use the session's browser state."""
    
    @tool
    def playwright_navigate(url: str) -> str:
        """Navigate to a URL in the browser.
        
        Args:
            url: The URL to navigate to.
        """
        if not session.is_open or not session.browser_session:
            raise RuntimeError("No active browser session")
        
        if not session.browser_session["is_active"]:
            raise RuntimeError("Browser session has been closed")
        
        # Create a new page in the session
        page_id = f"page_{len(session.browser_session['pages'])}"
        session.browser_session["pages"][page_id] = {
            "url": url,
            "elements": [f"button_{i}" for i in range(3)],
            "text_fields": [f"input_{i}" for i in range(2)],
        }
        session.browser_session["current_page"] = page_id
        
        return f"Navigated to {url} in session {session.session_id}"
    
    @tool
    def playwright_click(selector: str) -> str:
        """Click an element in the current page.
        
        Args:
            selector: The element selector to click.
        """
        if not session.is_open or not session.browser_session:
            raise RuntimeError("No active browser session")
        
        if not session.browser_session["is_active"]:
            raise RuntimeError("Browser session has been closed")
        
        current_page_id = session.browser_session.get("current_page")
        if not current_page_id:
            raise RuntimeError("No page loaded - navigate to a URL first")
        
        page = session.browser_session["pages"][current_page_id]
        
        # Check if element exists
        if selector not in page["elements"]:
            raise RuntimeError(f"Element '{selector}' not found. Available: {page['elements']}")
        
        return f"Clicked {selector} on {page['url']} in session {session.session_id}"
    
    @tool
    def playwright_type(selector: str, text: str) -> str:
        """Type text into an input field.
        
        Args:
            selector: The input field selector.
            text: The text to type.
        """
        if not session.is_open or not session.browser_session:
            raise RuntimeError("No active browser session")
        
        if not session.browser_session["is_active"]:
            raise RuntimeError("Browser session has been closed")
        
        current_page_id = session.browser_session.get("current_page")
        if not current_page_id:
            raise RuntimeError("No page loaded - navigate to a URL first")
        
        page = session.browser_session["pages"][current_page_id]
        
        # Check if text field exists
        if selector not in page["text_fields"]:
            raise RuntimeError(f"Text field '{selector}' not found. Available: {page['text_fields']}")
        
        # Store the typed text
        if "typed_text" not in page:
            page["typed_text"] = {}
        page["typed_text"][selector] = text
        
        return f"Typed '{text}' into {selector} on {page['url']} in session {session.session_id}"
    
    # Mark tools as MCP tools
    for tool_func in [playwright_navigate, playwright_click, playwright_type]:
        tool_func.metadata = {"mcp_server": "playwright", "source": "mcp"}
    
    return [playwright_navigate, playwright_click, playwright_type]


class TestMCPPlaywrightSessionIntegration:
    """Integration tests for MCP Playwright session management."""
    
    @pytest.mark.asyncio
    async def test_stateless_session_fails_on_multiple_operations(self):
        """Test that stateless MCP usage causes session termination between calls."""
        playwright_server = MockPlaywrightServer()
        mock_client = MockMCPClient(playwright_server)
        
        # Simulate stateless tool usage (new session per tool call)
        async def simulate_stateless_tools():
            # First tool call - navigate
            async with mock_client.session("playwright") as session1:
                tools = create_playwright_tools(session1)
                nav_tool = tools[0]
                result1 = nav_tool.invoke({"url": "https://example.com"})
                assert "Navigated to https://example.com" in result1
            
            # Second tool call - click (different session)
            async with mock_client.session("playwright") as session2:
                tools = create_playwright_tools(session2)
                click_tool = tools[1]
                # This should fail because it's a new session without navigation
                with pytest.raises(RuntimeError, match="No page loaded"):
                    click_tool.invoke({"selector": "button_0"})
        
        await simulate_stateless_tools()
        
        # Verify that multiple sessions were created
        assert len(mock_client.sessions_created) == 2
        assert playwright_server.session_counter == 2
    
    @pytest.mark.asyncio
    async def test_stateful_session_maintains_browser_state(self):
        """Test that stateful MCP session maintains browser state across operations."""
        playwright_server = MockPlaywrightServer()
        mock_client = MockMCPClient(playwright_server)
        
        # Use stateful session management
        async with mock_client.session("playwright") as session:
            tools = create_playwright_tools(session)
            nav_tool, click_tool, type_tool = tools
            
            # Navigate to a page
            nav_result = nav_tool.invoke({"url": "https://example.com"})
            assert "Navigated to https://example.com" in nav_result
            assert session.session_id in nav_result
            
            # Click an element - should work because session is maintained
            click_result = click_tool.invoke({"selector": "button_0"})
            assert "Clicked button_0 on https://example.com" in click_result
            assert session.session_id in click_result
            
            # Type text - should also work in the same session
            type_result = type_tool.invoke({"selector": "input_0", "text": "Hello World"})
            assert "Typed 'Hello World' into input_0" in type_result
            assert session.session_id in type_result
            
            # Verify all operations used the same session
            assert playwright_server.session_counter == 1
            assert len(playwright_server.browser_sessions) == 1
            
            # Verify browser state was maintained
            browser_session = playwright_server.get_session(session.session_id)
            assert browser_session is not None
            assert browser_session["is_active"]
            assert "page_0" in browser_session["pages"]
            assert browser_session["pages"]["page_0"]["typed_text"]["input_0"] == "Hello World"
    
    @pytest.mark.asyncio
    async def test_stateful_mcp_agent_executor(self):
        """Test StatefulMCPAgentExecutor maintains session across agent invocations."""
        playwright_server = MockPlaywrightServer()
        mock_client = MockMCPClient(playwright_server)
        
        with patch("langchain.agents.mcp_utils.load_mcp_tools") as mock_load_tools:
            # Setup mock tool loading
            async def load_tools_with_session(session):
                return create_playwright_tools(session)
            
            mock_load_tools.side_effect = load_tools_with_session
            
            with patch("langchain.agents.mcp_utils.create_agent") as mock_create_agent:
                # Create a mock agent that simulates tool usage
                mock_agent = MagicMock()
                
                async def simulate_agent_invoke(input_dict, config=None, **kwargs):
                    messages = input_dict.get("messages", [])
                    if not messages:
                        return {"messages": [AIMessage(content="No input provided")]}
                    
                    last_message = messages[-1].content.lower()
                    
                    # Get the tools from the mock
                    tools = mock_create_agent.call_args[1]["tools"]
                    
                    if "navigate" in last_message:
                        result = tools[0].invoke({"url": "https://test.com"})
                    elif "click" in last_message:
                        result = tools[1].invoke({"selector": "button_1"})
                    elif "type" in last_message:
                        result = tools[2].invoke({"selector": "input_1", "text": "Test input"})
                    else:
                        result = "Unknown command"
                    
                    return {"messages": [AIMessage(content=result)]}
                
                mock_agent.ainvoke = simulate_agent_invoke
                mock_create_agent.return_value = mock_agent
                
                # Use StatefulMCPAgentExecutor
                async with StatefulMCPAgentExecutor(
                    client=mock_client,
                    server_name="playwright",
                    model="gpt-4",
                    system_prompt="You are a browser automation assistant.",
                ) as executor:
                    # Multiple operations in the same session
                    nav_result = await executor.ainvoke({
                        "messages": [HumanMessage(content="Navigate to test.com")]
                    })
                    assert "Navigated to https://test.com" in nav_result["messages"][-1].content
                    
                    click_result = await executor.ainvoke({
                        "messages": [HumanMessage(content="Click button_1")]
                    })
                    assert "Clicked button_1" in click_result["messages"][-1].content
                    
                    type_result = await executor.ainvoke({
                        "messages": [HumanMessage(content="Type some text")]
                    })
                    assert "Typed 'Test input'" in type_result["messages"][-1].content
                    
                    # Verify single session was used
                    assert playwright_server.session_counter == 1
                    assert len(mock_client.sessions_created) == 1
    
    @pytest.mark.asyncio
    async def test_mcp_agent_session_context_manager(self):
        """Test mcp_agent_session context manager for simplified usage."""
        playwright_server = MockPlaywrightServer()
        mock_client = MockMCPClient(playwright_server)
        
        with patch("langchain.agents.mcp_utils.load_mcp_tools") as mock_load_tools:
            async def load_tools_with_session(session):
                return create_playwright_tools(session)
            
            mock_load_tools.side_effect = load_tools_with_session
            
            with patch("langchain.agents.mcp_utils.create_agent") as mock_create_agent:
                mock_agent = MagicMock()
                mock_agent.ainvoke = AsyncMock(return_value={
                    "messages": [AIMessage(content="Task completed")]
                })
                mock_create_agent.return_value = mock_agent
                
                async with mcp_agent_session(
                    client=mock_client,
                    server_name="playwright",
                    model="gpt-4",
                ) as agent:
                    # Use the agent
                    result = await agent.ainvoke({
                        "messages": [HumanMessage(content="Automate browser")]
                    })
                    assert result["messages"][-1].content == "Task completed"
                    
                    # Verify session was created
                    assert len(mock_client.sessions_created) == 1
                    assert playwright_server.session_counter == 1
                
                # Verify session was closed after context exit
                session = mock_client.sessions_created[0]
                assert not session.is_open
                browser_session = playwright_server.get_session(session.session_id)
                assert browser_session is not None
                assert not browser_session["is_active"]
    
    @pytest.mark.asyncio
    async def test_element_references_remain_valid_across_calls(self):
        """Test that element references remain valid across multiple tool calls."""
        playwright_server = MockPlaywrightServer()
        mock_client = MockMCPClient(playwright_server)
        
        async with mock_client.session("playwright") as session:
            tools = create_playwright_tools(session)
            nav_tool, click_tool, type_tool = tools
            
            # Navigate to create elements
            nav_tool.invoke({"url": "https://form.example.com"})
            
            # Interact with multiple elements in sequence
            elements_to_click = ["button_0", "button_1", "button_2"]
            for element in elements_to_click:
                result = click_tool.invoke({"selector": element})
                assert f"Clicked {element}" in result
                assert session.session_id in result
            
            # Type in multiple fields
            fields_to_fill = [("input_0", "First Name"), ("input_1", "Last Name")]
            for field, text in fields_to_fill:
                result = type_tool.invoke({"selector": field, "text": text})
                assert f"Typed '{text}' into {field}" in result
                assert session.session_id in result
            
            # Verify all operations succeeded in the same session
            browser_session = playwright_server.get_session(session.session_id)
            assert browser_session["is_active"]
            page = browser_session["pages"]["page_0"]
            assert page["typed_text"]["input_0"] == "First Name"
            assert page["typed_text"]["input_1"] == "Last Name"
    
    @pytest.mark.asyncio
    async def test_session_cleanup_on_error(self):
        """Test that session is properly cleaned up even when errors occur."""
        playwright_server = MockPlaywrightServer()
        mock_client = MockMCPClient(playwright_server)
        
        with pytest.raises(RuntimeError, match="Invalid selector"):
            async with mock_client.session("playwright") as session:
                tools = create_playwright_tools(session)
                nav_tool, click_tool, _ = tools
                
                # Navigate successfully
                nav_tool.invoke({"url": "https://example.com"})
                
                # Try to click non-existent element
                click_tool.invoke({"selector": "non_existent_button"})
        
        # Verify session was still cleaned up despite error
        assert len(mock_client.sessions_created) == 1
        session = mock_client.sessions_created[0]
        assert not session.is_open
        
        browser_session = playwright_server.get_session(session.session_id)
        assert not browser_session["is_active"]
    
    @pytest.mark.asyncio
    async def test_create_agent_with_mcp_session_config(self):
        """Test create_agent with mcp_session_config parameter."""
        playwright_server = MockPlaywrightServer()
        mock_client = MockMCPClient(playwright_server)
        
        # Create mock MCP tools
        mock_tools = [
            Mock(spec=BaseTool, name="playwright_navigate", metadata={"mcp_server": "playwright"}),
            Mock(spec=BaseTool, name="playwright_click", metadata={"mcp_server": "playwright"}),
        ]
        
        with patch("langchain.chat_models.init_chat_model") as mock_init_model:
            mock_model = MagicMock()
            mock_init_model.return_value = mock_model
            
            # Create agent with MCP session config
            agent = create_agent(
                model="gpt-4",
                tools=mock_tools,
                mcp_session_config={
                    "client": mock_client,
                    "server_name": "playwright",
                    "auto_cleanup": True,
                }
            )
            
            # Verify tools were marked with session config
            for tool in mock_tools:
                assert "__mcp_session_config__" in tool.metadata
                assert tool.metadata["__mcp_session_config__"]["server_name"] == "playwright"


class TestBrowserSessionPersistence:
    """Specific tests for browser session persistence scenarios."""
    
    @pytest.mark.asyncio
    async def test_complex_browser_workflow(self):
        """Test a complex browser workflow that requires session persistence."""
        playwright_server = MockPlaywrightServer()
        mock_client = MockMCPClient(playwright_server)
        
        async with mock_client.session("playwright") as session:
            tools = create_playwright_tools(session)
            nav_tool, click_tool, type_tool = tools
            
            # Simulate a multi-step form submission workflow
            workflow_steps = [
                ("navigate", {"url": "https://app.example.com/login"}),
                ("type", {"selector": "input_0", "text": "user@example.com"}),
                ("type", {"selector": "input_1", "text": "password123"}),
                ("click", {"selector": "button_0"}),  # Login button
                ("navigate", {"url": "https://app.example.com/dashboard"}),
                ("click", {"selector": "button_1"}),  # Some dashboard action
            ]
            
            results = []
            for step_type, params in workflow_steps:
                if step_type == "navigate":
                    result = nav_tool.invoke(params)
                elif step_type == "click":
                    # For dashboard navigation, we need to navigate first
                    if params["selector"] == "button_1" and "dashboard" not in [p["url"] for p in session.browser_session["pages"].values()]:
                        nav_tool.invoke({"url": "https://app.example.com/dashboard"})
                    result = click_tool.invoke(params)
                elif step_type == "type":
                    result = type_tool.invoke(params)
                else:
                    result = "Unknown step"
                
                results.append(result)
                # Verify session ID is consistent
                assert session.session_id in result
            
            # Verify all steps completed successfully
            assert len(results) == len(workflow_steps)
            assert all(session.session_id in r for r in results)
            
            # Verify session remained active throughout
            assert playwright_server.session_counter == 1
            browser_session = playwright_server.get_session(session.session_id)
            assert browser_session["is_active"]
            assert len(browser_session["pages"]) >= 2  # Login and dashboard pages
    
    @pytest.mark.asyncio
    async def test_session_persistence_prevents_ref_not_found_errors(self):
        """Test that session persistence prevents 'Ref not found' errors."""
        playwright_server = MockPlaywrightServer()
        mock_client = MockMCPClient(playwright_server)
        
        # Track element references
        element_refs = {}
        
        async with mock_client.session("playwright") as session:
            tools = create_playwright_tools(session)
            nav_tool, click_tool, _ = tools
            
            # Navigate and store element references
            nav_tool.invoke({"url": "https://example.com"})
            
            # Store available elements (simulating element discovery)
            browser_session = playwright_server.get_session(session.session_id)
            page = browser_session["pages"]["page_0"]
            for element in page["elements"]:
                element_refs[element] = session.session_id
            
            # Verify all element references remain valid
            for element_id in element_refs:
                result = click_tool.invoke({"selector": element_id})
                assert "Clicked" in result
                assert element_refs[element_id] in result
                
                # Verify no "Ref not found" error occurred
                assert "not found" not in result.lower()
                assert "ref" not in result.lower() or "references" in result.lower()
            
            # Verify session remained consistent
            assert len(set(element_refs.values())) == 1  # All refs from same session

"""Tests for Windows event loop policy fix for async PostgreSQL compatibility."""

import asyncio
import sys

import pytest

from langchain.agents import create_agent, set_windows_selector_event_loop_policy
from langchain_core.messages import HumanMessage

from .model import FakeToolCallingModel


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test")
def test_set_windows_selector_event_loop_policy_on_windows() -> None:
    """Test that set_windows_selector_event_loop_policy sets correct policy on Windows."""
    # Reset event loop policy to default
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    
    # Call the function
    set_windows_selector_event_loop_policy()
    
    # Verify the policy is set to WindowsSelectorEventLoopPolicy
    policy = asyncio.get_event_loop_policy()
    assert isinstance(policy, asyncio.WindowsSelectorEventLoopPolicy)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test")
def test_set_windows_selector_event_loop_policy_with_running_loop() -> None:
    """Test that function works when event loop is already running."""
    # Create a new event loop with default policy
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # If we have a ProactorEventLoop, the function should change it
        if isinstance(loop, asyncio.ProactorEventLoop):
            set_windows_selector_event_loop_policy()
            # Policy should be changed for future loops
            new_policy = asyncio.get_event_loop_policy()
            assert isinstance(new_policy, asyncio.WindowsSelectorEventLoopPolicy)
    finally:
        loop.close()
        asyncio.set_event_loop(None)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test")
def test_set_windows_selector_event_loop_policy_without_running_loop() -> None:
    """Test that function works when no event loop is running."""
    # Clear any existing event loop
    try:
        loop = asyncio.get_running_loop()
        # If we're in an async context, skip this test
        pytest.skip("Cannot test without running loop in async context")
    except RuntimeError:
        # No running loop - this is what we want
        pass
    
    # Reset policy
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    
    # Call the function
    set_windows_selector_event_loop_policy()
    
    # Verify policy is set
    policy = asyncio.get_event_loop_policy()
    assert isinstance(policy, asyncio.WindowsSelectorEventLoopPolicy)


@pytest.mark.skipif(sys.platform == "win32", reason="Non-Windows test")
def test_set_windows_selector_event_loop_policy_on_non_windows() -> None:
    """Test that function does nothing on non-Windows systems."""
    # Get current policy
    original_policy = asyncio.get_event_loop_policy()
    
    # Call the function
    set_windows_selector_event_loop_policy()
    
    # Policy should remain unchanged on non-Windows
    current_policy = asyncio.get_event_loop_policy()
    assert type(current_policy) == type(original_policy)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test")
def test_create_agent_calls_windows_event_loop_policy() -> None:
    """Test that create_agent automatically calls set_windows_selector_event_loop_policy."""
    # Reset to default policy
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    
    # Create an agent
    agent = create_agent(
        model=FakeToolCallingModel(tool_calls=[[], []]),
        tools=[],
    )
    
    # Verify the policy was set by create_agent
    policy = asyncio.get_event_loop_policy()
    assert isinstance(policy, asyncio.WindowsSelectorEventLoopPolicy)
    
    # Verify agent was created successfully
    assert agent is not None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test")
def test_create_agent_with_tools_sets_policy() -> None:
    """Test that create_agent sets policy even when tools are provided."""
    from langchain_core.tools import tool
    
    @tool
    def test_tool(x: int) -> str:
        """Test tool."""
        return f"Result: {x}"
    
    # Reset to default policy
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    
    # Create agent with tools
    agent = create_agent(
        model=FakeToolCallingModel(tool_calls=[[], []]),
        tools=[test_tool],
    )
    
    # Verify policy was set
    policy = asyncio.get_event_loop_policy()
    assert isinstance(policy, asyncio.WindowsSelectorEventLoopPolicy)
    
    # Verify agent works
    result = agent.invoke({"messages": [HumanMessage("test")]})
    assert "messages" in result


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test")
def test_multiple_calls_to_set_windows_selector_event_loop_policy() -> None:
    """Test that calling the function multiple times is safe."""
    # Reset policy
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    
    # Call multiple times
    set_windows_selector_event_loop_policy()
    set_windows_selector_event_loop_policy()
    set_windows_selector_event_loop_policy()
    
    # Should still be set correctly
    policy = asyncio.get_event_loop_policy()
    assert isinstance(policy, asyncio.WindowsSelectorEventLoopPolicy)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test")
async def test_windows_event_loop_policy_with_async_operations() -> None:
    """Test that the policy works with async operations."""
    # Set the policy
    set_windows_selector_event_loop_policy()
    
    # Verify we can create async operations
    async def async_task() -> str:
        return "async result"
    
    result = await async_task()
    assert result == "async result"
    
    # Verify policy is still set
    policy = asyncio.get_event_loop_policy()
    assert isinstance(policy, asyncio.WindowsSelectorEventLoopPolicy)


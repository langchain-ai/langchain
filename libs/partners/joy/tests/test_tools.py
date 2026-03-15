"""Tests for Joy LangChain tools.

These tests verify the tools work correctly against the live Joy API.
"""

import pytest

from langchain_joy import JoyDiscoverTool, JoyTrustTool


def test_trust_tool_init() -> None:
    """Test JoyTrustTool initializes correctly."""
    tool = JoyTrustTool()
    assert tool.name == "joy_trust_verify"
    assert "trust" in tool.description.lower()


def test_discover_tool_init() -> None:
    """Test JoyDiscoverTool initializes correctly."""
    tool = JoyDiscoverTool()
    assert tool.name == "joy_discover_agents"
    assert "discover" in tool.description.lower()


@pytest.mark.requires("httpx")
def test_trust_tool_invoke() -> None:
    """Test JoyTrustTool against live API."""
    tool = JoyTrustTool()
    # Use a known agent ID from Joy network
    result = tool.invoke({"agent_id": "ag_229e507d7d87f35cc2bc17ea"})
    assert "Trust Score" in result
    assert "Trusted" in result


@pytest.mark.requires("httpx")
def test_discover_tool_invoke() -> None:
    """Test JoyDiscoverTool against live API."""
    tool = JoyDiscoverTool()
    result = tool.invoke({"capability": "github", "limit": 3})
    # Should find at least one agent or return "No trusted agents"
    assert "agents" in result.lower() or "found" in result.lower()

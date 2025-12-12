"""Integration tests for ShellToolMiddleware with create_agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from langgraph._internal._typing import StateLike
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware.shell_tool import ShellToolMiddleware
from langchain.agents.middleware.types import _InputAgentState


def _get_model(provider: str) -> Any:
    """Get chat model for the specified provider."""
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model="claude-sonnet-4-5-20250929")
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model="gpt-4o-mini")
    else:
        msg = f"Unknown provider: {provider}"
        raise ValueError(msg)


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_shell_tool_basic_execution(tmp_path: Path, provider: str) -> None:
    """Test basic shell command execution across different models."""
    pytest.importorskip(f"langchain_{provider}")

    workspace = tmp_path / "workspace"
    agent: CompiledStateGraph[Any, Any, _InputAgentState, Any] = create_agent(
        model=_get_model(provider),
        middleware=[ShellToolMiddleware(workspace_root=workspace)],
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Run the command 'echo hello' and tell me what it outputs")]}
    )

    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert len(tool_messages) > 0, "Shell tool should have been called"

    tool_outputs = [msg.content for msg in tool_messages]
    assert any("hello" in output.lower() for output in tool_outputs), (
        "Shell output should contain 'hello'"
    )


@pytest.mark.requires("langchain_anthropic")
def test_shell_session_persistence(tmp_path: Path) -> None:
    """Test shell session state persists across multiple tool calls."""
    workspace = tmp_path / "workspace"
    agent: CompiledStateGraph[Any, Any, _InputAgentState, Any] = create_agent(
        model=_get_model("anthropic"),
        middleware=[ShellToolMiddleware(workspace_root=workspace)],
    )

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    "First run 'export TEST_VAR=hello'. "
                    "Then run 'echo $TEST_VAR' to verify it persists."
                )
            ]
        }
    )

    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert len(tool_messages) >= 2, "Shell tool should be called multiple times"

    tool_outputs = [msg.content for msg in tool_messages]
    assert any("hello" in output for output in tool_outputs), "Environment variable should persist"


@pytest.mark.requires("langchain_anthropic")
def test_shell_tool_error_handling(tmp_path: Path) -> None:
    """Test shell tool captures command errors."""
    workspace = tmp_path / "workspace"
    agent: CompiledStateGraph[Any, Any, _InputAgentState, Any] = create_agent(
        model=_get_model("anthropic"),
        middleware=[ShellToolMiddleware(workspace_root=workspace)],
    )

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    "Run the command 'ls /nonexistent_directory_12345' and show me the result"
                )
            ]
        }
    )

    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert len(tool_messages) > 0, "Shell tool should have been called"

    tool_outputs = " ".join(msg.content for msg in tool_messages)
    assert (
        "no such file" in tool_outputs.lower()
        or "cannot access" in tool_outputs.lower()
        or "not found" in tool_outputs.lower()
        or "exit code" in tool_outputs.lower()
    ), "Error should be captured in tool output"


@pytest.mark.requires("langchain_anthropic")
def test_shell_tool_with_custom_tools(tmp_path: Path) -> None:
    """Test shell tool works alongside custom tools."""
    from langchain_core.tools import tool

    workspace = tmp_path / "workspace"

    @tool
    def custom_greeting(name: str) -> str:
        """Greet someone by name."""
        return f"Hello, {name}!"

    agent: CompiledStateGraph[Any, Any, _InputAgentState, Any] = create_agent(
        model=_get_model("anthropic"),
        tools=[custom_greeting],
        middleware=[ShellToolMiddleware(workspace_root=workspace)],
    )

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    "First, use the custom_greeting tool to greet 'Alice'. "
                    "Then run the shell command 'echo world'."
                )
            ]
        }
    )

    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert len(tool_messages) >= 2, "Both tools should have been called"

    tool_outputs = " ".join(msg.content for msg in tool_messages)
    assert "Alice" in tool_outputs, "Custom tool should be used"
    assert "world" in tool_outputs, "Shell tool should be used"

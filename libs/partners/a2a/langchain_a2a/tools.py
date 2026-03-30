"""Tools for exposing A2A agents as LangChain tools."""

from collections.abc import Callable
from typing import Any

from langchain_core.tools import StructuredTool

from langchain_a2a.client import MultiAgentA2AClient


def _build_agent_callable(
    client: MultiAgentA2AClient,
    agent_name: str,
) -> Callable[[str], Any]:
    """Build a callable for invoking a single agent."""

    def _call(input: str) -> Any:
        return client.call_agent(agent=agent_name, input=input)

    return _call


def get_tools(
    client: MultiAgentA2AClient,
) -> list[StructuredTool]:
    """Build LangChain tools for available A2A agents."""
    tools: list[StructuredTool] = []
    for agent_name in client.list_agents():
        tools.append(
            StructuredTool.from_function(
                func=_build_agent_callable(client, agent_name),
                name=agent_name,
                description=f"Tool for interacting with the remote A2A agent `{agent_name}`.",
            )
        )
    return tools

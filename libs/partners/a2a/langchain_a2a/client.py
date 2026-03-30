"""Client for interacting with A2A agents."""

from typing import Any


class MultiAgentA2AClient:
    """Client for interacting with multiple A2A agents."""

    def __init__(self, endpoint: str) -> None:
        """Initialize the client.

        Args:
            endpoint: Base URL for the A2A service.
        """
        self.endpoint = endpoint

    def list_agents(self) -> list[str]:
        """List available agents.

        Returns:
            List of agent names.
        """
        # TODO: implement real call
        return []

    def call_agent(self, agent: str, input: str) -> Any:
        """Call a remote A2A agent.

        Args:
            agent: Agent name.
            input: Input string.

        Returns:
            Agent response.
        """
        # TODO: implement real call
        return {"agent": agent, "input": input}

"""LangChain tools for Joy trust network."""

from __future__ import annotations

from typing import Any, Optional, Type

import httpx
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

JOY_API_URL = "https://joy-connect.fly.dev"


class JoyTrustInput(BaseModel):
    """Input for Joy trust verification tool."""

    agent_id: str = Field(description="The Joy agent ID to verify (e.g., 'ag_xxx')")
    min_trust_score: float = Field(
        default=0.5,
        description="Minimum trust score required (0.0-2.0)",
    )


class JoyTrustTool(BaseTool):
    """Tool for verifying agent trustworthiness using Joy network.

    Use this tool to check if an AI agent should be trusted before
    delegating tasks or sharing sensitive information.

    Example:
        from langchain_joy import JoyTrustTool

        tool = JoyTrustTool()
        result = tool.invoke({"agent_id": "ag_xxx", "min_trust_score": 0.5})
    """

    name: str = "joy_trust_verify"
    description: str = (
        "Verify if an AI agent is trustworthy using the Joy trust network. "
        "Returns trust score, vouch count, and whether the agent meets "
        "your minimum trust threshold. Use before delegating sensitive tasks."
    )
    args_schema: Type[BaseModel] = JoyTrustInput

    def _run(
        self,
        agent_id: str,
        min_trust_score: float = 0.5,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Verify an agent's trust status."""
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(
                    f"{JOY_API_URL}/agents/{agent_id}",
                    headers={"User-Agent": "langchain-joy/0.1.0"},
                )
                response.raise_for_status()
                data = response.json()

            trust_score = float(data.get("trust_score", 0))
            vouch_count = int(data.get("vouch_count", 0))
            verified = data.get("verified", False)
            capabilities = data.get("capabilities", [])

            is_trusted = trust_score >= min_trust_score

            return (
                f"Agent: {agent_id}\n"
                f"Trusted: {is_trusted}\n"
                f"Trust Score: {trust_score:.2f} (min: {min_trust_score})\n"
                f"Vouch Count: {vouch_count}\n"
                f"Verified: {verified}\n"
                f"Capabilities: {', '.join(capabilities[:5])}"
            )

        except Exception as e:
            return f"Trust verification failed: {e}"


class JoyDiscoverInput(BaseModel):
    """Input for Joy agent discovery tool."""

    capability: str = Field(
        default="",
        description="Capability to search for (e.g., 'github', 'email', 'code')",
    )
    query: str = Field(
        default="",
        description="Free text search query",
    )
    min_trust_score: float = Field(
        default=0.5,
        description="Minimum trust score required (0.0-2.0)",
    )
    limit: int = Field(
        default=5,
        description="Maximum number of agents to return",
    )


class JoyDiscoverTool(BaseTool):
    """Tool for discovering trusted agents from Joy network.

    Use this tool to find AI agents with specific capabilities
    that meet your trust requirements.

    Example:
        from langchain_joy import JoyDiscoverTool

        tool = JoyDiscoverTool()
        result = tool.invoke({"capability": "github", "min_trust_score": 1.0})
    """

    name: str = "joy_discover_agents"
    description: str = (
        "Discover trusted AI agents from the Joy network. "
        "Search by capability (e.g., 'github', 'email') or free text query. "
        "Returns agents that meet your minimum trust threshold."
    )
    args_schema: Type[BaseModel] = JoyDiscoverInput

    def _run(
        self,
        capability: str = "",
        query: str = "",
        min_trust_score: float = 0.5,
        limit: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Discover trusted agents."""
        try:
            params: dict[str, Any] = {"limit": limit}
            if capability:
                params["capability"] = capability
            if query:
                params["query"] = query

            with httpx.Client(timeout=30.0) as client:
                response = client.get(
                    f"{JOY_API_URL}/agents/discover",
                    params=params,
                    headers={"User-Agent": "langchain-joy/0.1.0"},
                )
                response.raise_for_status()
                data = response.json()

            agents = data.get("agents", [])

            # Filter by trust score
            trusted = [
                a
                for a in agents
                if float(a.get("trust_score", 0)) >= min_trust_score
            ]

            if not trusted:
                return f"No trusted agents found for capability='{capability}' query='{query}'"

            lines = [f"Found {len(trusted)} trusted agents:\n"]
            for agent in trusted[:limit]:
                lines.append(
                    f"- {agent.get('name', 'Unknown')} ({agent.get('id')})\n"
                    f"  Score: {agent.get('trust_score', 0):.2f}, "
                    f"Vouches: {agent.get('vouch_count', 0)}\n"
                    f"  Capabilities: {', '.join(agent.get('capabilities', [])[:3])}"
                )

            return "\n".join(lines)

        except Exception as e:
            return f"Agent discovery failed: {e}"

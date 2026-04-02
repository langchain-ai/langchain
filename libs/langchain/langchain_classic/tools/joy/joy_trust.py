"""
Joy Trust Tools for LangChain

Add AI agent trust checking to your LangChain agents.
Discover trusted tools, verify agents, and check trust scores
before executing actions.

Install: pip install langchain requests
Usage:
    from joy_tools import JoyDiscoverTool, JoyTrustCheckTool
    tools = [JoyDiscoverTool(), JoyTrustCheckTool()]
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description")
"""

import json
import requests
from typing import Optional
from langchain.tools import BaseTool

JOY_BASE = "https://joy-connect.fly.dev"


class JoyDiscoverTool(BaseTool):
    """Discover trusted AI agents by capability on the Joy network."""
    
    name: str = "joy_discover"
    description: str = (
        "Find trusted AI agents by capability. Input should be a capability name "
        "like 'code-execution', 'data-analysis', 'web-scraping', etc. "
        "Returns a list of agents with their trust scores."
    )

    def _run(self, capability: str) -> str:
        try:
            res = requests.get(
                f"{JOY_BASE}/agents/discover",
                params={"capability": capability.strip(), "limit": 10},
                timeout=10,
            )
            data = res.json()
            agents = data.get("agents", [])
            if not agents:
                return f"No agents found for capability '{capability}'."
            
            lines = [f"Found {len(agents)} agents for '{capability}':\n"]
            for a in agents:
                verified = "✅" if a.get("verified") else "❌"
                lines.append(
                    f"- {a['name']} (trust: {a.get('trust_score', 0):.1%}, "
                    f"vouches: {a.get('vouch_count', 0)}, verified: {verified})\n"
                    f"  ID: {a['id']}"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"Error discovering agents: {e}"

    async def _arun(self, capability: str) -> str:
        return self._run(capability)


class JoyTrustCheckTool(BaseTool):
    """Check if a specific AI agent is trusted on the Joy network."""
    
    name: str = "joy_trust_check"
    description: str = (
        "Check if an AI agent is trusted before calling it. "
        "Input should be the agent ID (e.g., 'ag_xxx'). "
        "Returns trust score, vouch count, and whether it's verified."
    )

    min_score: float = 0.5

    def _run(self, agent_id: str) -> str:
        try:
            res = requests.get(
                f"{JOY_BASE}/agents/{agent_id.strip()}",
                timeout=10,
            )
            if res.status_code == 404:
                return f"Agent {agent_id} not found on Joy. NOT TRUSTED."
            
            data = res.json()
            score = data.get("trust_score", 0)
            trusted = score >= self.min_score
            verified = data.get("verified", False)
            
            status = "✅ TRUSTED" if trusted else "⚠️ NOT TRUSTED"
            return (
                f"{status}\n"
                f"Name: {data.get('name', 'Unknown')}\n"
                f"Trust Score: {score:.1%}\n"
                f"Vouches: {data.get('vouch_count', 0)}\n"
                f"Verified: {'Yes' if verified else 'No'}\n"
                f"Capabilities: {', '.join(data.get('capabilities', []))}"
            )
        except Exception as e:
            return f"Error checking trust: {e}"

    async def _arun(self, agent_id: str) -> str:
        return self._run(agent_id)


class JoyVouchTool(BaseTool):
    """Vouch for an AI agent's capability on the Joy network."""
    
    name: str = "joy_vouch"
    description: str = (
        "Vouch for an AI agent after testing its capabilities. "
        "Input should be JSON with 'targetId', 'capability', and optional 'score' (1-5). "
        "Requires API key."
    )

    api_key: str = ""

    def _run(self, input_str: str) -> str:
        if not self.api_key:
            return "Error: API key required. Set api_key when creating JoyVouchTool."
        
        try:
            params = json.loads(input_str)
            res = requests.post(
                f"{JOY_BASE}/vouches",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "targetId": params["targetId"],
                    "capability": params.get("capability", "general"),
                    "score": params.get("score", 5),
                },
                timeout=10,
            )
            if res.ok:
                return f"Successfully vouched for {params['targetId']}!"
            return f"Vouch failed: {res.json().get('error', res.status_code)}"
        except json.JSONDecodeError:
            return "Error: Input must be valid JSON with 'targetId' field."
        except Exception as e:
            return f"Error vouching: {e}"

    async def _arun(self, input_str: str) -> str:
        return self._run(input_str)


class JoyNetworkStatsTool(BaseTool):
    """Get current Joy network statistics."""
    
    name: str = "joy_stats"
    description: str = "Get current Joy trust network statistics — total agents, vouches, etc."

    def _run(self, _: str = "") -> str:
        try:
            res = requests.get(f"{JOY_BASE}/stats", timeout=10)
            data = res.json()
            return (
                f"Joy Network Stats:\n"
                f"- Agents: {data.get('agents', 0):,}\n"
                f"- Online: {data.get('agentsOnline', 0):,}\n"
                f"- Vouches: {data.get('vouches', 0):,}"
            )
        except Exception as e:
            return f"Error fetching stats: {e}"

    async def _arun(self, _: str = "") -> str:
        return self._run()


# Convenience: get all Joy tools
def get_joy_tools(api_key: Optional[str] = None):
    """Get all Joy tools for your LangChain agent."""
    tools = [
        JoyDiscoverTool(),
        JoyTrustCheckTool(),
        JoyNetworkStatsTool(),
    ]
    if api_key:
        tools.append(JoyVouchTool(api_key=api_key))
    return tools

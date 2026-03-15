"""LangChain integration for Joy trust network.

Joy is a decentralized trust network where AI agents vouch for each other.
This integration enables LangChain agents to verify trustworthiness before
delegating tasks or collaborating with other agents.

Example:
    from langchain_joy import JoyTrustVerifier

    verifier = JoyTrustVerifier(min_trust_score=0.5)

    # Check if an agent should be trusted
    if verifier.should_trust("ag_xxx"):
        # Safe to delegate
        pass

    # Use as a tool in your agent
    from langchain_joy import JoyTrustTool, JoyDiscoverTool

    tools = [JoyTrustTool(), JoyDiscoverTool()]
"""

from langchain_joy.tools import JoyDiscoverTool, JoyTrustTool
from langchain_joy.verifier import JoyTrustVerifier, TrustVerificationError

__all__ = [
    "JoyTrustVerifier",
    "TrustVerificationError",
    "JoyTrustTool",
    "JoyDiscoverTool",
]

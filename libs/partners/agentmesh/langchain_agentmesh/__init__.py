"""AgentMesh trust layer integration for LangChain.

This package provides cryptographic identity verification and trust-gated
tool execution for LangChain agents.
"""

from langchain_agentmesh.identity import CMVKIdentity, CMVKSignature
from langchain_agentmesh.trust import (
    TrustedAgentCard,
    TrustHandshake,
    TrustVerificationResult,
    TrustPolicy,
    DelegationChain,
    Delegation,
)
from langchain_agentmesh.tools import TrustGatedTool, TrustedToolExecutor
from langchain_agentmesh.callbacks import TrustCallbackHandler

__all__ = [
    # Identity
    "CMVKIdentity",
    "CMVKSignature",
    # Trust
    "TrustedAgentCard",
    "TrustHandshake",
    "TrustVerificationResult",
    "TrustPolicy",
    "DelegationChain",
    "Delegation",
    # Tools
    "TrustGatedTool",
    "TrustedToolExecutor",
    # Callbacks
    "TrustCallbackHandler",
]

__version__ = "0.1.0"
